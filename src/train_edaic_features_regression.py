from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.edaic_feature_dataset import EDAICFeatureDataset, collate_edaic_features
from src.models.edaic_feature_regression_baseline import EDAICFeatureRegressionBaseline
from src.utils.metrics import regression_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E-DAIC feature regression baseline on cached embeddings.")
    parser.add_argument("--config", required=True, help="Path to E-DAIC feature YAML config.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional train step limit for smoke tests.")
    parser.add_argument("--modality", choices=["text", "audio", "both"], default=None, help="Override model modality.")
    parser.add_argument("--overfit-small", type=int, default=None, help="Train/evaluate on N train items for debugging.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed for stability runs.")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override training.learning_rate.")
    parser.add_argument("--dropout", type=float, default=None, help="Override model.dropout.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override model.hidden_dim.")
    parser.add_argument(
        "--loss",
        choices=["smooth_l1", "mse"],
        default=None,
        help="Override training.regression_loss.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = float(args.learning_rate)
    if args.dropout is not None:
        config["model"]["dropout"] = float(args.dropout)
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = int(args.hidden_dim)
    if args.loss is not None:
        config["training"]["regression_loss"] = args.loss
    return config


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def phq_scores_from_dataset(dataset: EDAICFeatureDataset) -> np.ndarray:
    return np.array([float(item["phq_score"]) for item in dataset.items], dtype=float)


def validate_split_scores(datasets: Dict[str, EDAICFeatureDataset]) -> None:
    for split, dataset in datasets.items():
        scores = phq_scores_from_dataset(dataset)
        print(
            f"{split} PHQ score stats: "
            f"count={len(scores)} min={scores.min():.2f} max={scores.max():.2f} mean={scores.mean():.2f}"
        )


def build_dataloaders(
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    overfit_small: Optional[int] = None,
) -> Dict[str, DataLoader]:
    feature_cache_path = data_cfg["feature_cache_path"]
    datasets = {
        "train": EDAICFeatureDataset(feature_cache_path, split="train"),
        "dev": EDAICFeatureDataset(feature_cache_path, split="dev"),
        "test": EDAICFeatureDataset(feature_cache_path, split="test"),
    }
    validate_split_scores(datasets)

    if overfit_small is not None:
        if overfit_small <= 0:
            raise ValueError("--overfit-small must be greater than 0")
        indices = list(range(min(overfit_small, len(datasets["train"]))))
        datasets = {
            split: EDAICFeatureDataset(feature_cache_path, indices=indices, split="train")
            for split in ("train", "dev", "test")
        }
        print(f"Overfit-small mode: using {len(indices)} train items for train/dev/test")

    return {
        split: DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(data_cfg["num_workers"]),
            collate_fn=collate_edaic_features,
        )
        for split, dataset in datasets.items()
    }


def build_model(
    metadata: Dict[str, Any],
    model_cfg: Dict[str, Any],
    modality: str,
) -> EDAICFeatureRegressionBaseline:
    return EDAICFeatureRegressionBaseline(
        text_dim=int(metadata["text_dim"]),
        audio_dim=int(metadata["audio_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        modality=modality,
    )


def build_criterion(train_cfg: Dict[str, Any]) -> nn.Module:
    loss_name = str(train_cfg.get("regression_loss", "smooth_l1")).lower()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError("training.regression_loss must be one of: smooth_l1, mse")


def is_better_metric(metric_name: str, current: float, best: Optional[float]) -> bool:
    if best is None:
        return True
    if metric_name == "ccc":
        return current > best
    return current < best


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_train_steps: Optional[int] = None,
) -> float:
    model.train()
    running_loss = 0.0
    step_count = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            text_embeddings=batch["text_embeddings"],
            audio_embeddings=batch["audio_embeddings"],
        )
        loss = criterion(predictions, batch["phq_scores"])
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        step_count += 1
        progress.set_postfix(loss=running_loss / step_count)
        if max_train_steps is not None and step_count >= max_train_steps:
            break

    return running_loss / max(step_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    losses = []
    targets = []
    predictions = []
    rows = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch_to_device(batch, device)
        pred_scores = model(
            text_embeddings=batch["text_embeddings"],
            audio_embeddings=batch["audio_embeddings"],
        )
        loss = criterion(pred_scores, batch["phq_scores"])

        batch_targets = batch["phq_scores"].detach().cpu().numpy().tolist()
        batch_predictions = pred_scores.detach().cpu().numpy().tolist()
        losses.append(float(loss.item()))
        targets.extend(batch_targets)
        predictions.extend(batch_predictions)

        for participant_id, split, phq_score, label, pred_score in zip(
            batch["participant_id"],
            batch["split"],
            batch_targets,
            batch["labels"].detach().cpu().numpy().tolist(),
            batch_predictions,
        ):
            rows.append(
                {
                    "participant_id": participant_id,
                    "split": split,
                    "phq_score": phq_score,
                    "label": label,
                    "pred_phq_score": pred_score,
                    "abs_error": abs(pred_score - phq_score),
                }
            )

    metrics = regression_metrics(np.array(targets), np.array(predictions))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["pred_min"] = float(np.min(predictions)) if predictions else 0.0
    metrics["pred_max"] = float(np.max(predictions)) if predictions else 0.0
    metrics["pred_mean"] = float(np.mean(predictions)) if predictions else 0.0
    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    modality = args.modality or str(model_cfg.get("modality", "both"))
    device = resolve_device(str(train_cfg["device"]))

    output_dir = Path(train_cfg["output_dir"]) / modality
    if args.overfit_small is not None:
        output_dir = output_dir / f"overfit_small_{args.overfit_small}"
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(data_cfg, train_cfg, overfit_small=args.overfit_small)
    model = build_model(loaders["train"].dataset.metadata, model_cfg, modality).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = build_criterion(train_cfg)

    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = train_cfg.get("max_train_steps")

    monitor_metric = str(train_cfg.get("monitor_metric", "mae"))
    if monitor_metric not in {"mae", "rmse", "ccc", "loss"}:
        raise ValueError("training.monitor_metric must be one of: mae, rmse, ccc, loss")

    best_metric: Optional[float] = None
    best_metrics: Dict[str, float] = {}
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_train_steps=max_train_steps,
        )
        dev_metrics, _ = evaluate(model, loaders["dev"], criterion, device)
        dev_metrics["train_loss"] = train_loss
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"dev_mae={dev_metrics['mae']:.4f} dev_rmse={dev_metrics['rmse']:.4f} "
            f"dev_ccc={dev_metrics['ccc']:.4f} pred_mean={dev_metrics['pred_mean']:.4f}"
        )

        current_metric = float(dev_metrics[monitor_metric])
        if is_better_metric(monitor_metric, current_metric, best_metric):
            best_metric = current_metric
            best_metrics = dev_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "modality": modality,
                    "monitor_metric": monitor_metric,
                    "metrics": dev_metrics,
                    "metadata": loaders["train"].dataset.metadata,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dev_metrics, dev_predictions = evaluate(model, loaders["dev"], criterion, device)
    test_metrics, test_predictions = evaluate(model, loaders["test"], criterion, device)

    dev_predictions.to_csv(predictions_dir / "dev_predictions.csv", index=False)
    test_predictions.to_csv(predictions_dir / "test_predictions.csv", index=False)

    save_json(metrics_dir / "best_epoch_metrics.json", best_metrics)
    save_json(metrics_dir / "dev_metrics.json", dev_metrics)
    save_json(metrics_dir / "test_metrics.json", test_metrics)

    checkpoint["dev_metrics"] = dev_metrics
    checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, best_path)

    print("Best dev metrics:")
    print(json.dumps(dev_metrics, indent=2, ensure_ascii=False))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
