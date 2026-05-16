from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.edaic_feature_dataset import EDAICFeatureDataset, collate_edaic_features
from src.models.edaic_feature_baseline import EDAICFeatureBaseline
from src.utils.metrics import classification_metrics, find_best_threshold
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E-DAIC feature baseline on cached embeddings.")
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
        "--use-class-weights",
        choices=["true", "false"],
        default=None,
        help="Override training.use_class_weights.",
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
    if args.use_class_weights is not None:
        config["training"]["use_class_weights"] = args.use_class_weights == "true"
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


def labels_from_dataset(dataset: EDAICFeatureDataset) -> np.ndarray:
    return np.array([int(item["label"]) for item in dataset.items], dtype=int)


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    label_list = labels.astype(int).tolist()
    num_neg = label_list.count(0)
    num_pos = label_list.count(1)
    total = num_neg + num_pos
    weights = torch.tensor(
        [
            total / (2 * max(num_neg, 1)),
            total / (2 * max(num_pos, 1)),
        ],
        dtype=torch.float,
        device=device,
    )
    print(
        f"class_weights: neg={weights[0].item():.4f}, pos={weights[1].item():.4f} "
        f"(train counts: neg={num_neg}, pos={num_pos})"
    )
    return weights


def validate_split_labels(datasets: Dict[str, EDAICFeatureDataset]) -> None:
    for split, dataset in datasets.items():
        labels = labels_from_dataset(dataset)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique.tolist(), counts.tolist()))
        print(f"{split} label counts: {label_counts}")
        if len(label_counts) < 2:
            raise ValueError(f"Split '{split}' must contain both classes, got label counts: {label_counts}")


def balanced_small_indices(labels: np.ndarray, limit: int) -> List[int]:
    if limit <= 0:
        raise ValueError("--overfit-small must be greater than 0")

    selected: List[int] = []
    for target in (0, 1):
        matches = np.where(labels == target)[0].tolist()
        if matches:
            selected.append(matches[0])

    for index in range(len(labels)):
        if len(selected) >= limit:
            break
        if index not in selected:
            selected.append(index)
    return selected[:limit]


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
    validate_split_labels(datasets)

    if overfit_small is not None:
        train_labels = labels_from_dataset(datasets["train"])
        indices = balanced_small_indices(train_labels, overfit_small)
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


def build_model(metadata: Dict[str, Any], model_cfg: Dict[str, Any], modality: str) -> EDAICFeatureBaseline:
    return EDAICFeatureBaseline(
        text_dim=int(metadata["text_dim"]),
        audio_dim=int(metadata["audio_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        modality=modality,
    )


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
        logits = model(
            text_embeddings=batch["text_embeddings"],
            audio_embeddings=batch["audio_embeddings"],
        )
        loss = criterion(logits, batch["labels"])
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
    threshold: float = 0.5,
) -> tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    losses = []
    labels = []
    predictions = []
    probabilities = []
    rows = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch_to_device(batch, device)
        logits = model(
            text_embeddings=batch["text_embeddings"],
            audio_embeddings=batch["audio_embeddings"],
        )
        loss = criterion(logits, batch["labels"])
        probs = torch.softmax(logits, dim=-1)
        preds = (probs[:, 1] >= threshold).long()

        batch_labels = batch["labels"].detach().cpu().numpy().tolist()
        batch_preds = preds.detach().cpu().numpy().tolist()
        batch_probs = probs[:, 1].detach().cpu().numpy().tolist()
        losses.append(float(loss.item()))
        labels.extend(batch_labels)
        predictions.extend(batch_preds)
        probabilities.extend(batch_probs)

        for participant_id, split, phq_score, label, pred, prob in zip(
            batch["participant_id"],
            batch["split"],
            batch["phq_scores"].detach().cpu().numpy().tolist(),
            batch_labels,
            batch_preds,
            batch_probs,
        ):
            rows.append(
                {
                    "participant_id": participant_id,
                    "split": split,
                    "phq_score": phq_score,
                    "label": label,
                    "pred_label": pred,
                    "prob_depressed": prob,
                }
            )

    metrics = classification_metrics(np.array(labels), np.array(predictions))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["threshold"] = float(threshold)
    metrics["prob_min"] = float(np.min(probabilities)) if probabilities else 0.0
    metrics["prob_max"] = float(np.max(probabilities)) if probabilities else 0.0
    metrics["prob_mean"] = float(np.mean(probabilities)) if probabilities else 0.0
    metrics["pred_pos_rate"] = float(np.mean(predictions)) if predictions else 0.0
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
    train_labels = labels_from_dataset(loaders["train"].dataset)
    model = build_model(loaders["train"].dataset.metadata, model_cfg, modality).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    if bool(train_cfg.get("use_class_weights", True)):
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels, device))
    else:
        criterion = nn.CrossEntropyLoss()

    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = train_cfg.get("max_train_steps")

    best_metric = -1.0
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
        dev_metrics, _ = evaluate(model, loaders["dev"], criterion, device, threshold=0.5)
        dev_metrics["train_loss"] = train_loss
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"dev_acc={dev_metrics['acc']:.4f} dev_precision={dev_metrics['precision']:.4f} "
            f"dev_recall={dev_metrics['recall']:.4f} dev_f1={dev_metrics['f1']:.4f} "
            f"pred_pos_rate={dev_metrics['pred_pos_rate']:.4f} "
            f"prob_min={dev_metrics['prob_min']:.4f} prob_max={dev_metrics['prob_max']:.4f} "
            f"prob_mean={dev_metrics['prob_mean']:.4f}"
        )

        monitor_metric = str(train_cfg["monitor_metric"])
        current_metric = float(dev_metrics[monitor_metric])
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = dev_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "modality": modality,
                    "metrics_at_0_5": dev_metrics,
                    "metadata": loaders["train"].dataset.metadata,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dev_metrics_at_0_5, dev_predictions_at_0_5 = evaluate(model, loaders["dev"], criterion, device, threshold=0.5)
    test_metrics_at_0_5, test_predictions_at_0_5 = evaluate(model, loaders["test"], criterion, device, threshold=0.5)

    dev_labels = dev_predictions_at_0_5["label"].to_numpy()
    dev_probabilities = dev_predictions_at_0_5["prob_depressed"].to_numpy()
    best_threshold, dev_threshold_metrics = find_best_threshold(dev_labels, dev_probabilities)
    dev_metrics_with_threshold, dev_predictions_with_threshold = evaluate(
        model,
        loaders["dev"],
        criterion,
        device,
        threshold=best_threshold,
    )
    test_metrics_with_threshold, test_predictions_with_threshold = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        threshold=best_threshold,
    )
    dev_metrics_with_threshold["best_threshold"] = best_threshold
    dev_metrics_with_threshold["threshold_search_metrics"] = dev_threshold_metrics
    test_metrics_with_threshold["dev_best_threshold"] = best_threshold

    dev_predictions_at_0_5.to_csv(predictions_dir / "dev_predictions_at_0_5.csv", index=False)
    test_predictions_at_0_5.to_csv(predictions_dir / "test_predictions_at_0_5.csv", index=False)
    dev_predictions_with_threshold.to_csv(predictions_dir / "dev_predictions_with_threshold.csv", index=False)
    test_predictions_with_threshold.to_csv(predictions_dir / "test_predictions_with_threshold.csv", index=False)

    save_json(metrics_dir / "best_epoch_metrics.json", best_metrics)
    save_json(metrics_dir / "dev_metrics_at_0_5.json", dev_metrics_at_0_5)
    save_json(metrics_dir / "test_metrics_at_0_5.json", test_metrics_at_0_5)
    save_json(metrics_dir / "dev_metrics_with_threshold.json", dev_metrics_with_threshold)
    save_json(metrics_dir / "test_metrics_with_threshold.json", test_metrics_with_threshold)

    checkpoint["metrics_at_0_5"] = dev_metrics_at_0_5
    checkpoint["best_threshold"] = best_threshold
    checkpoint["metrics_with_threshold"] = dev_metrics_with_threshold
    torch.save(checkpoint, best_path)

    print("Best dev metrics at threshold 0.5:")
    print(json.dumps(dev_metrics_at_0_5, indent=2, ensure_ascii=False))
    print("Test metrics at threshold 0.5:")
    print(json.dumps(test_metrics_at_0_5, indent=2, ensure_ascii=False))
    print("Test metrics with dev best threshold:")
    print(json.dumps(test_metrics_with_threshold, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
