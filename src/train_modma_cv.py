from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.modma_feature_dataset import MODMAFeatureDataset, collate_modma_features
from src.models.modma_feature_baseline import MODMAFeatureBaseline
from src.utils.metrics import classification_metrics, find_best_threshold
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MODMA feature baseline with stratified cross-validation.")
    parser.add_argument("--config", required=True, help="Path to MODMA YAML config.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional train step limit for smoke tests.")
    parser.add_argument("--fold-limit", type=int, default=None, help="Optional fold limit for smoke tests.")
    parser.add_argument("--modality", choices=["text", "audio", "both"], default=None, help="Override model modality.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def compute_class_weights(labels: np.ndarray, train_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    train_labels = labels[train_indices].tolist()
    num_neg = train_labels.count(0)
    num_pos = train_labels.count(1)
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

        losses.append(float(loss.item()))
        labels.extend(batch["labels"].detach().cpu().numpy().tolist())
        predictions.extend(preds.detach().cpu().numpy().tolist())
        probabilities.extend(probs[:, 1].detach().cpu().numpy().tolist())
        for participant_id, label, pred, prob in zip(
            batch["participant_id"],
            batch["labels"].detach().cpu().numpy().tolist(),
            preds.detach().cpu().numpy().tolist(),
            probs[:, 1].detach().cpu().numpy().tolist(),
        ):
            rows.append(
                {
                    "participant_id": participant_id,
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


def build_model(metadata: Dict[str, Any], model_cfg: Dict[str, Any], modality: str) -> MODMAFeatureBaseline:
    return MODMAFeatureBaseline(
        text_dim=int(metadata["text_dim"]),
        audio_dim=int(metadata["audio_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        modality=modality,
    )


def summarize_cv(fold_metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"folds": fold_metrics}
    for key in ("acc", "precision", "recall", "f1", "loss"):
        values = np.array([fold["metrics_at_0_5"][key] for fold in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def validate_cv_labels(labels: np.ndarray, n_splits: int) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    if len(label_counts) < 2:
        raise ValueError(f"Cross-validation requires both classes, got label counts: {label_counts}")
    smallest_class = int(counts.min())
    if smallest_class < n_splits:
        raise ValueError(
            f"n_splits={n_splits} is larger than the smallest class count={smallest_class}. "
            f"Label counts: {label_counts}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    modality = args.modality or str(model_cfg.get("modality", "both"))
    device = resolve_device(str(train_cfg["device"]))
    output_dir = Path(train_cfg["output_dir"]) / modality
    output_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = MODMAFeatureDataset(data_cfg["feature_cache_path"])
    labels = np.array([int(item["label"]) for item in full_dataset.items])
    participants = [item["participant_id"] for item in full_dataset.items]
    print("Label counts:", dict(zip(*np.unique(labels, return_counts=True))))
    print(f"Training modality: {modality}")
    validate_cv_labels(labels, int(train_cfg["n_splits"]))

    splitter = StratifiedKFold(
        n_splits=int(train_cfg["n_splits"]),
        shuffle=True,
        random_state=int(config["seed"]),
    )

    fold_metrics = []
    for fold_idx, (train_indices, valid_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        if args.fold_limit is not None and fold_idx > args.fold_limit:
            break

        fold_dir = output_dir / f"fold_{fold_idx}"
        checkpoint_dir = fold_dir / "checkpoints"
        metrics_dir = fold_dir / "metrics"
        predictions_dir = fold_dir / "predictions"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"fold={fold_idx} train={len(train_indices)} valid={len(valid_indices)} "
            f"valid_participants={participants[valid_indices[0]]}..."
        )
        train_dataset = MODMAFeatureDataset(data_cfg["feature_cache_path"], train_indices)
        valid_dataset = MODMAFeatureDataset(data_cfg["feature_cache_path"], valid_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg["num_workers"]),
            collate_fn=collate_modma_features,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            collate_fn=collate_modma_features,
        )

        model = build_model(full_dataset.metadata, model_cfg, modality).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        if bool(train_cfg.get("use_class_weights", True)):
            criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels, train_indices, device))
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
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                max_train_steps=max_train_steps,
            )
            valid_metrics, _ = evaluate(model, valid_loader, criterion, device, threshold=0.5)
            valid_metrics["train_loss"] = train_loss
            print(
                f"fold={fold_idx} epoch={epoch} train_loss={train_loss:.4f} "
                f"valid_acc={valid_metrics['acc']:.4f} valid_precision={valid_metrics['precision']:.4f} "
                f"valid_recall={valid_metrics['recall']:.4f} valid_f1={valid_metrics['f1']:.4f} "
                f"pred_pos_rate={valid_metrics['pred_pos_rate']:.4f}"
            )

            monitor_metric = str(train_cfg["monitor_metric"])
            current_metric = float(valid_metrics[monitor_metric])
            if current_metric > best_metric:
                best_metric = current_metric
                best_metrics = valid_metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "fold": fold_idx,
                        "modality": modality,
                        "valid_indices": valid_indices.tolist(),
                        "metrics_at_0_5": valid_metrics,
                        "metadata": full_dataset.metadata,
                    },
                    best_path,
                )

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics_at_0_5, predictions_at_0_5 = evaluate(model, valid_loader, criterion, device, threshold=0.5)
        labels_valid = predictions_at_0_5["label"].to_numpy()
        probs_valid = predictions_at_0_5["prob_depressed"].to_numpy()
        best_threshold, _ = find_best_threshold(labels_valid, probs_valid)
        metrics_with_threshold, predictions_with_threshold = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            threshold=best_threshold,
        )
        metrics_with_threshold["best_threshold"] = best_threshold

        predictions_at_0_5.to_csv(predictions_dir / "valid_predictions_at_0_5.csv", index=False)
        predictions_with_threshold.to_csv(predictions_dir / "valid_predictions_with_threshold.csv", index=False)
        save_json(metrics_dir / "valid_metrics_at_0_5.json", metrics_at_0_5)
        save_json(metrics_dir / "valid_metrics_with_threshold.json", metrics_with_threshold)
        save_json(metrics_dir / "best_epoch_metrics.json", best_metrics)

        checkpoint["best_threshold"] = best_threshold
        checkpoint["metrics_at_0_5"] = metrics_at_0_5
        checkpoint["metrics_with_threshold"] = metrics_with_threshold
        torch.save(checkpoint, best_path)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "metrics_at_0_5": metrics_at_0_5,
                "metrics_with_threshold": metrics_with_threshold,
            }
        )

    summary = summarize_cv(fold_metrics)
    save_json(output_dir / "metrics" / "cv_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
