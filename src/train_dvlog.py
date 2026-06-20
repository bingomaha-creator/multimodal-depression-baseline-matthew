from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dvlog_dataset import (
    DVlogDataset,
    DVlogSample,
    FeatureNormalizer,
    collate_dvlog_pooled,
    collate_dvlog_temporal,
    discover_dvlog_samples,
    validate_dvlog_samples,
)
from src.models.dvlog_baselines import DVlogBiGRU, DVlogMLP
from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D-Vlog MLP and BiGRU baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["mlp", "bigru"])
    parser.add_argument("--modality", choices=["audio", "visual", "both"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--validate-data", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_samples(samples: Sequence[DVlogSample]) -> dict[str, list[DVlogSample]]:
    splits = {"train": [], "valid": [], "test": []}
    for sample in samples:
        splits[sample.split].append(sample)
    empty = [name for name, values in splits.items() if not values]
    if empty:
        raise ValueError(f"D-Vlog official splits are empty: {empty}")
    return splits


def compute_class_weights(labels: Sequence[int], device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Training split must contain both classes, got counts {counts.tolist()}")
    total = int(counts.sum())
    return torch.tensor(total / (2.0 * counts), dtype=torch.float32, device=device)


def is_better_checkpoint(f1: float, loss: float, best_f1: float, best_loss: float) -> bool:
    return f1 > best_f1 or (np.isclose(f1, best_f1) and loss < best_loss)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_batch(model: nn.Module, batch: dict[str, Any], model_name: str) -> torch.Tensor:
    if model_name == "mlp":
        return model(batch["audio_embeddings"], batch["visual_embeddings"])
    return model(
        batch["audio"],
        batch["visual"],
        batch["audio_lengths"],
        batch["visual_lengths"],
        batch["visual_mask"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    max_train_steps: int | None,
) -> float:
    model.train()
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_batch(model, batch, model_name)
        loss = criterion(logits, batch["labels"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if max_train_steps is not None and step >= max_train_steps:
            break
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch(batch, device)
        logits = forward_batch(model, batch, model_name)
        loss = criterion(logits, batch["labels"])
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= threshold).long()
        batch_labels = batch["labels"].cpu().tolist()
        batch_predictions = preds.cpu().tolist()
        batch_probabilities = probs.cpu().tolist()
        batch_durations = batch["duration"].cpu().tolist()
        losses.append(float(loss.item()))
        labels.extend(batch_labels)
        predictions.extend(batch_predictions)
        probabilities.extend(batch_probabilities)
        for sample_id, label, prediction, probability, gender, duration in zip(
            batch["sample_id"],
            batch_labels,
            batch_predictions,
            batch_probabilities,
            batch["gender"],
            batch_durations,
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "pred_label": prediction,
                    "prob_depressed": probability,
                    "gender": gender,
                    "duration": duration,
                }
            )
    metrics = classification_metrics(np.asarray(labels), np.asarray(predictions))
    metrics.update(
        {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "threshold": float(threshold),
            "num_samples": len(labels),
            "pred_pos_rate": float(np.mean(predictions)) if predictions else 0.0,
        }
    )
    return metrics, pd.DataFrame(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def build_model(model_name: str, modality: str, model_cfg: dict[str, Any]) -> nn.Module:
    if model_name == "mlp":
        return DVlogMLP(
            hidden_dim=int(model_cfg["classifier_dim"]),
            dropout=float(model_cfg["dropout"]),
            modality=modality,
        )
    return DVlogBiGRU(
        projection_dim=int(model_cfg["projection_dim"]),
        hidden_dim=int(model_cfg["gru_hidden_dim"]),
        classifier_dim=int(model_cfg["classifier_dim"]),
        dropout=float(model_cfg["dropout"]),
        modality=modality,
    )


def build_loader(
    samples: Sequence[DVlogSample],
    normalizer: FeatureNormalizer,
    model_name: str,
    modality: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    cache_in_memory: bool,
) -> DataLoader:
    representation = "pooled" if model_name == "mlp" else "temporal"
    dataset = DVlogDataset(
        samples,
        normalizer,
        representation=representation,
        modality=modality,
        cache_in_memory=cache_in_memory,
    )
    collate = collate_dvlog_pooled if model_name == "mlp" else collate_dvlog_temporal
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    samples = discover_dvlog_samples(config["data"]["dataset_root"])
    validation_summary = validate_dvlog_samples(samples, modality=args.modality or "both")
    print(json.dumps(validation_summary, indent=2, ensure_ascii=False))
    if args.validate_data:
        return
    if args.model is None or args.modality is None:
        raise ValueError("--model and --modality are required unless --validate-data is used")

    train_cfg = config["training"]
    model_cfg = config["model"]
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device or str(train_cfg.get("device", "auto")))
    splits = split_samples(samples)
    normalizer = FeatureNormalizer.fit(splits["train"], modality=args.modality)
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    cache_in_memory = bool(config["data"].get("cache_in_memory", True))
    loaders = {
        name: build_loader(
            values,
            normalizer,
            args.model,
            args.modality,
            batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            cache_in_memory=cache_in_memory,
        )
        for name, values in splits.items()
    }

    model = build_model(args.model, args.modality, model_cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    weights = None
    if bool(train_cfg.get("use_class_weights", True)):
        weights = compute_class_weights([sample.label for sample in splits["train"]], device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    output_root = Path(args.output_dir or train_cfg["output_dir"])
    run_dir = output_root / args.model / args.modality / f"seed_{seed}"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    max_train_steps = args.max_train_steps
    best_f1 = -1.0
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            args.model,
            max_train_steps,
        )
        valid_metrics, _ = evaluate(model, loaders["valid"], criterion, device, args.model)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['acc']:.4f} valid_f1={valid_metrics['f1']:.4f}"
        )
        if is_better_checkpoint(valid_metrics["f1"], valid_metrics["loss"], best_f1, best_loss):
            best_f1 = valid_metrics["f1"]
            best_loss = valid_metrics["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "config": config,
                    "model_name": args.model,
                    "modality": args.modality,
                    "seed": seed,
                    "epoch": epoch,
                    "valid_metrics_at_0_5": valid_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early_stopping epoch={epoch} patience={patience}")
                break

    if not checkpoint_path.is_file():
        raise RuntimeError(f"No validation-selected checkpoint was created: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    for split_name in ("valid", "test"):
        metrics, predictions = evaluate(model, loaders[split_name], criterion, device, args.model)
        save_json(run_dir / "metrics" / f"{split_name}_metrics_at_0_5.json", metrics)
        predictions_dir = run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_dir / f"{split_name}_predictions_at_0_5.csv", index=False)
        print(f"{split_name}: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
