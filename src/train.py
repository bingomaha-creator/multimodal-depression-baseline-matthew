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
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from src.datasets.edaic_dataset import EDAICDataset, collate_batch
from src.models.multimodal_baseline import MultimodalBaseline
from src.utils.metrics import classification_metrics, find_best_threshold
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoBERTa + wav2vec2 baseline on E-DAIC.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Override config smoke-test step limit.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def build_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model_name"])
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_cfg["audio_model_name"])

    loaders = {}
    for split in ("train", "dev", "test"):
        dataset = EDAICDataset(
            manifest_path=data_cfg["manifest_path"],
            split=split,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            config=data_cfg,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(data_cfg["num_workers"]),
            collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    dataset = loader.dataset
    data_cfg = dataset.config
    labels = (
        dataset.df[data_cfg["phq_column"]]
        .astype(float)
        .ge(float(data_cfg["positive_threshold"]))
        .astype(int)
        .tolist()
    )

    num_neg = labels.count(0)
    num_pos = labels.count(1)
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
            input_ids=batch["input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            audio_values=batch["audio_values"],
            audio_attention_mask=batch["audio_attention_mask"],
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
            input_ids=batch["input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            audio_values=batch["audio_values"],
            audio_attention_mask=batch["audio_attention_mask"],
        )
        loss = criterion(logits, batch["labels"])
        probs = torch.softmax(logits, dim=-1)
        preds = (probs[:, 1] >= threshold).long()

        losses.append(float(loss.item()))
        labels.extend(batch["labels"].detach().cpu().numpy().tolist())
        predictions.extend(preds.detach().cpu().numpy().tolist())
        probabilities.extend(probs[:, 1].detach().cpu().numpy().tolist())

        for participant_id, phq_score, label, pred, prob in zip(
            batch["participant_id"],
            batch["phq_scores"].detach().cpu().numpy().tolist(),
            batch["labels"].detach().cpu().numpy().tolist(),
            preds.detach().cpu().numpy().tolist(),
            probs[:, 1].detach().cpu().numpy().tolist(),
        ):
            rows.append(
                {
                    "participant_id": participant_id,
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
    return metrics, pd.DataFrame(rows)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    device = resolve_device(str(config["training"]["device"]))
    output_dir = Path(config["training"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(config)
    model = MultimodalBaseline(**config["model"]).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if bool(config["training"].get("use_class_weights", True)):
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(loaders["train"], device))
    else:
        criterion = nn.CrossEntropyLoss()

    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = config["training"].get("max_train_steps")

    best_metric = -1.0
    best_metrics: Dict[str, float] = {}
    best_path = checkpoint_dir / "best.pt"

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
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
            f"dev_threshold=0.50"
        )

        monitor_metric = str(config["training"]["monitor_metric"])
        current_metric = float(dev_metrics[monitor_metric])
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = dev_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "dev_metrics": dev_metrics,
                },
                best_path,
            )

    save_json(metrics_dir / "dev_best_metrics.json", best_metrics)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dev_metrics_at_0_5, dev_predictions = evaluate(model, loaders["dev"], criterion, device, threshold=0.5)
    dev_labels = dev_predictions["label"].to_numpy()
    dev_probabilities = dev_predictions["prob_depressed"].to_numpy()
    dev_best_threshold, _ = find_best_threshold(dev_labels, dev_probabilities)
    dev_metrics_best_threshold, dev_predictions_best_threshold = evaluate(
        model,
        loaders["dev"],
        criterion,
        device,
        threshold=dev_best_threshold,
    )
    dev_metrics_best_threshold["dev_best_threshold"] = dev_best_threshold
    dev_predictions.to_csv(predictions_dir / "dev_predictions.csv", index=False)
    dev_predictions_best_threshold.to_csv(
        predictions_dir / "dev_predictions_with_best_threshold.csv",
        index=False,
    )
    save_json(metrics_dir / "dev_metrics_at_0_5.json", dev_metrics_at_0_5)
    save_json(metrics_dir / "dev_metrics_best_threshold.json", dev_metrics_best_threshold)

    checkpoint["dev_best_threshold"] = dev_best_threshold
    torch.save(checkpoint, best_path)

    test_metrics_at_0_5, test_predictions = evaluate(model, loaders["test"], criterion, device, threshold=0.5)
    test_metrics_with_dev_threshold, test_predictions_with_dev_threshold = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        threshold=dev_best_threshold,
    )
    test_metrics_with_dev_threshold["dev_best_threshold"] = dev_best_threshold
    save_json(metrics_dir / "test_metrics_at_0_5.json", test_metrics_at_0_5)
    save_json(metrics_dir / "test_metrics_with_dev_threshold.json", test_metrics_with_dev_threshold)
    test_predictions.to_csv(predictions_dir / "test_predictions.csv", index=False)
    test_predictions_with_dev_threshold.to_csv(
        predictions_dir / "test_predictions_with_dev_threshold.csv",
        index=False,
    )

    print(
        f"test@0.5 acc={test_metrics_at_0_5['acc']:.4f} "
        f"precision={test_metrics_at_0_5['precision']:.4f} "
        f"recall={test_metrics_at_0_5['recall']:.4f} "
        f"f1={test_metrics_at_0_5['f1']:.4f}"
    )
    print(
        f"test@dev_threshold acc={test_metrics_with_dev_threshold['acc']:.4f} "
        f"precision={test_metrics_with_dev_threshold['precision']:.4f} "
        f"recall={test_metrics_with_dev_threshold['recall']:.4f} "
        f"f1={test_metrics_with_dev_threshold['f1']:.4f} "
        f"threshold={dev_best_threshold:.2f}"
    )
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
