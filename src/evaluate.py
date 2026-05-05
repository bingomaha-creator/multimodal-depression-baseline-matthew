from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from src.datasets.edaic_dataset import EDAICDataset, collate_batch
from src.models.multimodal_baseline import MultimodalBaseline
from src.train import evaluate, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved multimodal baseline checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"], help="Split to evaluate.")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = resolve_device(str(config["training"]["device"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_model_name"])
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model"]["audio_model_name"])
    dataset = EDAICDataset(
        manifest_path=config["data"]["manifest_path"],
        split=args.split,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        config=config["data"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultimodalBaseline(**config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("dev_best_threshold", checkpoint.get("best_threshold", 0.5)))

    metrics, predictions = evaluate(model, loader, nn.CrossEntropyLoss(), device, threshold=threshold)
    metrics["threshold"] = threshold
    output_dir = Path(config["training"]["output_dir"])
    save_json(output_dir / "metrics" / f"{args.split}_metrics.json", metrics)
    predictions_path = output_dir / "predictions" / f"{args.split}_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
