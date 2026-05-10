from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import torchaudio
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen text/audio embeddings for E-DAIC.")
    parser.add_argument("--config", required=True, help="Path to E-DAIC feature YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Optional output path overriding data.feature_cache_path.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def resolve_path(path_value: Any, manifest_root: Path) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = manifest_root / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_transcript_text(transcript_path: Path, text_column: str) -> str:
    if transcript_path.suffix.lower() != ".csv":
        return transcript_path.read_text(encoding="utf-8", errors="ignore")

    transcript_df = pd.read_csv(transcript_path)
    if text_column not in transcript_df.columns:
        raise ValueError(
            f"Transcript CSV {transcript_path} is missing text column '{text_column}'. "
            f"Available columns: {list(transcript_df.columns)}"
        )
    return " ".join(
        transcript_df[text_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )


def load_audio(audio_path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


def downsample_audio_mask(attention_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    mask = attention_mask.unsqueeze(1).float()
    mask = torch.nn.functional.interpolate(mask, size=target_length, mode="nearest")
    return mask.squeeze(1).to(dtype=torch.long)


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def validate_manifest(df: pd.DataFrame, data_cfg: Dict[str, Any]) -> None:
    required = [
        data_cfg["id_column"],
        data_cfg["label_column"],
        data_cfg["phq_column"],
        data_cfg["split_column"],
        data_cfg["text_column"],
        data_cfg["audio_column"],
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"E-DAIC manifest is missing columns: {missing}")

    labels = set(df[data_cfg["label_column"]].dropna().astype(int).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, got: {sorted(labels)}")

    splits = set(df[data_cfg["split_column"]].dropna().astype(str).str.lower().tolist())
    required_splits = {"train", "dev", "test"}
    missing_splits = sorted(required_splits - splits)
    if missing_splits:
        raise ValueError(f"E-DAIC manifest is missing splits: {missing_splits}")


def validate_embedding(name: str, participant_id: str, embedding: torch.Tensor) -> None:
    if torch.isnan(embedding).any() or torch.isinf(embedding).any():
        raise ValueError(f"{name} embedding for {participant_id} contains NaN or Inf")
    if torch.allclose(embedding, torch.zeros_like(embedding)):
        raise ValueError(f"{name} embedding for {participant_id} is all zeros")


@torch.no_grad()
def encode_text(
    text: str,
    tokenizer: AutoTokenizer,
    text_model: AutoModel,
    max_text_length: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenizer(
        text,
        max_length=max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu()


@torch.no_grad()
def encode_audio(
    audio_path: Path,
    feature_extractor: Wav2Vec2FeatureExtractor,
    audio_model: Wav2Vec2Model,
    sample_rate: int,
    max_audio_length: int,
    device: torch.device,
) -> torch.Tensor:
    waveform = load_audio(audio_path, sample_rate)
    audio_inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        max_length=max_audio_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    audio_values = audio_inputs["input_values"].to(device)
    audio_attention_mask = audio_inputs["attention_mask"].to(device)
    outputs = audio_model(input_values=audio_values, attention_mask=audio_attention_mask)
    pooled = masked_mean_pool(
        outputs.last_hidden_state,
        downsample_audio_mask(audio_attention_mask, outputs.last_hidden_state.shape[1]),
    )
    return pooled.squeeze(0).detach().cpu()


def count_by_split_and_label(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for item in items:
        split = str(item["split"])
        label = str(int(item["label"]))
        counts.setdefault(split, {"0": 0, "1": 0})
        counts[split][label] += 1
    return counts


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    model_cfg = config["model"]
    device = resolve_device(str(config["training"]["device"]))

    manifest_path = Path(data_cfg["manifest_path"])
    manifest = pd.read_csv(manifest_path, dtype={data_cfg["id_column"]: str})
    validate_manifest(manifest, data_cfg)
    manifest[data_cfg["split_column"]] = manifest[data_cfg["split_column"]].astype(str).str.lower()
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model_name"])
    text_model = AutoModel.from_pretrained(model_cfg["text_model_name"]).to(device).eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_cfg["audio_model_name"])
    audio_model = Wav2Vec2Model.from_pretrained(model_cfg["audio_model_name"]).to(device).eval()

    items: List[Dict[str, Any]] = []
    manifest_root = manifest_path.parent
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract E-DAIC features"):
        participant_id = str(row[data_cfg["id_column"]])
        transcript_path = resolve_path(row[data_cfg["text_column"]], manifest_root)
        audio_path = resolve_path(row[data_cfg["audio_column"]], manifest_root)
        text = load_transcript_text(transcript_path, str(data_cfg.get("transcript_text_column", "Text")))

        text_embedding = encode_text(
            text=text,
            tokenizer=tokenizer,
            text_model=text_model,
            max_text_length=int(data_cfg["max_text_length"]),
            device=device,
        )
        audio_embedding = encode_audio(
            audio_path=audio_path,
            feature_extractor=feature_extractor,
            audio_model=audio_model,
            sample_rate=int(data_cfg["sample_rate"]),
            max_audio_length=int(data_cfg["max_audio_length"]),
            device=device,
        )
        validate_embedding("text", participant_id, text_embedding)
        validate_embedding("audio", participant_id, audio_embedding)

        items.append(
            {
                "participant_id": participant_id,
                "split": str(row[data_cfg["split_column"]]).lower(),
                "phq_score": float(row[data_cfg["phq_column"]]),
                "label": int(row[data_cfg["label_column"]]),
                "text_embedding": text_embedding,
                "audio_embedding": audio_embedding,
            }
        )

    output = Path(args.output or data_cfg["feature_cache_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "text_dim": int(items[0]["text_embedding"].shape[-1]) if items else 0,
        "audio_dim": int(items[0]["audio_embedding"].shape[-1]) if items else 0,
        "num_items": len(items),
        "split_label_counts": count_by_split_and_label(items),
        "text_model_name": str(model_cfg["text_model_name"]),
        "audio_model_name": str(model_cfg["audio_model_name"]),
        "max_audio_length": int(data_cfg["max_audio_length"]),
        "max_text_length": int(data_cfg["max_text_length"]),
    }
    torch.save({"metadata": metadata, "items": items}, output)
    print(f"Wrote {output}")
    print(metadata)


if __name__ == "__main__":
    main()
