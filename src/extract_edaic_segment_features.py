from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import torchaudio
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.extract_edaic_features import (
    downsample_audio_mask,
    masked_mean_pool,
    resolve_device,
    resolve_path,
    validate_embedding,
    validate_manifest,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract timestamp-aligned segment embeddings for E-DAIC.")
    parser.add_argument("--config", required=True, help="Path to E-DAIC segment feature YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Optional output path overriding data.feature_cache_path.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_audio(audio_path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


def validate_transcript_columns(transcript_df: pd.DataFrame, data_cfg: Dict[str, Any], transcript_path: Path) -> None:
    required = [
        data_cfg["transcript_start_column"],
        data_cfg["transcript_end_column"],
        data_cfg["transcript_text_column"],
    ]
    confidence_column = data_cfg.get("transcript_confidence_column")
    if confidence_column:
        required.append(confidence_column)

    missing = [column for column in required if column not in transcript_df.columns]
    if missing:
        raise ValueError(
            f"Transcript CSV {transcript_path} is missing columns: {missing}. "
            f"Available columns: {list(transcript_df.columns)}"
        )


def build_segments(transcript_path: Path, data_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    transcript_df = pd.read_csv(transcript_path)
    validate_transcript_columns(transcript_df, data_cfg, transcript_path)

    start_column = data_cfg["transcript_start_column"]
    end_column = data_cfg["transcript_end_column"]
    text_column = data_cfg["transcript_text_column"]
    confidence_column = data_cfg.get("transcript_confidence_column")
    min_confidence = float(data_cfg.get("min_confidence", 0.0))
    segment_seconds = float(data_cfg["segment_seconds"])
    min_segment_text_chars = int(data_cfg["min_segment_text_chars"])
    max_segments = int(data_cfg["max_segments"])
    padding_seconds = float(data_cfg["audio_padding_seconds"])

    df = transcript_df.copy()
    df[start_column] = pd.to_numeric(df[start_column], errors="coerce")
    df[end_column] = pd.to_numeric(df[end_column], errors="coerce")
    df[text_column] = df[text_column].fillna("").astype(str).str.strip()
    df = df.dropna(subset=[start_column, end_column])
    df = df[df[text_column] != ""]

    if confidence_column:
        df[confidence_column] = pd.to_numeric(df[confidence_column], errors="coerce").fillna(0.0)
        df = df[df[confidence_column] >= min_confidence]

    if df.empty:
        return []

    df["window_index"] = (df[start_column] // segment_seconds).astype(int)
    segments: List[Dict[str, Any]] = []
    for _, group in df.groupby("window_index", sort=True):
        text_parts = group[text_column].dropna().astype(str).str.strip().tolist()
        text = " ".join(part for part in text_parts if part)
        if len(text) < min_segment_text_chars:
            continue

        start_time = max(0.0, float(group[start_column].min()) - padding_seconds)
        end_time = float(group[end_column].max()) + padding_seconds
        if end_time <= start_time:
            continue

        segments.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "num_utterances": int(len(group)),
            }
        )
        if len(segments) >= max_segments:
            break

    return segments


def crop_audio(waveform: torch.Tensor, sample_rate: int, start_time: float, end_time: float) -> torch.Tensor:
    start_index = max(0, int(math.floor(start_time * sample_rate)))
    end_index = min(waveform.shape[0], int(math.ceil(end_time * sample_rate)))
    if end_index <= start_index:
        return torch.zeros(1, dtype=waveform.dtype)
    return waveform[start_index:end_index]


@torch.no_grad()
def encode_text_segment(
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
def encode_audio_segment(
    waveform: torch.Tensor,
    feature_extractor: Wav2Vec2FeatureExtractor,
    audio_model: Wav2Vec2Model,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    audio_inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        padding=True,
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


def pad_segment_embeddings(embeddings: List[torch.Tensor], max_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not embeddings:
        raise ValueError("Cannot pad an empty segment embedding list")

    embedding_dim = int(embeddings[0].shape[-1])
    padded = torch.zeros(max_segments, embedding_dim, dtype=embeddings[0].dtype)
    mask = torch.zeros(max_segments, dtype=torch.float)
    for index, embedding in enumerate(embeddings[:max_segments]):
        padded[index] = embedding
        mask[index] = 1.0
    return padded, mask


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
    skipped: List[Dict[str, str]] = []
    manifest_root = manifest_path.parent
    sample_rate = int(data_cfg["sample_rate"])
    max_segments = int(data_cfg["max_segments"])

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract E-DAIC segment features"):
        participant_id = str(row[data_cfg["id_column"]])
        transcript_path = resolve_path(row[data_cfg["text_column"]], manifest_root)
        audio_path = resolve_path(row[data_cfg["audio_column"]], manifest_root)
        segments = build_segments(transcript_path, data_cfg)
        if not segments:
            skipped.append({"participant_id": participant_id, "reason": "no valid transcript segments"})
            continue

        waveform = load_audio(audio_path, sample_rate)
        text_embeddings: List[torch.Tensor] = []
        audio_embeddings: List[torch.Tensor] = []
        segment_metadata: List[Dict[str, Any]] = []
        for segment in segments:
            audio_segment = crop_audio(waveform, sample_rate, segment["start_time"], segment["end_time"])
            if audio_segment.numel() <= 1:
                continue

            text_embedding = encode_text_segment(
                text=segment["text"],
                tokenizer=tokenizer,
                text_model=text_model,
                max_text_length=int(data_cfg["max_text_length"]),
                device=device,
            )
            audio_embedding = encode_audio_segment(
                waveform=audio_segment,
                feature_extractor=feature_extractor,
                audio_model=audio_model,
                sample_rate=sample_rate,
                device=device,
            )
            validate_embedding("text", participant_id, text_embedding)
            validate_embedding("audio", participant_id, audio_embedding)
            text_embeddings.append(text_embedding)
            audio_embeddings.append(audio_embedding)
            segment_metadata.append(
                {
                    "start_time": float(segment["start_time"]),
                    "end_time": float(segment["end_time"]),
                    "num_utterances": int(segment["num_utterances"]),
                    "text_chars": int(len(segment["text"])),
                }
            )

        if not text_embeddings:
            skipped.append({"participant_id": participant_id, "reason": "no valid audio/text segment pairs"})
            continue

        padded_text, segment_mask = pad_segment_embeddings(text_embeddings, max_segments)
        padded_audio, _ = pad_segment_embeddings(audio_embeddings, max_segments)
        items.append(
            {
                "participant_id": participant_id,
                "split": str(row[data_cfg["split_column"]]).lower(),
                "phq_score": float(row[data_cfg["phq_column"]]),
                "label": int(row[data_cfg["label_column"]]),
                "text_embeddings": padded_text,
                "audio_embeddings": padded_audio,
                "segment_mask": segment_mask,
                "num_segments": int(segment_mask.sum().item()),
                "segments": segment_metadata,
            }
        )

    if not items:
        raise ValueError(f"No E-DAIC segment features were extracted. Skipped: {skipped[:10]}")

    output = Path(args.output or data_cfg["feature_cache_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    num_segments = [int(item["num_segments"]) for item in items]
    metadata = {
        "text_dim": int(items[0]["text_embeddings"].shape[-1]),
        "audio_dim": int(items[0]["audio_embeddings"].shape[-1]),
        "max_segments": max_segments,
        "num_items": len(items),
        "num_skipped": len(skipped),
        "split_label_counts": count_by_split_and_label(items),
        "segment_count_min": int(min(num_segments)),
        "segment_count_max": int(max(num_segments)),
        "segment_count_mean": float(sum(num_segments) / len(num_segments)),
        "text_model_name": str(model_cfg["text_model_name"]),
        "audio_model_name": str(model_cfg["audio_model_name"]),
        "segment_seconds": float(data_cfg["segment_seconds"]),
        "max_text_length": int(data_cfg["max_text_length"]),
        "skipped": skipped,
    }
    torch.save({"metadata": metadata, "items": items}, output)
    print(f"Wrote {output}")
    print(metadata)


if __name__ == "__main__":
    main()
