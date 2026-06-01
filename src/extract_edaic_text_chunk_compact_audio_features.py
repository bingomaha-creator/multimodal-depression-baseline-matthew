from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.extract_edaic_compact_speech_features import (
    encode_audio_waveform,
    load_transcript_pieces,
    make_audio_waveform_for_chunk,
)
from src.extract_edaic_features import (
    count_by_split_and_label,
    load_audio,
    load_transcript_text,
    resolve_device,
    resolve_path,
    validate_embedding,
    validate_manifest,
)
from src.extract_edaic_text_chunk_features import encode_text_chunks
from src.utils.compact_speech import build_compact_audio_chunks, filter_transcript_pieces
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text-chunk and compact-audio embeddings for E-DAIC.")
    parser.add_argument("--config", required=True, help="Path to E-DAIC text-chunk compact-audio YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Optional output path overriding data.feature_cache_path.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def add_counts(total: Dict[str, int], counts: Dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + int(value)


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

    sample_rate = int(data_cfg["sample_rate"])
    max_audio_length = int(round(float(data_cfg["max_audio_chunk_seconds"]) * sample_rate))
    min_audio_chunk_seconds = float(data_cfg.get("min_audio_chunk_seconds", 0.0))

    items: List[Dict[str, Any]] = []
    manifest_root = manifest_path.parent
    text_chunk_counts: List[int] = []
    audio_chunk_counts: List[int] = []
    valid_row_counts: List[int] = []
    compact_seconds: List[float] = []
    original_seconds: List[float] = []
    filtered_reason_counts: Dict[str, int] = {}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract E-DAIC text-chunk compact-audio features"):
        participant_id = str(row[data_cfg["id_column"]])
        transcript_path = resolve_path(row[data_cfg["text_column"]], manifest_root)
        audio_path = resolve_path(row[data_cfg["audio_column"]], manifest_root)

        full_text = load_transcript_text(transcript_path, str(data_cfg.get("transcript_text_column", "Text")))
        text_embedding, num_text_chunks = encode_text_chunks(
            text=full_text,
            tokenizer=tokenizer,
            text_model=text_model,
            max_text_length=int(data_cfg["max_text_length"]),
            max_text_chunks=int(data_cfg["max_text_chunks"]),
            device=device,
        )

        waveform = load_audio(audio_path, sample_rate)
        audio_duration_seconds = float(waveform.numel()) / sample_rate
        original_seconds.append(audio_duration_seconds)

        pieces = load_transcript_pieces(transcript_path, data_cfg)
        valid_pieces, filter_counts = filter_transcript_pieces(
            pieces,
            max_raw_segment_seconds=float(data_cfg["max_raw_segment_seconds"]),
            audio_duration_seconds=audio_duration_seconds,
        )
        add_counts(filtered_reason_counts, filter_counts)

        audio_chunks = build_compact_audio_chunks(
            valid_pieces,
            max_audio_chunk_seconds=float(data_cfg["max_audio_chunk_seconds"]),
            max_chunks=int(data_cfg["max_audio_chunks"]),
        )
        audio_chunks = [chunk for chunk in audio_chunks if chunk.audio_seconds >= min_audio_chunk_seconds]
        if not audio_chunks:
            raise ValueError(f"No valid compact audio chunks for participant {participant_id}")

        audio_embeddings = []
        for chunk in audio_chunks:
            chunk_waveform = make_audio_waveform_for_chunk(chunk, waveform, sample_rate)
            audio_embeddings.append(
                encode_audio_waveform(
                    waveform=chunk_waveform,
                    feature_extractor=feature_extractor,
                    audio_model=audio_model,
                    sample_rate=sample_rate,
                    max_audio_length=max_audio_length,
                    device=device,
                )
            )
        audio_embedding = torch.stack(audio_embeddings).mean(dim=0)

        validate_embedding("text", participant_id, text_embedding)
        validate_embedding("audio", participant_id, audio_embedding)

        text_chunk_counts.append(num_text_chunks)
        audio_chunk_counts.append(len(audio_chunks))
        valid_row_counts.append(len(valid_pieces))
        compact_seconds.append(sum(chunk.audio_seconds for chunk in audio_chunks))
        items.append(
            {
                "participant_id": participant_id,
                "split": str(row[data_cfg["split_column"]]).lower(),
                "phq_score": float(row[data_cfg["phq_column"]]),
                "label": int(row[data_cfg["label_column"]]),
                "text_embedding": text_embedding,
                "audio_embedding": audio_embedding,
                "num_text_chunks": num_text_chunks,
                "num_compact_audio_chunks": len(audio_chunks),
                "num_valid_transcript_rows": len(valid_pieces),
                "num_filtered_rows": len(pieces) - len(valid_pieces),
                "compact_audio_seconds": float(compact_seconds[-1]),
                "original_audio_seconds": audio_duration_seconds,
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
        "max_text_length": int(data_cfg["max_text_length"]),
        "max_text_chunks": int(data_cfg["max_text_chunks"]),
        "max_raw_segment_seconds": float(data_cfg["max_raw_segment_seconds"]),
        "max_audio_chunk_seconds": float(data_cfg["max_audio_chunk_seconds"]),
        "min_audio_chunk_seconds": min_audio_chunk_seconds,
        "max_audio_chunks": int(data_cfg["max_audio_chunks"]),
        "filtered_reason_counts": filtered_reason_counts,
        "text_chunk_count_min": int(min(text_chunk_counts)) if text_chunk_counts else 0,
        "text_chunk_count_max": int(max(text_chunk_counts)) if text_chunk_counts else 0,
        "text_chunk_count_mean": float(sum(text_chunk_counts) / len(text_chunk_counts)) if text_chunk_counts else 0.0,
        "audio_chunk_count_min": int(min(audio_chunk_counts)) if audio_chunk_counts else 0,
        "audio_chunk_count_max": int(max(audio_chunk_counts)) if audio_chunk_counts else 0,
        "audio_chunk_count_mean": float(sum(audio_chunk_counts) / len(audio_chunk_counts)) if audio_chunk_counts else 0.0,
        "valid_row_count_mean": float(sum(valid_row_counts) / len(valid_row_counts)) if valid_row_counts else 0.0,
        "original_audio_seconds_mean": float(sum(original_seconds) / len(original_seconds)) if original_seconds else 0.0,
        "compact_audio_seconds_mean": float(sum(compact_seconds) / len(compact_seconds)) if compact_seconds else 0.0,
    }
    torch.save({"metadata": metadata, "items": items}, output)
    print(f"Wrote {output}")
    print(metadata)


if __name__ == "__main__":
    main()
