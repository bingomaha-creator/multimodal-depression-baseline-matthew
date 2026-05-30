from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.extract_edaic_features import (
    count_by_split_and_label,
    downsample_audio_mask,
    load_audio,
    masked_mean_pool,
    resolve_device,
    resolve_path,
    validate_embedding,
    validate_manifest,
)
from src.extract_edaic_text_chunk_features import tokenize_full_text
from src.utils.compact_speech import (
    CompactSpeechChunk,
    TranscriptPiece,
    build_compact_speech_chunks,
    filter_transcript_pieces,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract timestamp-filtered compact speech features for E-DAIC.")
    parser.add_argument("--config", required=True, help="Path to E-DAIC compact speech YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Optional output path overriding data.feature_cache_path.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_transcript_pieces(transcript_path: Path, data_cfg: Dict[str, Any]) -> List[TranscriptPiece]:
    transcript = pd.read_csv(transcript_path)
    required = [
        data_cfg["start_time_column"],
        data_cfg["end_time_column"],
        data_cfg["transcript_text_column"],
    ]
    missing = [column for column in required if column not in transcript.columns]
    if missing:
        raise ValueError(f"Transcript CSV {transcript_path} is missing columns: {missing}")

    pieces: List[TranscriptPiece] = []
    for _, row in transcript.iterrows():
        start = pd.to_numeric(row[data_cfg["start_time_column"]], errors="coerce")
        end = pd.to_numeric(row[data_cfg["end_time_column"]], errors="coerce")
        text_value = row[data_cfg["transcript_text_column"]]
        text = "" if pd.isna(text_value) else str(text_value).strip()
        if pd.isna(start) or pd.isna(end):
            pieces.append(TranscriptPiece(start=-1.0, end=-1.0, text=text))
        else:
            pieces.append(TranscriptPiece(start=float(start), end=float(end), text=text))
    return pieces


def slice_waveform_seconds(waveform: torch.Tensor, start: float, end: float, sample_rate: int) -> torch.Tensor:
    start_index = max(0, int(round(start * sample_rate)))
    end_index = min(int(round(end * sample_rate)), int(waveform.numel()))
    return waveform[start_index:end_index]


def make_audio_waveform_for_chunk(
    chunk: CompactSpeechChunk,
    waveform: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    segments = [
        slice_waveform_seconds(waveform, piece.start, piece.end, sample_rate)
        for piece in chunk.pieces
    ]
    segments = [segment for segment in segments if segment.numel() > 0]
    if not segments:
        return torch.zeros(1, dtype=waveform.dtype)
    return torch.cat(segments, dim=0)


@torch.no_grad()
def encode_text_chunk(
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
def encode_audio_waveform(
    waveform: torch.Tensor,
    feature_extractor: Wav2Vec2FeatureExtractor,
    audio_model: Wav2Vec2Model,
    sample_rate: int,
    max_audio_length: int,
    device: torch.device,
) -> torch.Tensor:
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
    embedding = pooled.squeeze(0).detach().cpu()

    del audio_values, audio_attention_mask, outputs, pooled
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embedding


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
    max_text_length = int(data_cfg["max_text_length"])
    min_audio_chunk_seconds = float(data_cfg.get("min_audio_chunk_seconds", 0.0))

    items: List[Dict[str, Any]] = []
    manifest_root = manifest_path.parent
    chunk_counts: List[int] = []
    valid_row_counts: List[int] = []
    compact_seconds: List[float] = []
    original_seconds: List[float] = []
    filtered_reason_counts: Dict[str, int] = {}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract E-DAIC compact speech features"):
        participant_id = str(row[data_cfg["id_column"]])
        transcript_path = resolve_path(row[data_cfg["text_column"]], manifest_root)
        audio_path = resolve_path(row[data_cfg["audio_column"]], manifest_root)
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

        chunks = build_compact_speech_chunks(
            valid_pieces,
            token_counter=lambda text: len(tokenize_full_text(tokenizer, text)),
            max_audio_chunk_seconds=float(data_cfg["max_audio_chunk_seconds"]),
            max_text_chunk_tokens=int(data_cfg["max_text_chunk_tokens"]),
            max_chunks=int(data_cfg["max_chunks"]),
        )
        chunks = [chunk for chunk in chunks if chunk.audio_seconds >= min_audio_chunk_seconds]
        if not chunks:
            raise ValueError(f"No valid compact speech chunks for participant {participant_id}")

        text_embeddings = []
        audio_embeddings = []
        for chunk in chunks:
            text_embeddings.append(
                encode_text_chunk(
                    text=chunk.text,
                    tokenizer=tokenizer,
                    text_model=text_model,
                    max_text_length=max_text_length,
                    device=device,
                )
            )
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

        text_embedding = torch.stack(text_embeddings).mean(dim=0)
        audio_embedding = torch.stack(audio_embeddings).mean(dim=0)
        validate_embedding("text", participant_id, text_embedding)
        validate_embedding("audio", participant_id, audio_embedding)

        chunk_counts.append(len(chunks))
        valid_row_counts.append(len(valid_pieces))
        compact_seconds.append(sum(chunk.audio_seconds for chunk in chunks))
        items.append(
            {
                "participant_id": participant_id,
                "split": str(row[data_cfg["split_column"]]).lower(),
                "phq_score": float(row[data_cfg["phq_column"]]),
                "label": int(row[data_cfg["label_column"]]),
                "text_embedding": text_embedding,
                "audio_embedding": audio_embedding,
                "num_compact_chunks": len(chunks),
                "num_valid_transcript_rows": len(valid_pieces),
                "num_filtered_rows": len(pieces) - len(valid_pieces),
                "compact_speech_seconds": float(compact_seconds[-1]),
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
        "max_text_length": max_text_length,
        "max_raw_segment_seconds": float(data_cfg["max_raw_segment_seconds"]),
        "max_audio_chunk_seconds": float(data_cfg["max_audio_chunk_seconds"]),
        "min_audio_chunk_seconds": min_audio_chunk_seconds,
        "max_text_chunk_tokens": int(data_cfg["max_text_chunk_tokens"]),
        "max_chunks": int(data_cfg["max_chunks"]),
        "filtered_reason_counts": filtered_reason_counts,
        "chunk_count_min": int(min(chunk_counts)) if chunk_counts else 0,
        "chunk_count_max": int(max(chunk_counts)) if chunk_counts else 0,
        "chunk_count_mean": float(sum(chunk_counts) / len(chunk_counts)) if chunk_counts else 0.0,
        "valid_row_count_mean": float(sum(valid_row_counts) / len(valid_row_counts)) if valid_row_counts else 0.0,
        "original_audio_seconds_mean": float(sum(original_seconds) / len(original_seconds)) if original_seconds else 0.0,
        "compact_speech_seconds_mean": float(sum(compact_seconds) / len(compact_seconds)) if compact_seconds else 0.0,
    }
    torch.save({"metadata": metadata, "items": items}, output)
    print(f"Wrote {output}")
    print(metadata)


if __name__ == "__main__":
    main()
