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
    encode_audio,
    load_transcript_text,
    resolve_device,
    resolve_path,
    validate_embedding,
    validate_manifest,
)
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract full-transcript chunk text embeddings for E-DAIC.")
    parser.add_argument("--config", required=True, help="Path to E-DAIC text chunk YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Optional output path overriding data.feature_cache_path.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def chunk_token_ids(token_ids: List[int], chunk_size: int, max_chunks: int) -> List[List[int]]:
    chunks = [token_ids[index : index + chunk_size] for index in range(0, len(token_ids), chunk_size)]
    chunks = [chunk for chunk in chunks if chunk]
    return chunks[:max_chunks]


@torch.no_grad()
def encode_text_chunks(
    text: str,
    tokenizer: AutoTokenizer,
    text_model: AutoModel,
    max_text_length: int,
    max_text_chunks: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunk_size = max_text_length - tokenizer.num_special_tokens_to_add(pair=False)
    chunks = chunk_token_ids(token_ids, chunk_size, max_text_chunks)
    if not chunks:
        chunks = [[]]

    embeddings = []
    for chunk in chunks:
        encoded = tokenizer.prepare_for_model(
            chunk,
            add_special_tokens=True,
            max_length=max_text_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = text_model(**encoded)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu())

    return torch.stack(embeddings).mean(dim=0), len(chunks)


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
    chunk_counts: List[int] = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract E-DAIC text chunk features"):
        participant_id = str(row[data_cfg["id_column"]])
        transcript_path = resolve_path(row[data_cfg["text_column"]], manifest_root)
        audio_path = resolve_path(row[data_cfg["audio_column"]], manifest_root)
        text = load_transcript_text(transcript_path, str(data_cfg.get("transcript_text_column", "Text")))

        text_embedding, num_text_chunks = encode_text_chunks(
            text=text,
            tokenizer=tokenizer,
            text_model=text_model,
            max_text_length=int(data_cfg["max_text_length"]),
            max_text_chunks=int(data_cfg["max_text_chunks"]),
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
        chunk_counts.append(num_text_chunks)

        items.append(
            {
                "participant_id": participant_id,
                "split": str(row[data_cfg["split_column"]]).lower(),
                "phq_score": float(row[data_cfg["phq_column"]]),
                "label": int(row[data_cfg["label_column"]]),
                "text_embedding": text_embedding,
                "audio_embedding": audio_embedding,
                "num_text_chunks": num_text_chunks,
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
        "max_text_chunks": int(data_cfg["max_text_chunks"]),
        "text_chunk_count_min": int(min(chunk_counts)) if chunk_counts else 0,
        "text_chunk_count_max": int(max(chunk_counts)) if chunk_counts else 0,
        "text_chunk_count_mean": float(sum(chunk_counts) / len(chunk_counts)) if chunk_counts else 0.0,
    }
    torch.save({"metadata": metadata, "items": items}, output)
    print(f"Wrote {output}")
    print(metadata)


if __name__ == "__main__":
    main()
