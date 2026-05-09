from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Extract frozen text/audio embeddings for MODMA.")
    parser.add_argument("--config", required=True, help="Path to MODMA YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional subject limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Optional output path overriding data.feature_cache_path.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def downsample_audio_mask(attention_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    mask = attention_mask.unsqueeze(1).float()
    mask = torch.nn.functional.interpolate(mask, size=target_length, mode="nearest")
    return mask.squeeze(1).to(dtype=torch.long)


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def load_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


@torch.no_grad()
def encode_text(
    text_path: str,
    tokenizer: AutoTokenizer,
    text_model: AutoModel,
    max_text_length: int,
    device: torch.device,
) -> torch.Tensor:
    text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
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
    audio_path: str,
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


def validate_manifest(df: pd.DataFrame, data_cfg: Dict[str, Any]) -> None:
    required = [
        data_cfg["id_column"],
        data_cfg["label_column"],
        data_cfg["text_column"],
        data_cfg["audio_paths_column"],
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"MODMA manifest is missing columns: {missing}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    model_cfg = config["model"]
    device = resolve_device(str(config["training"]["device"]))

    manifest = pd.read_csv(data_cfg["manifest_path"], dtype={data_cfg["id_column"]: str})
    validate_manifest(manifest, data_cfg)
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model_name"])
    text_model = AutoModel.from_pretrained(model_cfg["text_model_name"]).to(device).eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_cfg["audio_model_name"])
    audio_model = Wav2Vec2Model.from_pretrained(model_cfg["audio_model_name"]).to(device).eval()

    items: List[Dict[str, Any]] = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract MODMA features"):
        participant_id = str(row[data_cfg["id_column"]])
        audio_paths = json.loads(row[data_cfg["audio_paths_column"]])
        expected_segments = int(data_cfg["num_segments"])
        if len(audio_paths) != expected_segments:
            raise ValueError(f"{participant_id} has {len(audio_paths)} audio paths, expected {expected_segments}")

        text_embedding = encode_text(
            text_path=str(row[data_cfg["text_column"]]),
            tokenizer=tokenizer,
            text_model=text_model,
            max_text_length=int(data_cfg["max_text_length"]),
            device=device,
        )
        audio_embeddings = torch.stack(
            [
                encode_audio_segment(
                    audio_path=audio_path,
                    feature_extractor=feature_extractor,
                    audio_model=audio_model,
                    sample_rate=int(data_cfg["sample_rate"]),
                    max_audio_length=int(data_cfg["max_audio_length"]),
                    device=device,
                )
                for audio_path in audio_paths
            ]
        )

        items.append(
            {
                "participant_id": participant_id,
                "label": int(row[data_cfg["label_column"]]),
                "text_embedding": text_embedding,
                "audio_embeddings": audio_embeddings,
            }
        )

    output = Path(args.output or data_cfg["feature_cache_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "text_dim": int(items[0]["text_embedding"].shape[-1]) if items else 0,
        "audio_dim": int(items[0]["audio_embeddings"].shape[-1]) if items else 0,
        "num_segments": int(data_cfg["num_segments"]),
        "num_items": len(items),
    }
    torch.save({"metadata": metadata, "items": items}, output)
    print(f"Wrote {output}")
    print(metadata)


if __name__ == "__main__":
    main()
