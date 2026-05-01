from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, Wav2Vec2FeatureExtractor


class EDAICDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        feature_extractor: Wav2Vec2FeatureExtractor,
        config: Dict[str, Any],
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_root = self.manifest_path.parent
        self.split = split
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.config = config

        df = pd.read_csv(self.manifest_path)
        self._validate_columns(df)
        split_column = config["split_column"]
        df[split_column] = df[split_column].astype(str).str.lower()
        self.df = df[df[split_column] == split.lower()].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split '{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | float | int]:
        row = self.df.iloc[index]
        transcript_path = self._resolve_path(row[self.config["text_column"]])
        audio_path = self._resolve_path(row[self.config["audio_column"]])
        phq_score = float(row[self.config["phq_column"]])
        label = int(phq_score >= float(self.config["positive_threshold"]))

        text = self._load_transcript(transcript_path)
        text_inputs = self.tokenizer(
            text,
            max_length=int(self.config["max_text_length"]),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        waveform = self._load_audio(audio_path)
        audio_inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=int(self.config["sample_rate"]),
            max_length=int(self.config["max_audio_length"]),
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "participant_id": str(row[self.config["id_column"]]),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "text_attention_mask": text_inputs["attention_mask"].squeeze(0),
            "audio_values": audio_inputs["input_values"].squeeze(0),
            "audio_attention_mask": audio_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "phq_score": torch.tensor(phq_score, dtype=torch.float),
        }

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = [
            self.config["id_column"],
            self.config["audio_column"],
            self.config["text_column"],
            self.config["phq_column"],
            self.config["split_column"],
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

    def _resolve_path(self, value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.manifest_root / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0)
        target_sample_rate = int(self.config["sample_rate"])
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        return waveform

    def _load_transcript(self, transcript_path: Path) -> str:
        if transcript_path.suffix.lower() != ".csv":
            return transcript_path.read_text(encoding="utf-8", errors="ignore")

        transcript_df = pd.read_csv(transcript_path)
        text_column = self.config.get("transcript_text_column", "Text")
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


def collate_batch(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "participant_id": [item["participant_id"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "text_attention_mask": torch.stack([item["text_attention_mask"] for item in batch]),
        "audio_values": torch.stack([item["audio_values"] for item in batch]),
        "audio_attention_mask": torch.stack([item["audio_attention_mask"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "phq_scores": torch.stack([item["phq_score"] for item in batch]),
    }
