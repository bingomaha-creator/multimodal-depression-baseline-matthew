from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


LABEL_MAP = {"normal": 0, "depression": 1}
VALID_SPLITS = {"train", "valid", "test"}
AUDIO_DIM = 25
VISUAL_DIM = 136
REQUIRED_COLUMNS = {"index", "label", "duration", "gender", "fold"}


@dataclass(frozen=True)
class DVlogSample:
    sample_id: str
    label: int
    split: str
    gender: str
    duration: float
    acoustic_path: Path
    visual_path: Path


def discover_dvlog_samples(dataset_root: str | Path) -> list[DVlogSample]:
    root = Path(dataset_root)
    labels_path = root / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"D-Vlog labels file not found: {labels_path}")

    frame = pd.read_csv(labels_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"D-Vlog labels file is missing columns {missing_columns}: {labels_path}")
    if frame["index"].duplicated().any():
        duplicate_ids = frame.loc[frame["index"].duplicated(), "index"].astype(str).tolist()
        raise ValueError(f"Duplicate D-Vlog sample IDs in {labels_path}: {duplicate_ids[:10]}")

    samples: list[DVlogSample] = []
    for row in frame.itertuples(index=False):
        sample_id = str(int(row.index))
        label_name = str(row.label).strip().lower()
        split = str(row.fold).strip().lower()
        if label_name not in LABEL_MAP:
            raise ValueError(f"Unknown D-Vlog label {row.label!r} for sample {sample_id}")
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown D-Vlog split {row.fold!r} for sample {sample_id}")
        samples.append(
            DVlogSample(
                sample_id=sample_id,
                label=LABEL_MAP[label_name],
                split=split,
                gender=str(row.gender).strip().lower(),
                duration=float(row.duration),
                acoustic_path=root / sample_id / f"{sample_id}_acoustic.npy",
                visual_path=root / sample_id / f"{sample_id}_visual.npy",
            )
        )
    return sorted(samples, key=lambda sample: int(sample.sample_id))


def _load_array(path: Path, expected_dim: int, modality: str, sample_id: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {modality} feature file for sample {sample_id}: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(
            f"Expected non-empty 2D {modality} features for sample {sample_id}, got shape {array.shape}: {path}"
        )
    if array.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} {modality} features for sample {sample_id}, got shape {array.shape}: {path}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Found non-finite {modality} values for sample {sample_id}: {path}")
    return np.asarray(array, dtype=np.float32)


def load_feature_pair(sample: DVlogSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio = _load_array(sample.acoustic_path, AUDIO_DIM, "acoustic", sample.sample_id)
    visual = _load_array(sample.visual_path, VISUAL_DIM, "visual", sample.sample_id)
    if audio.shape[0] != visual.shape[0]:
        raise ValueError(
            f"Cross-modal length mismatch for sample {sample.sample_id}: "
            f"acoustic={audio.shape[0]}, visual={visual.shape[0]}"
        )
    visual_mask = np.any(visual != 0.0, axis=1)
    if not visual_mask.any():
        raise ValueError(f"Sample {sample.sample_id} has no valid visual frames: {sample.visual_path}")
    return audio, visual, visual_mask


def validate_dvlog_samples(samples: Sequence[DVlogSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("No D-Vlog samples were discovered")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate D-Vlog sample IDs were discovered")

    split_counts = {split: 0 for split in ("train", "valid", "test")}
    label_counts = {0: 0, 1: 0}
    missing_visual_rows = 0
    total_rows = 0
    for sample in samples:
        audio, _, visual_mask = load_feature_pair(sample)
        split_counts[sample.split] += 1
        label_counts[sample.label] += 1
        total_rows += int(audio.shape[0])
        missing_visual_rows += int((~visual_mask).sum())

    empty_splits = [split for split, count in split_counts.items() if count == 0]
    if empty_splits:
        raise ValueError(f"D-Vlog official splits are empty: {empty_splits}")
    if len(samples) == 961:
        expected_ids = [str(index) for index in range(961)]
        if sample_ids != expected_ids:
            raise ValueError("The full D-Vlog release must contain contiguous sample IDs 0 through 960")
        expected_splits = {"train": 647, "valid": 102, "test": 212}
        if split_counts != expected_splits:
            raise ValueError(
                f"Unexpected official D-Vlog split counts: {split_counts}; expected {expected_splits}"
            )
    return {
        "num_samples": len(samples),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "total_time_steps": total_rows,
        "missing_visual_rows": missing_visual_rows,
    }


@dataclass
class FeatureNormalizer:
    audio_mean: np.ndarray
    audio_std: np.ndarray
    visual_mean: np.ndarray
    visual_std: np.ndarray

    @classmethod
    def fit(cls, samples: Iterable[DVlogSample]) -> "FeatureNormalizer":
        audio_sum = np.zeros(AUDIO_DIM, dtype=np.float64)
        audio_sq_sum = np.zeros(AUDIO_DIM, dtype=np.float64)
        visual_sum = np.zeros(VISUAL_DIM, dtype=np.float64)
        visual_sq_sum = np.zeros(VISUAL_DIM, dtype=np.float64)
        audio_count = 0
        visual_count = 0
        for sample in samples:
            audio, visual, visual_mask = load_feature_pair(sample)
            valid_visual = visual[visual_mask]
            audio_sum += audio.sum(axis=0, dtype=np.float64)
            audio_sq_sum += np.square(audio, dtype=np.float64).sum(axis=0)
            visual_sum += valid_visual.sum(axis=0, dtype=np.float64)
            visual_sq_sum += np.square(valid_visual, dtype=np.float64).sum(axis=0)
            audio_count += audio.shape[0]
            visual_count += valid_visual.shape[0]
        if audio_count == 0 or visual_count == 0:
            raise ValueError("Cannot fit D-Vlog normalizer without valid training frames")

        audio_mean, audio_std = _mean_std(audio_sum, audio_sq_sum, audio_count)
        visual_mean, visual_std = _mean_std(visual_sum, visual_sq_sum, visual_count)
        return cls(audio_mean, audio_std, visual_mean, visual_std)

    def transform_audio(self, audio: np.ndarray) -> np.ndarray:
        return np.asarray((audio - self.audio_mean) / self.audio_std, dtype=np.float32)

    def transform_visual(self, visual: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        transformed = np.asarray((visual - self.visual_mean) / self.visual_std, dtype=np.float32)
        transformed[~valid_mask] = 0.0
        return transformed

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "audio_mean": self.audio_mean,
            "audio_std": self.audio_std,
            "visual_mean": self.visual_mean,
            "visual_std": self.visual_std,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FeatureNormalizer":
        return cls(
            audio_mean=np.asarray(state["audio_mean"], dtype=np.float32),
            audio_std=np.asarray(state["audio_std"], dtype=np.float32),
            visual_mean=np.asarray(state["visual_mean"], dtype=np.float32),
            visual_std=np.asarray(state["visual_std"], dtype=np.float32),
        )


def _mean_std(total: np.ndarray, square_total: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    mean = total / float(count)
    variance = np.maximum(square_total / float(count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def summarize_sequence(features: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    if valid_mask is None:
        valid = features
    else:
        valid = features[np.asarray(valid_mask, dtype=bool)]
    if valid.shape[0] == 0:
        raise ValueError("Cannot summarize a sequence without valid frames")
    return np.concatenate([valid.mean(axis=0), valid.std(axis=0)], axis=0).astype(np.float32)


class DVlogDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[DVlogSample],
        normalizer: FeatureNormalizer,
        representation: str,
        cache_in_memory: bool = True,
    ) -> None:
        if representation not in {"pooled", "temporal"}:
            raise ValueError("representation must be one of: pooled, temporal")
        self.samples = list(samples)
        self.normalizer = normalizer
        self.representation = representation
        self.cached_items = [self._load_item(sample) for sample in self.samples] if cache_in_memory else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cached_items is not None:
            return self.cached_items[index]
        return self._load_item(self.samples[index])

    def _load_item(self, sample: DVlogSample) -> dict[str, Any]:
        audio, visual, visual_mask = load_feature_pair(sample)
        audio = self.normalizer.transform_audio(audio)
        visual = self.normalizer.transform_visual(visual, visual_mask)
        item: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "label": sample.label,
            "gender": sample.gender,
            "duration": sample.duration,
            "length": audio.shape[0],
        }
        if self.representation == "pooled":
            item["audio_embedding"] = summarize_sequence(audio)
            item["visual_embedding"] = summarize_sequence(visual, visual_mask)
        else:
            item["audio"] = audio
            item["visual"] = visual
            item["visual_mask"] = visual_mask
        return item


def _metadata_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "gender": [item["gender"] for item in batch],
        "duration": torch.tensor([item["duration"] for item in batch], dtype=torch.float32),
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
    }


def collate_dvlog_pooled(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = _metadata_batch(batch)
    result["audio_embeddings"] = torch.tensor(
        np.stack([item["audio_embedding"] for item in batch]), dtype=torch.float32
    )
    result["visual_embeddings"] = torch.tensor(
        np.stack([item["visual_embedding"] for item in batch]), dtype=torch.float32
    )
    return result


def collate_dvlog_temporal(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = _metadata_batch(batch)
    result["audio"] = pad_sequence(
        [torch.from_numpy(item["audio"]) for item in batch], batch_first=True
    )
    result["visual"] = pad_sequence(
        [torch.from_numpy(item["visual"]) for item in batch], batch_first=True
    )
    result["visual_mask"] = pad_sequence(
        [torch.from_numpy(item["visual_mask"]) for item in batch], batch_first=True, padding_value=False
    ).bool()
    return result
