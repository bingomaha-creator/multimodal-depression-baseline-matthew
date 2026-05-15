from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


class EDAICSegmentFeatureDataset(Dataset):
    def __init__(
        self,
        feature_cache_path: str,
        indices: Optional[Iterable[int]] = None,
        split: Optional[str] = None,
    ) -> None:
        cache = torch.load(feature_cache_path, map_location="cpu")
        self.metadata = cache.get("metadata", {})
        self.items: List[Dict[str, Any]] = cache["items"]

        if split is not None:
            split_lower = split.lower()
            self.items = [item for item in self.items if str(item["split"]).lower() == split_lower]

        if indices is not None:
            self.items = [self.items[index] for index in indices]

        if not self.items:
            detail = f" split='{split}'" if split is not None else ""
            raise ValueError(f"No E-DAIC segment feature items found{detail} in {feature_cache_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            "participant_id": str(item["participant_id"]),
            "split": str(item["split"]),
            "phq_score": torch.tensor(float(item["phq_score"]), dtype=torch.float),
            "text_embeddings": item["text_embeddings"].float(),
            "audio_embeddings": item["audio_embeddings"].float(),
            "segment_mask": item["segment_mask"].float(),
            "num_segments": torch.tensor(int(item["num_segments"]), dtype=torch.long),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
        }


def collate_edaic_segment_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "participant_id": [item["participant_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "phq_scores": torch.stack([item["phq_score"] for item in batch]),
        "text_embeddings": torch.stack([item["text_embeddings"] for item in batch]),
        "audio_embeddings": torch.stack([item["audio_embeddings"] for item in batch]),
        "segment_mask": torch.stack([item["segment_mask"] for item in batch]),
        "num_segments": torch.stack([item["num_segments"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
    }
