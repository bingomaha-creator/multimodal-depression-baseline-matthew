from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


class EDAICFeatureDataset(Dataset):
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
            raise ValueError(f"No E-DAIC feature items found{detail} in {feature_cache_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            "participant_id": str(item["participant_id"]),
            "split": str(item["split"]),
            "phq_score": torch.tensor(float(item["phq_score"]), dtype=torch.float),
            "text_embedding": item["text_embedding"].float(),
            "audio_embedding": item["audio_embedding"].float(),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
        }


def collate_edaic_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "participant_id": [item["participant_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "phq_scores": torch.stack([item["phq_score"] for item in batch]),
        "text_embeddings": torch.stack([item["text_embedding"] for item in batch]),
        "audio_embeddings": torch.stack([item["audio_embedding"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
    }
