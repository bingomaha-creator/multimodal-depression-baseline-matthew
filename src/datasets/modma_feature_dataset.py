from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


class MODMAFeatureDataset(Dataset):
    def __init__(self, feature_cache_path: str, indices: Optional[Iterable[int]] = None) -> None:
        cache = torch.load(feature_cache_path, map_location="cpu")
        self.metadata = cache.get("metadata", {})
        self.items: List[Dict[str, Any]] = cache["items"]
        if indices is not None:
            self.items = [self.items[index] for index in indices]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            "participant_id": item["participant_id"],
            "text_embedding": item["text_embedding"].float(),
            "audio_embeddings": item["audio_embeddings"].float(),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
        }


def collate_modma_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "participant_id": [item["participant_id"] for item in batch],
        "text_embeddings": torch.stack([item["text_embedding"] for item in batch]),
        "audio_embeddings": torch.stack([item["audio_embeddings"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
    }
