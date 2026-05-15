from __future__ import annotations

import torch
from torch import nn


def masked_mean(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(embeddings.dtype)
    summed = (embeddings * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return summed / counts


class EDAICSegmentFeatureBaseline(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        modality: str = "both",
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        if modality not in {"text", "audio", "both"}:
            raise ValueError("modality must be one of: text, audio, both")
        self.modality = modality

        input_dim = 0
        if modality in {"text", "both"}:
            input_dim += text_dim
        if modality in {"audio", "both"}:
            input_dim += audio_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = []
        if self.modality in {"text", "both"}:
            features.append(masked_mean(text_embeddings, segment_mask))
        if self.modality in {"audio", "both"}:
            features.append(masked_mean(audio_embeddings, segment_mask))
        fused = torch.cat(features, dim=-1)
        return self.classifier(fused)
