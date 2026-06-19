from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


MODALITIES = {"audio", "visual", "both"}


def _validate_modality(modality: str) -> None:
    if modality not in MODALITIES:
        raise ValueError("modality must be one of: audio, visual, both")


class DVlogMLP(nn.Module):
    def __init__(
        self,
        audio_dim: int = 50,
        visual_dim: int = 272,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        modality: str = "both",
    ) -> None:
        super().__init__()
        _validate_modality(modality)
        self.modality = modality
        input_dim = (audio_dim if modality in {"audio", "both"} else 0) + (
            visual_dim if modality in {"visual", "both"} else 0
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, audio_embeddings: torch.Tensor, visual_embeddings: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.modality in {"audio", "both"}:
            parts.append(audio_embeddings)
        if self.modality in {"visual", "both"}:
            parts.append(visual_embeddings)
        return self.classifier(torch.cat(parts, dim=-1))


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 4

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        pooling_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projection(features)
        packed = pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=features.shape[1],
        )
        mask = pooling_mask.bool().unsqueeze(-1)
        counts = mask.sum(dim=1).clamp_min(1)
        mean_pool = (output * mask).sum(dim=1) / counts
        max_pool = output.masked_fill(~mask, torch.finfo(output.dtype).min).max(dim=1).values
        no_valid_frames = ~pooling_mask.bool().any(dim=1)
        if no_valid_frames.any():
            max_pool = max_pool.masked_fill(no_valid_frames.unsqueeze(-1), 0.0)
        return self.dropout(torch.cat([mean_pool, max_pool], dim=-1))


class DVlogBiGRU(nn.Module):
    def __init__(
        self,
        audio_dim: int = 25,
        visual_dim: int = 136,
        projection_dim: int = 128,
        hidden_dim: int = 128,
        classifier_dim: int = 256,
        dropout: float = 0.2,
        modality: str = "both",
    ) -> None:
        super().__init__()
        _validate_modality(modality)
        self.modality = modality
        self.audio_encoder = (
            TemporalEncoder(audio_dim, projection_dim, hidden_dim, dropout)
            if modality in {"audio", "both"}
            else None
        )
        self.visual_encoder = (
            TemporalEncoder(visual_dim, projection_dim, hidden_dim, dropout)
            if modality in {"visual", "both"}
            else None
        )
        encoder_dim = hidden_dim * 4
        input_dim = encoder_dim * (2 if modality == "both" else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, 2),
        )

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        lengths: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        time_index = torch.arange(audio.shape[1], device=lengths.device).unsqueeze(0)
        audio_mask = time_index < lengths.unsqueeze(1)
        parts = []
        if self.audio_encoder is not None:
            parts.append(self.audio_encoder(audio, lengths, audio_mask))
        if self.visual_encoder is not None:
            parts.append(self.visual_encoder(visual, lengths, visual_mask))
        return self.classifier(torch.cat(parts, dim=-1))
