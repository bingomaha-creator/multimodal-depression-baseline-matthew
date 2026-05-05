from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from transformers import AutoModel, Wav2Vec2Model


class MultimodalBaseline(nn.Module):
    def __init__(
        self,
        text_model_name: str = "roberta-base",
        audio_model_name: str = "facebook/wav2vec2-base",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbones: bool = True,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.freeze_backbones_enabled = freeze_backbones

        text_dim = self.text_encoder.config.hidden_size
        audio_dim = self.audio_encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        if freeze_backbones:
            self.freeze_backbones()

    def freeze_backbones(self) -> None:
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad = False
        self.text_encoder.eval()
        self.audio_encoder.eval()

    def train(self, mode: bool = True) -> "MultimodalBaseline":
        super().train(mode)
        if self.freeze_backbones_enabled:
            self.text_encoder.eval()
            self.audio_encoder.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_values: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        backbone_context = torch.no_grad() if self.freeze_backbones_enabled else nullcontext()

        with backbone_context:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=text_attention_mask,
            )
            audio_outputs = self.audio_encoder(
                input_values=audio_values,
                attention_mask=audio_attention_mask,
            )

        text_embedding = text_outputs.last_hidden_state[:, 0, :]
        audio_embedding = self._masked_mean_pool(
            audio_outputs.last_hidden_state,
            self._downsample_audio_mask(audio_attention_mask, audio_outputs.last_hidden_state.shape[1]),
        )

        fused = torch.cat([text_embedding, audio_embedding], dim=-1)
        return self.classifier(fused)

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    @staticmethod
    def _downsample_audio_mask(attention_mask: torch.Tensor, target_length: int) -> torch.Tensor:
        mask = attention_mask.unsqueeze(1).float()
        mask = torch.nn.functional.interpolate(mask, size=target_length, mode="nearest")
        return mask.squeeze(1).to(dtype=torch.long)
