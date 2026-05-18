"""MLP classifier head on ASCNN embedding → malware logit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


class ClassifierHead(nn.Module):
    """Linear → BN → ReLU → Linear(1) on pooled ASCNN embedding."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected last dim {self.embed_dim}, got {embedding.shape[-1]}"
            )
        return self.net(embedding)


def build_classifier_head(
    embed_dim: int = 128,
    hidden_dim: int = 128,
) -> ClassifierHead:
    return ClassifierHead(embed_dim=embed_dim, hidden_dim=hidden_dim)


def build_classifier_head_from_config(cfg: PipelineConfig) -> ClassifierHead:
    model_cfg = cfg.model
    embed_dim = int(model_cfg.get("ascnn_embed_dim", 128))
    hidden_dim = int(model_cfg.get("classifier_hidden_dim", embed_dim))
    return build_classifier_head(embed_dim=embed_dim, hidden_dim=hidden_dim)
