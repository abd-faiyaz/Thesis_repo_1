"""Late fusion head: concat(e_h, e_i) → malware logit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


class FusionHead(nn.Module):
    """Linear → BN → ReLU → Linear(1) on fused branch embeddings."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.dim() == 1:
            fused = fused.unsqueeze(0)
        if fused.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {fused.shape[-1]}")
        return self.net(fused)


def build_fusion_head(
    header_embed_dim: int = 128,
    manifest_embed_dim: int = 128,
    hidden_dim: int = 128,
) -> FusionHead:
    return FusionHead(
        input_dim=header_embed_dim + manifest_embed_dim,
        hidden_dim=hidden_dim,
    )


def build_fusion_head_from_config(cfg: PipelineConfig) -> FusionHead:
    model_cfg = cfg.model
    h = int(model_cfg.get("hidden_dim", 128))
    m = int(model_cfg.get("ascnn_embed_dim", 128))
    return build_fusion_head(h, m, hidden_dim=h)
