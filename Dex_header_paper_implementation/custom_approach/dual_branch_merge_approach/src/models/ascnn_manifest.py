"""Branch 2: ASCNN(I) on manifest BoW → embedding e_i (default 128-d)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.models.adaptive_shrinkage_unit import AdaptiveShrinkageUnit

if TYPE_CHECKING:
    from src.config import PipelineConfig


class ASCNNManifest(nn.Module):
    """
    Three ASU Conv1d blocks + global average pool (paper Fig. 7 manifest path).

    Input BoW (B, L) or (B, 1, L) → output (B, embed_dim).
    """

    def __init__(
        self,
        bow_dim: int = 4381,
        embed_dim: int = 128,
        *,
        channels: tuple[int, int, int] = (64, 128, 128),
    ) -> None:
        super().__init__()
        self.bow_dim = bow_dim
        self.embed_dim = embed_dim

        c1, c2, c3 = channels
        self.asu1 = AdaptiveShrinkageUnit(1, c1, kernel_size=3, stride=2)
        self.asu2 = AdaptiveShrinkageUnit(c1, c2, kernel_size=3, stride=2)
        self.asu3 = AdaptiveShrinkageUnit(c2, c3, kernel_size=3, stride=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        if c3 != embed_dim:
            self.proj = nn.Linear(c3, embed_dim)
        else:
            self.proj = nn.Identity()

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected bow (L), (B, L), or (B, 1, L), got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 input channel for BoW, got {x.shape[1]}")

        length = x.shape[-1]
        if length < self.bow_dim:
            pad = self.bow_dim - length
            x = nn.functional.pad(x, (0, pad))
        elif length > self.bow_dim:
            x = x[..., : self.bow_dim]
        return x

    def forward(self, bow: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(bow)
        x = self.asu1(x)
        x = self.asu2(x)
        x = self.asu3(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


def build_ascnn_manifest(
    bow_dim: int = 4381,
    embed_dim: int = 128,
) -> ASCNNManifest:
    return ASCNNManifest(bow_dim=bow_dim, embed_dim=embed_dim)


def build_ascnn_manifest_from_config(cfg: PipelineConfig) -> ASCNNManifest:
    model_cfg = cfg.model
    return build_ascnn_manifest(
        bow_dim=int(model_cfg.get("bow_padded_len", 4381)),
        embed_dim=int(model_cfg.get("ascnn_embed_dim", 128)),
    )
