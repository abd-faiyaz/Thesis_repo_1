"""ASCNN on concat(H, I): single-tower conv stack (paper Fig. 7 applied to combined input)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adaptive_shrinkage_unit import AdaptiveShrinkageUnit

if TYPE_CHECKING:
    from src.config import PipelineConfig


class ASCNNCombined(nn.Module):
    """
    Three ASU Conv1d blocks + global average pool on combined header+BoW sequence.

    Input (B, L) or (B, 1, L) with L <= combined_padded_len → output (B, embed_dim).
    """

    def __init__(
        self,
        combined_padded_len: int = 4488,
        embed_dim: int = 128,
        *,
        channels: tuple[int, int, int] = (64, 128, 128),
    ) -> None:
        super().__init__()
        self.combined_padded_len = combined_padded_len
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
            raise ValueError(
                f"Expected combined (L), (B, L), or (B, 1, L), got {x.shape}"
            )

        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 input channel, got {x.shape[1]}")

        length = x.shape[-1]
        if length < self.combined_padded_len:
            pad = self.combined_padded_len - length
            x = F.pad(x, (0, pad))
        elif length > self.combined_padded_len:
            x = x[..., : self.combined_padded_len]
        return x

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(combined)
        x = self.asu1(x)
        x = self.asu2(x)
        x = self.asu3(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


def build_ascnn_combined(
    combined_padded_len: int = 4488,
    embed_dim: int = 128,
    *,
    channels: tuple[int, int, int] = (64, 128, 128),
) -> ASCNNCombined:
    return ASCNNCombined(
        combined_padded_len=combined_padded_len,
        embed_dim=embed_dim,
        channels=channels,
    )


def build_ascnn_combined_from_config(cfg: PipelineConfig) -> ASCNNCombined:
    model_cfg = cfg.model
    ch = model_cfg.get("ascnn_channels", [64, 128, 128])
    channels = (int(ch[0]), int(ch[1]), int(ch[2]))
    return build_ascnn_combined(
        combined_padded_len=int(model_cfg.get("combined_padded_len", 4488)),
        embed_dim=int(model_cfg.get("ascnn_embed_dim", 128)),
        channels=channels,
    )
