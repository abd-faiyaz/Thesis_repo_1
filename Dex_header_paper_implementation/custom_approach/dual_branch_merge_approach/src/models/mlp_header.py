"""Branch 1: MLP(H) embedding (no sigmoid — fusion head owns the logit)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


class MLPHeaderBranch(nn.Module):
    """
    Two FC+BN+ReLU blocks on Dex header features → embedding e_h (default 128-d).
    """

    def __init__(self, input_dim: int = 104, embed_dim: int = 128) -> None:
        super().__init__()
        if input_dim < 1 or embed_dim < 1:
            raise ValueError("input_dim and embed_dim must be positive")

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.shape[-1]}")
        x = self.block1(x)
        return self.block2(x)


def build_mlp_header_branch(input_dim: int = 104, embed_dim: int = 128) -> MLPHeaderBranch:
    return MLPHeaderBranch(input_dim=input_dim, embed_dim=embed_dim)


def build_mlp_header_branch_from_config(cfg: PipelineConfig) -> MLPHeaderBranch:
    model_cfg = cfg.model
    return build_mlp_header_branch(
        input_dim=int(model_cfg.get("header_dim", 104)),
        embed_dim=int(model_cfg.get("hidden_dim", 128)),
    )
