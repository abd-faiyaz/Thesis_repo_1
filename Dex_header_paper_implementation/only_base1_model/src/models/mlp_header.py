"""Base Model 1: MLP(H) — two hidden blocks + sigmoid output (Phase 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


class MLPHeader(nn.Module):
    """
    Shallow MLP on Dex header features (MSFDroid Base Model 1 / MLP(H)).

    Architecture:
        Linear(input_dim, hidden_dim) -> BatchNorm1d -> ReLU
        Linear(hidden_dim, hidden_dim) -> BatchNorm1d -> ReLU
        Linear(hidden_dim, 1) -> Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) feature vectors from DexDataset.
        Returns:
            (batch, 1) malware probabilities in [0, 1].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]}"
            )
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward; returns shape (batch, 1)."""
        return self.forward(x)


def build_mlp_header(input_dim: int, hidden_dim: int = 128) -> MLPHeader:
    """Construct MLP(H) with explicit dimensions."""
    return MLPHeader(input_dim=input_dim, hidden_dim=hidden_dim)


def build_mlp_header_from_config(
    cfg: PipelineConfig,
    input_dim: int,
) -> MLPHeader:
    """Build model using hidden_dim from config/default.yaml."""
    hidden_dim = int(cfg.model.get("hidden_dim", 128))
    return build_mlp_header(input_dim=input_dim, hidden_dim=hidden_dim)
