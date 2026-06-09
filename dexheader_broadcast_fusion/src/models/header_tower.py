"""MLP(H) trunk 104→128→128 → z_H (no classification head)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HeaderTower(nn.Module):
    def __init__(self, input_dim: int = 104, hidden_dim: int = 128) -> None:
        super().__init__()
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

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        if H.dim() == 1:
            H = H.unsqueeze(0)
        x = self.block1(H)
        return self.block2(x)
