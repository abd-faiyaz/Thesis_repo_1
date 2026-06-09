"""Linear(R→d_R)+ReLU receiver embedding tower."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReceiverTower(nn.Module):
    def __init__(self, receiver_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.receiver_dim = receiver_dim
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(receiver_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        if R.dim() == 1:
            R = R.unsqueeze(0)
        return self.encoder(R)
