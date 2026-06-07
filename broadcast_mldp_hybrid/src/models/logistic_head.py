"""Fallback logistic head: d → 1 logits (smallest ONNX export)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LogisticHead(nn.Module):
    """Single linear layer returning logits for BCEWithLogitsLoss."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]}"
            )
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_logistic_head(input_dim: int) -> LogisticHead:
    return LogisticHead(input_dim)
