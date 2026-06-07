"""Mode B Stage 1 — MLDP logistic head on x_S."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


class MldpLogistic(nn.Module):
    """Linear(|S| → 1) returning logits."""

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


class MldpStage1TinyMlp(nn.Module):
    """Optional Stage-1 tiny MLP: |S| → h → 1 logits."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]}"
            )
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_mldp_logistic(input_dim: int) -> MldpLogistic:
    return MldpLogistic(input_dim)


def build_mldp_stage1_from_config(cfg: PipelineConfig, input_dim: int) -> nn.Module:
    model_cfg = cfg.model
    head = str(model_cfg.get("mode_b_stage1", "logistic")).strip().lower()
    if head == "logistic":
        return build_mldp_logistic(input_dim)
    if head in {"tiny_mlp", "mlp"}:
        return MldpStage1TinyMlp(
            input_dim,
            hidden_dim=int(model_cfg.get("mode_b_stage1_mlp_hidden", 32)),
        )
    raise ValueError(
        f"Unknown model.mode_b_stage1={head!r}; use 'logistic' or 'tiny_mlp'"
    )
