"""Mode A — fused tiny MLP on x = [x_S || H]."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


class FusedMlp(nn.Module):
    """
    Linear(d → h) → ReLU → Dropout(0.2) → Linear(h → 1).

    Forward returns logits for BCEWithLogitsLoss; apply sigmoid at export/inference.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

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


def build_fused_mlp(
    input_dim: int,
    *,
    hidden_dim: int = 64,
    dropout: float = 0.2,
) -> FusedMlp:
    return FusedMlp(input_dim, hidden_dim=hidden_dim, dropout=dropout)


def build_fused_mlp_from_config(cfg: PipelineConfig, input_dim: int) -> FusedMlp:
    model_cfg = cfg.model
    return build_fused_mlp(
        input_dim,
        hidden_dim=int(model_cfg.get("mode_a_hidden", 64)),
        dropout=float(model_cfg.get("mode_a_dropout", 0.2)),
    )
