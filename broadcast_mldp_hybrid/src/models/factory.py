"""Model factory helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from src.models.logistic_head import build_logistic_head
from src.models.tiny_mlp import build_tiny_mlp_from_config

if TYPE_CHECKING:
    from src.config import PipelineConfig


def build_deployment_model_from_config(
    cfg: PipelineConfig,
    input_dim: int,
) -> nn.Module:
    """Select tiny MLP or logistic fallback from classifier.deployment."""
    deployment = str(cfg.classifier.get("deployment", "tiny_mlp")).strip().lower()
    if deployment == "logistic":
        return build_logistic_head(input_dim)
    if deployment == "tiny_mlp":
        return build_tiny_mlp_from_config(cfg, input_dim)
    raise ValueError(
        f"Unknown classifier.deployment={deployment!r}; use 'tiny_mlp' or 'logistic'"
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
