"""Model factory helpers."""

from __future__ import annotations

import torch.nn as nn

from src.config import PipelineConfig
from src.models.fused_mlp import build_fused_mlp_from_config
from src.models.mldp_logistic import build_mldp_logistic, build_mldp_stage1_from_config


def build_mode_a_from_config(cfg: PipelineConfig, fused_dim: int) -> nn.Module:
    return build_fused_mlp_from_config(cfg, fused_dim)


def build_mode_b_stage1_from_config(cfg: PipelineConfig, s_dim: int) -> nn.Module:
    return build_mldp_stage1_from_config(cfg, s_dim)


def build_mode_b_stage1_logistic(s_dim: int) -> nn.Module:
    return build_mldp_logistic(s_dim)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_fp32_bytes(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) * 4
