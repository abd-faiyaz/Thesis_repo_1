"""Loss functions for dual-branch fusion training."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


def _pos_weight_from_class_balance(path) -> float | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    value = data.get("pos_weight")
    return float(value) if value is not None else None


def resolve_pos_weight(cfg: PipelineConfig) -> float | None:
    """
    BCE pos_weight for malware (label 1).
    Uses training.pos_weight if set, else training.benign_to_malware_ratio.
    """
    train_cfg = cfg.training
    pos_weight = train_cfg.get("pos_weight")
    if pos_weight is not None:
        return float(pos_weight)
    ratio = train_cfg.get("benign_to_malware_ratio")
    if ratio is not None:
        return float(ratio)
    if bool(train_cfg.get("auto_pos_weight", True)):
        from_path = _pos_weight_from_class_balance(cfg.paths.class_balance)
        if from_path is not None:
            return from_path
    return None


def build_criterion(cfg: PipelineConfig, device: torch.device) -> nn.Module:
    pos_weight = resolve_pos_weight(cfg)
    if pos_weight is not None:
        weight = torch.tensor([pos_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    return nn.BCEWithLogitsLoss()
