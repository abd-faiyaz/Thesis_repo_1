"""Build loss, optimizer, and device from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_training_objects(
    cfg: PipelineConfig,
    model: nn.Module,
    *,
    pos_weight: float,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.device]:
    train_cfg = cfg.training
    device = resolve_device(str(train_cfg.get("device", "cuda")))

    weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.005)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )

    model.to(device)
    return criterion, optimizer, device
