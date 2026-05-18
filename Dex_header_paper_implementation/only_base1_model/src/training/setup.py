"""Build loss, optimizer, scheduler, and device from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig
    from src.models.mlp_header import MLPHeader


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_training_objects(
    cfg: PipelineConfig,
    model: MLPHeader,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, torch.device]:
    train_cfg = cfg.training
    device = resolve_device(str(train_cfg.get("device", "cuda")))

    criterion: nn.Module = nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.005)),
        momentum=float(train_cfg.get("momentum", 0.9)),
    )

    scheduler_name = str(train_cfg.get("lr_scheduler", "StepLR"))
    gamma = float(train_cfg.get("lr_decay_factor", 0.5))
    step_size = int(train_cfg.get("lr_step_size", 10))

    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")

    model.to(device)
    return criterion, optimizer, scheduler, device
