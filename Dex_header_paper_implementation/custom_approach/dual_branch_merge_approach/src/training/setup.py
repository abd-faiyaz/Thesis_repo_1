"""Build optimizer, scheduler, and device from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PipelineConfig
    from src.models.dual_branch_net import DualBranchNet


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer_and_scheduler(
    cfg: PipelineConfig,
    model: DualBranchNet,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    train_cfg = cfg.training
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

    return optimizer, scheduler


def build_training_objects(
    cfg: PipelineConfig,
    model: DualBranchNet,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, torch.device]:
    train_cfg = cfg.training
    device = resolve_device(str(train_cfg.get("device", "cuda")))
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    model.to(device)
    return optimizer, scheduler, device
