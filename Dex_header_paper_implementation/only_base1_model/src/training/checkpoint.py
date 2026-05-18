"""Save/load training checkpoints for power-outage resume (Phase 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from src.models.mlp_header import MLPHeader


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Persist training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
    """Load checkpoint if it exists; otherwise return None."""
    if not path.is_file():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def build_checkpoint_state(
    *,
    next_epoch: int,
    model: MLPHeader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loss: float,
    val_loss: float | None = None,
    val_metrics: dict[str, float] | None = None,
    feature_dim: int | None = None,
    hidden_dim: int | None = None,
) -> dict[str, Any]:
    """Checkpoint payload written after each completed epoch."""
    state: dict[str, Any] = {
        "next_epoch": next_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "current_loss": train_loss,
        "train_loss": train_loss,
    }
    if val_loss is not None:
        state["val_loss"] = val_loss
    if val_metrics is not None:
        state["val_metrics"] = val_metrics
    if feature_dim is not None:
        state["feature_dim"] = feature_dim
    if hidden_dim is not None:
        state["hidden_dim"] = hidden_dim
    return state


def restore_from_checkpoint(
    checkpoint: dict[str, Any],
    model: MLPHeader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> int:
    """
    Load weights and optimizer/scheduler state.
    Returns next_epoch index to continue training from.
    """
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("next_epoch", checkpoint.get("epoch", 0)))
