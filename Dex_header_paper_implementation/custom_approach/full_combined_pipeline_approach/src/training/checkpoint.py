"""Save/load training checkpoints for power-outage resume (Phase 5)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from src.models.combined_net import CombinedNet


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def build_checkpoint_state(
    *,
    next_epoch: int,
    model: CombinedNet,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loss: float,
    val_loss: float | None = None,
    best_val_loss: float | None = None,
    global_step: int = 0,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "next_epoch": next_epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "current_loss": train_loss,
        "train_loss": train_loss,
        "rng_state": capture_rng_state(),
    }
    if val_loss is not None:
        state["val_loss"] = val_loss
    if best_val_loss is not None:
        state["best_val_loss"] = best_val_loss
    return state


def restore_from_checkpoint(
    checkpoint: dict[str, Any],
    model: CombinedNet,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> tuple[int, int, float | None]:
    """
    Load weights and optimizer/scheduler/RNG state.
    Returns (next_epoch, global_step, best_val_loss).
    """
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"))

    next_epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", 0)))
    global_step = int(checkpoint.get("global_step", 0))
    best_val = checkpoint.get("best_val_loss")
    best_val_loss = float(best_val) if best_val is not None else None
    return next_epoch, global_step, best_val_loss
