"""One-epoch train/validation loops with tqdm (dual-branch batches)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from src.models.dual_branch_net import DualBranchNet


def _batch_labels(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    return y.to(device).float().view(-1, 1)


def train_one_epoch(
    model: DualBranchNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    global_step: int = 0,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    samples = 0
    t0 = time.perf_counter()

    pbar = tqdm(
        loader,
        desc=f"Train {epoch + 1}/{total_epochs}",
        unit="batch",
        leave=False,
    )
    for header, bow, batch_y in pbar:
        header = header.to(device)
        bow = bow.to(device)
        batch_y = _batch_labels(batch_y, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(header, bow)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        n_batches += 1
        samples += header.size(0)
        global_step += 1

        elapsed = time.perf_counter() - t0
        samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg=f"{total_loss / n_batches:.4f}",
            samples_per_s=f"{samples_per_sec:.1f}",
        )

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def validate_one_epoch(
    model: DualBranchNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Val   {epoch + 1}/{total_epochs}",
        unit="batch",
        leave=False,
    )
    for header, bow, batch_y in pbar:
        header = header.to(device)
        bow = bow.to(device)
        batch_y = _batch_labels(batch_y, device)
        logits = model(header, bow)
        loss = criterion(logits, batch_y)
        batch_loss = float(loss.item())
        total_loss += batch_loss
        n_batches += 1
        pbar.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)
