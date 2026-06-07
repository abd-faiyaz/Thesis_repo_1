"""One-epoch train/validation loops with logits + sigmoid scoring."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_metrics


def _batch_labels(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    return y.to(device).float().view(-1, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    desc: str = "Train",
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    samples = 0
    t0 = time.perf_counter()

    pbar = tqdm(loader, desc=f"{desc} {epoch + 1}/{total_epochs}", unit="batch", leave=True)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = _batch_labels(batch_y, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        n_batches += 1
        samples += batch_x.size(0)

        elapsed = time.perf_counter() - t0
        samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg=f"{total_loss / n_batches:.4f}",
            samples_per_s=f"{samples_per_sec:.1f}",
        )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    epoch: int = 0,
    total_epochs: int = 1,
    desc: str = "Val",
    show_progress: bool = True,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    iterator: DataLoader | tqdm = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc=f"{desc} {epoch + 1}/{total_epochs}",
            unit="batch",
            leave=True,
        )

    for batch_x, batch_y in iterator:
        batch_x = batch_x.to(device)
        batch_y_dev = _batch_labels(batch_y, device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y_dev)
        batch_loss = float(loss.item())
        total_loss += batch_loss
        n_batches += 1

        scores = torch.sigmoid(logits).view(-1).cpu().numpy()
        labels = batch_y.cpu().numpy().astype(int).ravel()
        preds = (scores >= threshold).astype(int)

        y_true_list.append(labels)
        y_pred_list.append(preds)
        y_score_list.append(scores)

        if show_progress and isinstance(iterator, tqdm):
            iterator.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg=f"{total_loss / n_batches:.4f}",
            )

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_score = np.concatenate(y_score_list)
    metrics = compute_metrics(y_true, y_pred, y_score)
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics
