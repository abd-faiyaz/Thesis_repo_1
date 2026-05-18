"""Validation metrics: accuracy, F1, AUC (Phase 6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from src.config import PipelineConfig
    from src.models.dual_branch_net import DualBranchNet

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def format_metrics(metrics: dict[str, float]) -> str:
    auc = metrics.get("roc_auc", float("nan"))
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "n/a"
    return (
        f"ACC={metrics['accuracy']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"AUC={auc_str}"
    )


@torch.no_grad()
def collect_predictions(
    model: DualBranchNet,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for header, bow, batch_y in loader:
        header = header.to(device)
        bow = bow.to(device)
        scores = model.predict_proba(header, bow).view(-1).cpu().numpy()
        labels = batch_y.cpu().numpy().astype(int).ravel()
        preds = (scores >= threshold).astype(int)

        y_true_list.append(labels)
        y_pred_list.append(preds)
        y_score_list.append(scores)

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_score_list),
    )


@torch.no_grad()
def validation_epoch(
    model: DualBranchNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    epoch: int = 0,
    total_epochs: int = 1,
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
            desc=f"Val   {epoch + 1}/{total_epochs}",
            unit="batch",
            leave=True,
        )

    for header, bow, batch_y in iterator:
        header = header.to(device)
        bow = bow.to(device)
        batch_y_dev = batch_y.to(device).float().view(-1, 1)

        logits = model(header, bow)
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
    return total_loss / max(n_batches, 1), metrics


def run_evaluation(
    cfg: PipelineConfig,
    *,
    checkpoint_path: Path | None = None,
    split: str = "val",
) -> dict[str, Any]:
    from src.config import ensure_artifact_dirs
    from src.data.dataloaders import build_dataloaders_from_config
    from src.models.dual_branch_net import build_dual_branch_net_from_config
    from src.training.checkpoint import load_checkpoint, restore_from_checkpoint
    from src.training.losses import build_criterion
    from src.training.setup import build_training_objects

    ensure_artifact_dirs(cfg)
    train_loader, val_loader, _, _ = build_dataloaders_from_config(cfg)
    loader = val_loader if split == "val" else train_loader

    ckpt_path = checkpoint_path or cfg.paths.best_checkpoint
    if not ckpt_path.is_file():
        ckpt_path = cfg.paths.latest_checkpoint

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    model = build_dual_branch_net_from_config(cfg)
    optimizer, scheduler, device = build_training_objects(cfg, model)
    criterion = build_criterion(cfg, device)
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    assert checkpoint is not None
    restore_from_checkpoint(checkpoint, model, optimizer, scheduler)

    threshold = float(cfg.evaluation.get("threshold", 0.5))
    val_loss, metrics = validation_epoch(
        model,
        loader,
        criterion,
        device,
        threshold=threshold,
        show_progress=True,
    )

    result = {"split": split, "loss": val_loss, **metrics, "checkpoint": str(ckpt_path)}
    print(f"Evaluation ({split}) — loss={val_loss:.4f} {format_metrics(metrics)}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DualBranchNet on cached shards.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    from src.config import load_config

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_evaluation(cfg, checkpoint_path=args.checkpoint, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
