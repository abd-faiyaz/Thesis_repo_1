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
    from src.models.mlp_header import MLPHeader

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """
    Paper metrics: ACC, F1, ROC-AUC.
    y_true / y_pred: binary labels; y_score: predicted malware probabilities.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        # Only one class present in y_true
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
    model: MLPHeader,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on loader; return (y_true, y_pred, y_score)."""
    model.eval()
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        probs = model(batch_x).view(-1)
        scores = probs.cpu().numpy()
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
    model: MLPHeader,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    epoch: int = 0,
    total_epochs: int = 1,
    show_progress: bool = True,
) -> tuple[float, dict[str, float]]:
    """Validation pass: BCE loss + sklearn ACC / F1 / AUC."""
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

    for batch_x, batch_y in iterator:
        batch_x = batch_x.to(device)
        batch_y_dev = batch_y.to(device).float().view(-1, 1)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y_dev)
        batch_loss = float(loss.item())
        total_loss += batch_loss
        n_batches += 1

        scores = outputs.view(-1).cpu().numpy()
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


def run_evaluation(
    cfg: PipelineConfig,
    *,
    checkpoint_path: Path | None = None,
    split: str = "val",
    metrics_out: Path | None = None,
) -> dict[str, Any]:
    """
    Load trained checkpoint and compute metrics on val (or train) split.
    Returns dict with loss and metric values.
    """
    from src.config import ensure_artifact_dirs, load_config
    from src.data.dataloaders import build_dataloaders_from_config, build_test_loader_from_config
    from src.models.mlp_header import build_mlp_header
    from src.training.checkpoint import load_checkpoint, restore_from_checkpoint
    from src.training.setup import build_training_objects

    ensure_artifact_dirs(cfg)
    train_loader, val_loader, feature_dim = build_dataloaders_from_config(cfg)
    if split == "test":
        loader, feature_dim = build_test_loader_from_config(cfg)
    elif split == "val":
        loader = val_loader
    else:
        loader = train_loader

    ckpt_path = checkpoint_path or cfg.paths.latest_checkpoint
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    feature_dim = int(checkpoint.get("feature_dim", feature_dim))
    hidden_dim = int(
        checkpoint.get("hidden_dim", cfg.model.get("hidden_dim", 128))
    )
    model = build_mlp_header(input_dim=feature_dim, hidden_dim=hidden_dim)
    criterion, optimizer, scheduler, device = build_training_objects(cfg, model)
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
    y_true, y_pred, y_score = collect_predictions(
        model, loader, device, threshold=threshold
    )
    from src.training.run_logging import (
        build_confusion_matrix,
        build_metrics_payload,
        finalize_run_manifest,
        log_checkpoint_summary,
        write_metrics_json,
    )

    confusion = build_confusion_matrix(y_true, y_pred)
    payload = build_metrics_payload(
        cfg,
        split=split,
        n_samples=int(y_true.shape[0]),
        loss=val_loss,
        metrics=metrics,
        threshold=threshold,
        checkpoint_path=ckpt_path,
        confusion_matrix=confusion,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )
    default_out = None
    if metrics_out is None and split == "test":
        default_out = cfg.root / "artifacts" / "metrics" / "test_results.json"
    out_path = write_metrics_json(
        cfg, payload, split=split, metrics_out=metrics_out or default_out
    )
    log_checkpoint_summary(cfg, ckpt_path)
    manifest = finalize_run_manifest(cfg)
    if manifest is not None:
        print(f"  run manifest → {manifest}")

    result: dict[str, Any] = {
        **payload,
        "metrics_path": str(out_path),
        "loss": val_loss,
        **metrics,
        "checkpoint": str(ckpt_path),
    }
    print(f"Evaluation ({split}) — loss={val_loss:.4f} {format_metrics(metrics)}")
    print(f"  metrics written → {out_path}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MLP(H) on validation split.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "train", "test"), default="test")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="JSON path for metrics (default: artifacts/metrics/test_results.json for test)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    from src.config import load_config

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_evaluation(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        metrics_out=args.metrics_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
