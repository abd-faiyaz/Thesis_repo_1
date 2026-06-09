"""Validation metrics: accuracy, F1, AUC (Phase 6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from shared_calibration import (
    build_val_thresholds_payload,
    find_repo_root,
    format_cascade_band_summary,
    write_split_scores_bundle,
    write_thresholds,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from src.config import PipelineConfig
    from src.models.combined_net import CombinedNet

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
    model: CombinedNet,
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
    model: CombinedNet,
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


def _metrics_thresholds_path(cfg: PipelineConfig) -> Path:
    return cfg.root / "artifacts" / "metrics" / "thresholds.json"


def _manifest_apk_ids(cfg: PipelineConfig, split: str) -> list[str]:
    from src.data.store import load_shard_manifest

    if split == "val":
        manifest_path = cfg.paths.manifest_val
    elif split == "test":
        manifest_path = cfg.paths.manifest_test
    else:
        manifest_path = cfg.paths.manifest_train
    manifest = load_shard_manifest(manifest_path)
    return [entry.apk_id for entry in manifest.entries]


def export_split_scores(
    cfg: PipelineConfig,
    model: CombinedNet,
    loader: DataLoader,
    device: torch.device,
    *,
    split: str,
    threshold: float,
) -> Path | None:
    from src.pipeline_integration import get_pipeline_settings

    y_true, _, y_score = collect_predictions(model, loader, device, threshold=threshold)
    settings = get_pipeline_settings(cfg)
    out = write_split_scores_bundle(
        model_id=settings.model_id,
        split=split,
        metrics_dir=cfg.root / "artifacts" / "metrics",
        apk_ids=_manifest_apk_ids(cfg, split),
        labels=y_true,
        scores=y_score,
        threshold=threshold,
        repo_root=find_repo_root(cfg.root),
        sync_val_to_workspace=split == "val",
    )
    print(f"  {split} scores → {out}")
    return out


def write_val_thresholds(
    cfg: PipelineConfig,
    model: CombinedNet,
    val_loader: DataLoader,
    device: torch.device,
    *,
    tune_on_val: bool | None = None,
    calibrate_bands: bool = True,
    out_path: Path | None = None,
) -> dict:
    eval_cfg = cfg.evaluation
    default_threshold = float(eval_cfg.get("threshold", 0.5))
    do_tune = (
        bool(eval_cfg.get("tune_threshold_on_val", True))
        if tune_on_val is None
        else tune_on_val
    )

    y_true, _, y_score = collect_predictions(
        model, val_loader, device, threshold=default_threshold
    )
    from src.pipeline_integration import get_pipeline_settings

    settings = get_pipeline_settings(cfg)
    payload = build_val_thresholds_payload(
        model_id=settings.model_id,
        y_true=y_true,
        scores=y_score,
        default=default_threshold,
        tune=do_tune,
        calibrate_bands=calibrate_bands,
        cascade_targets=cfg.raw.get("cascade", {}),
        extra={
            "description": "Predict malware when malware_probability >= tuned_val",
        },
    )
    tuned_threshold = float(payload["tuned_val"])
    payload["benign_threshold"] = 1.0 - tuned_threshold
    metrics_dir = cfg.root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = out_path or _metrics_thresholds_path(cfg)
    write_thresholds(thresholds_path, payload)
    band_summary = format_cascade_band_summary(payload)
    print(
        f"  val-tuned threshold={tuned_threshold:.4f}"
        + (f"  cascade {band_summary}" if band_summary else "")
        + f" → {thresholds_path}"
    )
    write_split_scores_bundle(
        model_id=settings.model_id,
        split="val",
        metrics_dir=cfg.root / "artifacts" / "metrics",
        apk_ids=_manifest_apk_ids(cfg, "val"),
        labels=y_true,
        scores=y_score,
        threshold=tuned_threshold,
        repo_root=find_repo_root(cfg.root),
    )
    return payload


def _tune_val_threshold(
    cfg: PipelineConfig,
    model: CombinedNet,
    val_loader: DataLoader,
    device: torch.device,
    *,
    tune_on_val: bool | None = None,
) -> float:
    payload = write_val_thresholds(
        cfg, model, val_loader, device, tune_on_val=tune_on_val
    )
    return float(payload["tuned_val"])


def run_evaluation(
    cfg: PipelineConfig,
    *,
    checkpoint_path: Path | None = None,
    split: str = "val",
    metrics_out: Path | None = None,
    tune_on_val: bool | None = None,
) -> dict[str, Any]:
    from src.config import ensure_artifact_dirs
    from src.data.dataloaders import build_dataloaders_from_config, build_test_loader_from_config
    from src.models.combined_net import build_combined_net_from_config
    from src.training.checkpoint import load_checkpoint, load_model_from_checkpoint
    from src.training.losses import build_criterion
    from src.training.setup import build_training_objects, resolve_device

    ensure_artifact_dirs(cfg)
    train_loader, val_loader, _, _ = build_dataloaders_from_config(cfg)
    if split == "test":
        loader, _, _ = build_test_loader_from_config(cfg)
    elif split == "val":
        loader = val_loader
    else:
        loader = train_loader

    ckpt_path = checkpoint_path or cfg.paths.best_checkpoint
    if not ckpt_path.is_file():
        ckpt_path = cfg.paths.latest_checkpoint

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    model = build_combined_net_from_config(cfg)
    load_model_from_checkpoint(checkpoint, model)
    device = resolve_device(str(cfg.training.get("device", "cpu")))
    model.to(device)
    criterion = build_criterion(cfg, device)

    threshold = _tune_val_threshold(
        cfg, model, val_loader, device, tune_on_val=tune_on_val
    )
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
    from src.pipeline_integration import (
        build_confusion_matrix,
        export_offline_evaluation,
        get_pipeline_settings,
        write_local_metrics_json,
    )

    confusion = build_confusion_matrix(y_true, y_pred)
    n_samples = int(len(y_true))

    result: dict[str, Any] = {
        "split": split,
        "loss": val_loss,
        **metrics,
        "checkpoint": str(ckpt_path),
        "threshold": threshold,
        "n_samples": n_samples,
        "confusion_matrix": confusion,
    }
    print(f"Evaluation ({split}) — loss={val_loss:.4f} {format_metrics(metrics)}")

    if split == "test" and metrics_out is None:
        out = cfg.paths.checkpoint_dir / "test_results.json"
    else:
        out = metrics_out or (cfg.paths.checkpoint_dir / f"metrics_{split}.json")
    write_local_metrics_json(out, result)
    print(f"  metrics written → {out}")
    if split != "val":
        export_split_scores(cfg, model, loader, device, split=split, threshold=threshold)

    try:
        from src.thesis_archive import after_eval

        after_eval(out)
    except ImportError:
        pass

    settings = get_pipeline_settings(cfg)
    export_offline_evaluation(
        cfg,
        split=split,
        metrics=metrics,
        n_samples=n_samples,
        threshold=threshold,
        checkpoint_path=ckpt_path,
        confusion_matrix=confusion,
        val_loss=val_loss,
    )

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CombinedNet on cached shards."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "train", "test"), default="test")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="JSON path for metrics (default: artifacts/checkpoints/metrics_{split}.json)",
    )
    parser.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Skip val max-F1 threshold tuning; use evaluation.threshold from config.",
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
        tune_on_val=not args.no_tune_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
