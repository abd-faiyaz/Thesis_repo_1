"""Structured metrics and run logs for BM1 (artifacts/ + optional output_archives/)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.config import PipelineConfig

from src.data.dataloaders import resolve_split_settings

MODEL_ID = "mlp_header"
DOMAIN_ID = "dex_header_d3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics_dir(cfg: PipelineConfig) -> Path:
    path = cfg.root / "artifacts" / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_run_dir(cfg: PipelineConfig) -> Path | None:
    run_id = os.environ.get("BM1_RUN_ID", "").strip()
    if not run_id:
        return None
    root = cfg.root / "output_archives" / run_id
    for sub in ("logs", "metrics", "corpus_stats", "figures", "config", "export", "parity"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def mirror_to_archive(cfg: PipelineConfig, artifact_path: Path, archive_rel: str) -> Path | None:
    """Copy artifact into output_archives/<BM1_RUN_ID>/ if BM1_RUN_ID is set."""
    archive = archive_run_dir(cfg)
    if archive is None or not artifact_path.is_file():
        return None
    dest = archive / archive_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(artifact_path.read_bytes())
    return dest


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def reset_epochs_log(cfg: PipelineConfig, *, fresh: bool) -> Path:
    path = metrics_dir(cfg) / "epochs.jsonl"
    if fresh and path.is_file():
        path.unlink()
    return path


def log_epoch(
    cfg: PipelineConfig,
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    val_metrics: dict[str, float],
) -> None:
    record: dict[str, Any] = {
        "timestamp": _utc_now(),
        "epoch": epoch,
        "total_epochs": total_epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": learning_rate,
        "accuracy": val_metrics.get("accuracy"),
        "f1": val_metrics.get("f1"),
        "roc_auc": val_metrics.get("roc_auc"),
    }
    path = metrics_dir(cfg) / "epochs.jsonl"
    append_jsonl(path, record)
    mirror_to_archive(cfg, path, "metrics/epochs.jsonl")


def log_training_run_info(
    cfg: PipelineConfig,
    *,
    train_samples: int,
    val_samples: int,
    feature_dim: int,
    hidden_dim: int,
    start_epoch: int,
    total_epochs: int,
    fresh_start: bool,
    device: str,
) -> None:
    import torch

    payload: dict[str, Any] = {
        "timestamp": _utc_now(),
        "run_id": os.environ.get("BM1_RUN_ID"),
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "start_epoch": start_epoch,
        "total_epochs": total_epochs,
        "fresh_start": fresh_start,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "apk_root": os.environ.get("APK_ROOT", str(cfg.paths.apk_root)),
        "batch_size": int(cfg.data.get("batch_size", 16)),
        **resolve_split_settings(cfg),
    }
    path = metrics_dir(cfg) / "training_run_info.json"
    write_json(path, payload)
    mirror_to_archive(cfg, path, "metrics/training_run_info.json")


def log_preprocess_summary(
    cfg: PipelineConfig,
    summary: dict[str, Any],
    *,
    apk_root: Path,
) -> None:
    payload = {
        "timestamp": _utc_now(),
        "run_id": os.environ.get("BM1_RUN_ID"),
        "apk_root": str(apk_root),
        **summary,
    }
    path = metrics_dir(cfg) / "preprocess_summary.json"
    write_json(path, payload)
    mirror_to_archive(cfg, path, "metrics/preprocess_summary.json")


def build_metrics_payload(
    cfg: PipelineConfig,
    *,
    split: str,
    n_samples: int,
    loss: float,
    metrics: dict[str, float],
    threshold: float,
    checkpoint_path: Path,
    confusion_matrix: list[list[int]],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    import torch

    hardware: dict[str, Any] = {
        "device": str(cfg.training.get("device", "cuda")),
        "batch_size": int(cfg.data.get("batch_size", 16)),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        hardware["gpu_name"] = torch.cuda.get_device_name(0)

    payload: dict[str, Any] = {
        "timestamp": _utc_now(),
        "run_id": os.environ.get("BM1_RUN_ID"),
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "split": split,
        "n_samples": n_samples,
        "loss": loss,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
        },
        "threshold": threshold,
        "confusion_matrix": confusion_matrix,
        "checkpoint": str(checkpoint_path),
        "hardware": hardware,
        "apk_root": os.environ.get("APK_ROOT", str(cfg.paths.apk_root)),
        **resolve_split_settings(cfg),
    }
    if y_true is not None:
        payload["class_counts"] = {
            "benign": int(np.sum(y_true == 0)),
            "malware": int(np.sum(y_true == 1)),
        }
    return payload


def write_metrics_json(
    cfg: PipelineConfig,
    payload: dict[str, Any],
    *,
    split: str,
    metrics_out: Path | None = None,
) -> Path:
    out = metrics_out or (metrics_dir(cfg) / f"metrics_{split}.json")
    write_json(out, payload)
    mirror_to_archive(cfg, out, f"metrics/metrics_{split}.json")
    return out


def log_checkpoint_summary(cfg: PipelineConfig, checkpoint_path: Path) -> None:
    import torch

    if not checkpoint_path.is_file():
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = {
        "timestamp": _utc_now(),
        "checkpoint": str(checkpoint_path),
        "next_epoch": ckpt.get("next_epoch"),
        "train_loss": ckpt.get("train_loss"),
        "val_loss": ckpt.get("val_loss"),
        "val_metrics": ckpt.get("val_metrics"),
        "feature_dim": ckpt.get("feature_dim"),
        "hidden_dim": ckpt.get("hidden_dim"),
    }
    path = metrics_dir(cfg) / "checkpoint_summary.json"
    write_json(path, summary)
    mirror_to_archive(cfg, path, "metrics/checkpoint_summary.json")


def _git_commit_hash(cfg: PipelineConfig) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cfg.root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def finalize_run_manifest(cfg: PipelineConfig) -> Path | None:
    """Write RUN_MANIFEST.json into output_archives when BM1_RUN_ID is set."""
    archive = archive_run_dir(cfg)
    if archive is None:
        return None

    manifest: dict[str, Any] = {
        "run_id": os.environ.get("BM1_RUN_ID"),
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "created_at": _utc_now(),
        "git_commit": _git_commit_hash(cfg),
        "apk_root": os.environ.get("APK_ROOT", str(cfg.paths.apk_root)),
    }

    pre_path = metrics_dir(cfg) / "preprocess_summary.json"
    if pre_path.is_file():
        manifest["preprocessing"] = json.loads(pre_path.read_text(encoding="utf-8"))

    train_info_path = metrics_dir(cfg) / "training_run_info.json"
    if train_info_path.is_file():
        manifest["training"] = json.loads(train_info_path.read_text(encoding="utf-8"))

    test_metrics_path = metrics_dir(cfg) / "test_results.json"
    if test_metrics_path.is_file():
        test_data = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        manifest["final_test_metrics"] = test_data.get("metrics")
        manifest["final_test_loss"] = test_data.get("loss")
        manifest["n_test_samples"] = test_data.get("n_samples")

    ckpt_path = metrics_dir(cfg) / "checkpoint_summary.json"
    if ckpt_path.is_file():
        manifest["checkpoint_summary"] = json.loads(ckpt_path.read_text(encoding="utf-8"))

    manifest["artifact_paths"] = {
        "features_pt": str(cfg.paths.processed_dir / "dex_header_features.pt"),
        "checkpoint": str(cfg.paths.latest_checkpoint),
        "normalization": str(cfg.paths.normalization_stats),
        "metrics_dir": str(metrics_dir(cfg)),
    }

    out = archive / "RUN_MANIFEST.json"
    write_json(out, manifest)
    return out
