"""Shared split I/O and metrics export for legacy model evaluation."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "Shared_pipeline_Files").is_dir():
            return candidate
    return here.parent


def load_dataset_paths(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    cfg_path = root / "Shared_pipeline_Files/data/dataset_paths.yaml"
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_split_paths(split_file: Path) -> list[str]:
    lines = split_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def label_from_rel_path(rel_path: str) -> int:
    parts = rel_path.replace("\\", "/").lower().split("/")
    if "malware" in parts:
        return 1
    if "benign" in parts:
        return 0
    raise ValueError(f"Cannot infer label from split path: {rel_path}")


def resolve_apk_path(apk_root: Path, rel_path: str) -> Path:
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return candidate
    return apk_root / rel_path


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
    return metrics


def write_test_results(
    *,
    out_path: Path,
    model_id: str,
    domain: str,
    split: str,
    metrics: dict[str, float | None],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    checkpoint_path: str,
    n_samples: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "domain": domain,
        "split": split,
        "n_samples": n_samples,
        "metrics": metrics,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "checkpoint": checkpoint_path,
        "class_counts": {
            "benign": int(np.sum(np.asarray(y_true) == 0)),
            "malware": int(np.sum(np.asarray(y_true) == 1)),
        },
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def export_offline_json(
    *,
    model_id: str,
    domain: str,
    split: str,
    metrics: dict[str, float | None],
    n_samples: int,
    threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    checkpoint_path: str,
    root: Path | None = None,
) -> Path:
    root = root or repo_root()
    tools_dir = root / "Shared_pipeline_Files/tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from offline_metrics_export import write_offline_metrics  # type: ignore

    numeric_metrics = {k: float(v) for k, v in metrics.items() if v is not None}
    return write_offline_metrics(
        model_id=model_id,
        split=split,
        metrics=numeric_metrics,
        n_samples=n_samples,
        threshold=threshold,
        confusion_matrix=confusion_matrix(y_true, y_pred),
        checkpoint_path=checkpoint_path,
        domain=domain,
        project_root=root / "Shared_pipeline_Files",
    )
