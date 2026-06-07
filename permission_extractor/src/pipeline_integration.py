"""Thesis pipeline hooks (Shared_pipeline_Files, offline eval JSON)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.config import PipelineConfig

_SHARED_DIR_NAME = "Shared_pipeline_Files"


@dataclass(frozen=True)
class PipelineSettings:
    enabled: bool
    model_id: str
    domain: str
    use_shared_splits: bool
    shared_splits_dir: Path | None
    shared_manifest_csv: Path | None
    export_offline_json: bool
    export_dir: Path | None


def find_repo_root(start: Path | None = None) -> Path | None:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        if (parent / _SHARED_DIR_NAME).is_dir():
            return parent
    return None


def find_shared_pipeline_root(start: Path | None = None) -> Path | None:
    repo = find_repo_root(start)
    if repo is None:
        return None
    return repo / _SHARED_DIR_NAME


def _resolve_optional_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def get_pipeline_settings(cfg: PipelineConfig) -> PipelineSettings:
    raw = cfg.raw.get("pipeline", {})
    repo = find_repo_root(cfg.root)

    return PipelineSettings(
        enabled=bool(raw.get("enabled", True)),
        model_id=str(raw.get("model_id", "mldp_pruned_permission")),
        domain=str(raw.get("domain", "manifest_permissions_mldp")),
        use_shared_splits=bool(raw.get("use_shared_splits", False)),
        shared_splits_dir=_resolve_optional_path(
            repo or cfg.root, raw.get("shared_splits_dir")
        ),
        shared_manifest_csv=_resolve_optional_path(
            repo or cfg.root, raw.get("shared_manifest_csv")
        ),
        export_offline_json=bool(raw.get("export_offline_json", True)),
        export_dir=_resolve_optional_path(repo or cfg.root, raw.get("export_dir")),
    )


def resolve_split_settings(cfg: PipelineConfig) -> dict[str, Any]:
    """Summarize split policy for metrics JSON (thesis eval protocol)."""
    pre = cfg.preprocessing
    split_mode = str(pre.get("split_mode", "stratified_development"))
    dev_years = pre.get("development_years", [2020, 2021])
    holdout_years = pre.get("temporal_holdout_years", [2022, 2023])
    return {
        "split_mode": split_mode,
        "train_years": list(dev_years),
        "test_years": list(holdout_years),
        "development_years": list(dev_years),
        "temporal_holdout_years": list(holdout_years),
        "train_ratio": pre.get("train_ratio"),
        "val_ratio": pre.get("val_ratio"),
        "dev_test_ratio": pre.get("dev_test_ratio"),
        "seed": pre.get("random_seed", 42),
        "splits_dir": str(cfg.paths.splits_dir),
    }


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def write_local_metrics_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_test_results_payload(
    cfg: PipelineConfig,
    *,
    split_result: dict[str, Any],
    threshold: float,
    checkpoint_path: Path,
    model_id: str,
    domain: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "domain": domain,
        "split": "test",
        "n_samples": split_result["n_samples"],
        "metrics": split_result["metrics"],
        "threshold": threshold,
        "confusion_matrix": split_result["confusion_matrix"],
        "checkpoint": str(checkpoint_path),
        **resolve_split_settings(cfg),
    }
    if extra:
        payload.update(extra)
    return payload


def export_offline_evaluation(
    cfg: PipelineConfig,
    *,
    split: str,
    metrics: dict[str, float],
    n_samples: int,
    threshold: float,
    checkpoint_path: Path,
    confusion_matrix: list[list[int]] | None,
) -> Path | None:
    settings = get_pipeline_settings(cfg)
    if not settings.enabled or not settings.export_offline_json:
        return None

    shared = find_shared_pipeline_root(cfg.root)
    if shared is None:
        print("  pipeline: Shared_pipeline_Files not found — skip offline JSON export")
        return None

    tools_dir = shared / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    try:
        from offline_metrics_export import write_offline_metrics  # type: ignore
    except ImportError:
        print("  pipeline: offline_metrics_export.py not importable — skip shared export")
        return None

    hardware: dict[str, Any] = {
        "device": str(cfg.training.get("device", "cpu")),
    }

    out = write_offline_metrics(
        model_id=settings.model_id,
        split=split,
        metrics=metrics,
        n_samples=n_samples,
        threshold=threshold,
        confusion_matrix=confusion_matrix,
        checkpoint_path=str(checkpoint_path),
        domain=settings.domain,
        hardware=hardware,
        project_root=shared,
    )
    print(f"  pipeline offline JSON → {out}")
    return out
