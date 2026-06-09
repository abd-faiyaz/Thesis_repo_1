"""Thesis pipeline hooks (Shared_pipeline_Files, offline eval JSON, optional shared splits)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
    """Walk parents until thesis repo root (contains Shared_pipeline_Files/)."""
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
    shared = find_shared_pipeline_root(cfg.root)

    shared_splits = _resolve_optional_path(
        repo or cfg.root,
        raw.get("shared_splits_dir"),
    )
    shared_manifest = _resolve_optional_path(
        repo or cfg.root,
        raw.get("shared_manifest_csv"),
    )
    export_dir = _resolve_optional_path(repo or cfg.root, raw.get("export_dir"))

    return PipelineSettings(
        enabled=bool(raw.get("enabled", True)),
        model_id=str(raw.get("model_id", "dual_branch_dex_manifest")),
        domain=str(raw.get("domain", "dex_header_manifest_dual")),
        use_shared_splits=bool(raw.get("use_shared_splits", False)),
        shared_splits_dir=shared_splits,
        shared_manifest_csv=shared_manifest,
        export_offline_json=bool(raw.get("export_offline_json", True)),
        export_dir=export_dir,
    )


def read_split_apk_paths(split_file: Path) -> list[str]:
    lines = split_file.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _build_path_lookups(
    rows: list[Any],
    apk_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_rel: dict[str, Any] = {}
    by_abs: dict[str, Any] = {}
    root = apk_root.resolve()
    for row in rows:
        by_abs[row.apk_path.resolve().as_posix()] = row
        try:
            by_rel[row.apk_path.resolve().relative_to(root).as_posix()] = row
        except ValueError:
            pass
    return by_rel, by_abs


def resolve_row_for_split_path(
    path_str: str,
    apk_root: Path,
    by_rel: dict[str, Any],
    by_abs: dict[str, Any],
) -> Any:
    normalized = path_str.strip().replace("\\", "/")
    if not normalized:
        raise KeyError("empty split path")
    candidate = Path(normalized)
    if candidate.is_absolute():
        key = candidate.resolve().as_posix()
        if key in by_abs:
            return by_abs[key]
    rel = normalized.lstrip("./")
    if rel in by_rel:
        return by_rel[rel]
    full = (apk_root / rel).resolve().as_posix()
    if full in by_abs:
        return by_abs[full]
    raise KeyError(f"Split path not found under apk_root: {path_str}")


def partition_rows_from_shared_paths(
    rows: list[Any],
    train_paths: list[str],
    val_paths: list[str],
    apk_root: Path,
    *,
    test_paths: list[str] | None = None,
) -> tuple[list[Any], list[Any], list[Any] | None, list[Any]]:
    by_rel, by_abs = _build_path_lookups(rows, apk_root)
    train_rows = [
        resolve_row_for_split_path(p, apk_root, by_rel, by_abs) for p in train_paths
    ]
    val_rows = [
        resolve_row_for_split_path(p, apk_root, by_rel, by_abs) for p in val_paths
    ]
    test_rows: list[Any] | None = None
    if test_paths:
        test_rows = [
            resolve_row_for_split_path(p, apk_root, by_rel, by_abs) for p in test_paths
        ]
    seen: set[str] = set()
    indexed: list[Any] = []
    for row in train_rows + val_rows + (test_rows or []):
        if row.apk_id not in seen:
            seen.add(row.apk_id)
            indexed.append(row)
    return train_rows, val_rows, test_rows, indexed


def load_shared_split_paths(
    settings: PipelineSettings,
) -> tuple[list[str], list[str], list[str] | None] | None:
    if not settings.use_shared_splits or settings.shared_splits_dir is None:
        return None
    train_file = settings.shared_splits_dir / "train.txt"
    val_file = settings.shared_splits_dir / "val.txt"
    test_file = settings.shared_splits_dir / "test.txt"
    if not train_file.is_file() or not val_file.is_file():
        return None
    train_paths = read_split_apk_paths(train_file)
    val_paths = read_split_apk_paths(val_file)
    test_paths = read_split_apk_paths(test_file) if test_file.is_file() else None
    return train_paths, val_paths, test_paths


def load_shared_train_val_paths(settings: PipelineSettings) -> tuple[list[str], list[str]] | None:
    shared = load_shared_split_paths(settings)
    if shared is None:
        return None
    train_paths, val_paths, _test_paths = shared
    return train_paths, val_paths


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[list[int]]:
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


def export_offline_evaluation(
    cfg: PipelineConfig,
    *,
    split: str,
    metrics: dict[str, float],
    n_samples: int,
    threshold: float,
    checkpoint_path: Path,
    confusion_matrix: list[list[int]] | None,
    val_loss: float | None = None,
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
        "batch_size": int(cfg.training.get("batch_size", cfg.data.get("batch_size", 16))),
    }
    if val_loss is not None:
        hardware["val_loss"] = val_loss

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
