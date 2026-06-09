#!/usr/bin/env python3
"""Load model_plot_registry.json and normalize offline metric payloads."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def registry_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "Shared_pipeline_Files/data/model_plot_registry.json"


def load_registry(root: Path | None = None) -> dict[str, Any]:
    path = registry_path(root)
    return json.loads(path.read_text(encoding="utf-8"))


def registry_models(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return list(registry.get("models", []))


def model_by_id(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    for entry in registry_models(registry):
        if entry.get("model_id") == model_id:
            return entry
    raise KeyError(f"Unknown model_id in registry: {model_id}")


def csv_models(registry: dict[str, Any]) -> list[dict[str, Any]]:
    order = registry.get("plot_order", [])
    by_id = {m["model_id"]: m for m in registry_models(registry) if m.get("include_in_csv", True)}
    ordered = [by_id[mid] for mid in order if mid in by_id]
    for mid, entry in by_id.items():
        if mid not in order:
            ordered.append(entry)
    return ordered


def _metrics_block(payload: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    key = entry.get("offline_metrics_from", "metrics")
    if key == "metrics":
        block = payload.get("metrics")
        if isinstance(block, dict):
            return block
        # Checkpoint-style flat metrics (pattern A/B artifacts).
        flat = {
            k: payload.get(k)
            for k in ("accuracy", "f1", "roc_auc", "precision", "recall")
            if payload.get(k) is not None
        }
        if flat:
            return flat
        raise KeyError("No metrics block in payload")
    nested = payload.get(key)
    if not isinstance(nested, dict):
        raise KeyError(f"Expected payload[{key!r}] dict")
    return nested


def normalize_offline_payload(
    payload: dict[str, Any],
    entry: dict[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    metrics_raw = _metrics_block(payload, entry)
    metrics: dict[str, float] = {}
    for key in ("accuracy", "f1", "roc_auc", "precision", "recall"):
        val = metrics_raw.get(key)
        if val is not None:
            metrics[key] = float(val)

    if "roc_auc" not in metrics:
        raise ValueError(f"roc_auc missing for {entry['model_id']} in {source_path}")

    n_samples = payload.get("n_samples")
    if n_samples is None:
        raise ValueError(f"n_samples missing for {entry['model_id']} in {source_path}")

    split = payload.get("split", "test")
    domain = payload.get("domain") or entry.get("domain")
    threshold = payload.get("threshold")
    if threshold is None and entry.get("offline_metrics_from") == "mode_a":
        threshold = metrics_raw.get("threshold")

    out: dict[str, Any] = {
        "model_id": entry["model_id"],
        "domain": domain,
        "split": split,
        "n_samples": int(n_samples),
        "metrics": metrics,
        "source_path": str(source_path),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
    if payload.get("confusion_matrix") is not None:
        out["confusion_matrix"] = payload["confusion_matrix"]
    elif metrics_raw.get("confusion_matrix") is not None:
        out["confusion_matrix"] = metrics_raw["confusion_matrix"]
    if payload.get("split_mode"):
        out["split_mode"] = payload["split_mode"]
    if entry.get("comments_extra"):
        out["comments_extra"] = entry["comments_extra"]
    return out


def find_test_results_source(root: Path, entry: dict[str, Any]) -> Path | None:
    for rel in entry.get("test_results_candidates", []):
        path = root / rel
        if path.is_file():
            return path

    model_id = entry["model_id"]
    offline_dir = root / "Shared_pipeline_Files/results/offline"
    if offline_dir.is_dir():
        matches = sorted(
            offline_dir.glob(f"{model_id}_test_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


def build_comments(registry: dict[str, Any], entry: dict[str, Any], normalized: dict[str, Any]) -> str:
    """Short pros/cons blurb for extended-abstract table (5–6 words)."""
    _ = registry, normalized
    comment = entry.get("comment", "")
    if comment:
        return str(comment)
    return str(entry.get("method", entry.get("model_id", "")))


def validate_registry(root: Path | None = None) -> list[str]:
    """Return list of validation errors (empty if OK)."""
    root = root or repo_root()
    registry = load_registry(root)
    errors: list[str] = []

    plot_order = registry.get("plot_order", [])
    model_ids = {m["model_id"] for m in registry_models(registry)}
    for mid in plot_order:
        if mid not in model_ids:
            errors.append(f"plot_order references unknown model_id: {mid}")

    required = ("model_id", "method", "features", "domain")
    for entry in registry_models(registry):
        for key in required:
            if not entry.get(key):
                errors.append(f"{entry.get('model_id', '?')}: missing {key}")
        if entry.get("include_in_csv") and not find_test_results_source(root, entry):
            errors.append(f"{entry['model_id']}: no test_results or offline export found")

    offline_domains = {m.get("domain") for m in registry_models(registry)}
    for stage_id in registry.get("device_stage_ids", []):
        if stage_id in model_ids:
            continue
        if stage_id in {"mldp_dexheader_cascade_mode_a", "mldp_dexheader_cascade_mode_b"}:
            continue
        if stage_id not in model_ids:
            errors.append(f"device_stage_ids {stage_id!r} has no matching offline registry row")

    _ = offline_domains  # reserved for future domain cross-checks
    return errors
