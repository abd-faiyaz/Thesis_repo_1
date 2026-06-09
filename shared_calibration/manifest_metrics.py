"""Stamp validation metrics from val_scores.json into export_manifest.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_val_metrics(val_scores_path: Path) -> dict[str, float]:
    payload = json.loads(Path(val_scores_path).read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    out: dict[str, float] = {}
    if "f1" in metrics and metrics["f1"] is not None:
        out["val_f1"] = float(metrics["f1"])
    if "accuracy" in metrics and metrics["accuracy"] is not None:
        out["val_accuracy"] = float(metrics["accuracy"])
    if not out:
        raise ValueError(f"No metrics.f1/accuracy in {val_scores_path}")
    return out


def stamp_export_manifest(
    manifest_path: Path,
    val_scores_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Merge val_f1 / val_accuracy from val_scores into export_manifest.json."""
    manifest_path = Path(manifest_path)
    metrics = load_val_metrics(val_scores_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = False
    for key, value in metrics.items():
        if manifest.get(key) != value:
            manifest[key] = value
            changed = True
    if changed and not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"path": str(manifest_path), "changed": changed, **metrics}
