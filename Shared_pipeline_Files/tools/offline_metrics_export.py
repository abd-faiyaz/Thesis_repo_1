#!/usr/bin/env python3
"""Write offline evaluation JSON to Shared_pipeline_Files/results/offline/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def shared_results_root(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "results" / "offline"


def write_offline_metrics(
    *,
    model_id: str,
    split: str,
    metrics: dict[str, float],
    n_samples: int,
    threshold: float,
    confusion_matrix: list[list[int]] | None = None,
    checkpoint_path: str | None = None,
    domain: str | None = None,
    hardware: dict[str, Any] | None = None,
    project_root: Path | None = None,
    run_id: str | None = None,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{model_id}_{split}_{ts}"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "model_id": model_id,
        "split": split,
        "n_samples": n_samples,
        "metrics": metrics,
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if domain:
        payload["domain"] = domain
    if confusion_matrix is not None:
        payload["confusion_matrix"] = confusion_matrix
    if checkpoint_path:
        payload["checkpoint_path"] = checkpoint_path
    if hardware:
        payload["hardware"] = hardware

    out_dir = shared_results_root(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_id}_{split}_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
