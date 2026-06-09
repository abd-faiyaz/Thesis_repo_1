#!/usr/bin/env python3
"""Merge offline test metrics + Scan A/B device pulls into plot_metrics_table.json.

Usage:
  aggregate_plot_metrics.py
  aggregate_plot_metrics.py --scan-a path --scan-b path --offline-dir path
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from device_metrics_lib import (  # noqa: E402
    COST_SCORE_FORMULA,
    SIZE_BUCKET_EDGES_MB,
    build_per_apk_series,
    build_plot_metrics_table_scan_a,
    build_size_bucket_medians,
    compute_cost_scores,
    compute_feasibility_ranks,
    filter_scans_by_cascade_mode,
    load_device_records,
    repo_root,
    resolve_device_metrics_path,
    summarize_scan_b,
)
from plot_registry_lib import csv_models, load_registry  # noqa: E402

REQUIRED_OFFLINE = ("accuracy", "f1", "roc_auc")
REQUIRED_SCAN_A = (
    "parse_ms",
    "vectorize_ms",
    "inference_ms",
    "cpu_ms",
    "stage_total_ms",
    "mem_mb",
)
REQUIRED_DERIVED = ("f1_pct", "cost_score", "device_feasibility")


def load_offline_latest(latest_dir: Path, registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in csv_models(registry):
        model_id = entry["model_id"]
        path = latest_dir / f"{model_id}.json"
        if not path.is_file():
            continue
        out[model_id] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _attach_offline(
    models: list[dict[str, Any]],
    registry: dict[str, Any],
    offline_by_model: dict[str, dict[str, Any]],
) -> None:
    entry_by_id = {e["model_id"]: e for e in csv_models(registry)}
    for row in models:
        model_id = row["model_id"]
        entry = entry_by_id.get(model_id, {})
        row["method"] = entry.get("method", "")
        row["features"] = entry.get("features", "")
        offline = offline_by_model.get(model_id)
        if offline:
            row["offline"] = {
                "accuracy": offline.get("metrics", {}).get("accuracy"),
                "f1": offline.get("metrics", {}).get("f1"),
                "roc_auc": offline.get("metrics", {}).get("roc_auc"),
                "n_samples": offline.get("n_samples"),
                "split": offline.get("split"),
                "source_path": offline.get("source_path"),
            }
            f1 = row["offline"].get("f1")
            if f1 is not None:
                row.setdefault("derived", {})["f1_pct"] = float(f1) * 100.0


def build_offline_only_table(
    *,
    registry: dict[str, Any],
    offline_by_model: dict[str, dict[str, Any]],
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    models: list[dict[str, Any]] = []
    for entry in csv_models(registry):
        model_id = entry["model_id"]
        offline = offline_by_model.get(model_id)
        row: dict[str, Any] = {
            "model_id": model_id,
            "method": entry.get("method", ""),
            "features": entry.get("features", ""),
            "device_scan_a": {"n_stage_samples": 0},
            "derived": {},
        }
        if offline:
            metrics = offline.get("metrics", {})
            row["offline"] = {
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "n_samples": offline.get("n_samples"),
                "split": offline.get("split"),
                "source_path": offline.get("source_path"),
            }
            if metrics.get("f1") is not None:
                row["derived"]["f1_pct"] = float(metrics["f1"]) * 100.0
        models.append(row)
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "n_scans_scan_a": 0,
        "n_scans_scan_b": 0,
        "models": models,
        "per_apk_series": [],
        "size_bucket_medians": {},
    }


def build_aggregate_table(
    *,
    registry: dict[str, Any],
    offline_by_model: dict[str, dict[str, Any]],
    scan_a_scans: list[dict[str, Any]],
    scan_a_sessions: list[dict[str, Any]],
    scan_b_scans: list[dict[str, Any]] | None = None,
    scan_b_sessions: list[dict[str, Any]] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    scan_a_active = filter_scans_by_cascade_mode(
        scan_a_scans, scan_a_sessions, cascade_enabled=False
    )

    base = build_plot_metrics_table_scan_a(
        scan_a_active, scan_a_sessions, root=root
    )
    models = base["models"]
    _attach_offline(models, registry, offline_by_model)

    # Recompute cost scores and feasibility after offline attach.
    cost_scores = compute_cost_scores(models)
    feas_ranks = compute_feasibility_ranks(models, root=root)
    for row in models:
        derived = row.setdefault("derived", {})
        mid = row["model_id"]
        if mid in cost_scores:
            derived["cost_score"] = cost_scores[mid]
        if mid in feas_ranks:
            derived["device_feasibility"] = feas_ranks[mid]

    table: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_score": {
            "formula": COST_SCORE_FORMULA,
            "weights": {
                "stage_total_ms": 1.0,
                "cpu_ms": 1.0,
                "mem_mb": 1.0,
                "scale_range": [0.1, 100.0],
            },
        },
        "size_bucket_edges_mb": SIZE_BUCKET_EDGES_MB,
        "n_scans_scan_a": len(scan_a_active),
        "models": models,
        "per_apk_series": build_per_apk_series(scan_a_active, root=root),
        "size_bucket_medians": build_size_bucket_medians(scan_a_active, root=root),
    }

    if scan_b_scans is not None:
        scan_b_sessions = scan_b_sessions or []
        scan_b_active = filter_scans_by_cascade_mode(
            scan_b_scans, scan_b_sessions, cascade_enabled=True
        )
        table["n_scans_scan_b"] = len(scan_b_active)
        table["device_scan_b"] = summarize_scan_b(scan_b_active)

    return table


def missing_required_fields(table: dict[str, Any]) -> list[str]:
    if table.get("offline_only"):
        errors: list[str] = []
        for row in table.get("models", []):
            offline = row.get("offline") or {}
            for key in REQUIRED_OFFLINE:
                if offline.get(key) is None:
                    errors.append(f"{row.get('model_id', '?')}: offline.{key} missing")
        return errors

    errors: list[str] = []
    for row in table.get("models", []):
        model_id = row.get("model_id", "?")
        offline = row.get("offline") or {}
        for key in REQUIRED_OFFLINE:
            if offline.get(key) is None:
                errors.append(f"{model_id}: offline.{key} missing")
        scan_a = row.get("device_scan_a") or {}
        if scan_a.get("n_stage_samples", 0) > 0:
            for key in REQUIRED_SCAN_A:
                if scan_a.get(key) is None:
                    errors.append(f"{model_id}: device_scan_a.{key} missing")
            derived = row.get("derived") or {}
            for key in REQUIRED_DERIVED:
                if derived.get(key) in (None, ""):
                    errors.append(f"{model_id}: derived.{key} missing")
        elif scan_a.get("n_stage_samples", 0) == 0:
            errors.append(f"{model_id}: no Scan A stage samples")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-dir",
        type=Path,
        default=None,
        help="Directory with results/offline/latest/{model_id}.json",
    )
    parser.add_argument(
        "--scan-a-dir",
        type=Path,
        default=None,
        help="Scan A pull directory (scan_a_all_models)",
    )
    parser.add_argument(
        "--scan-b-dir",
        type=Path,
        default=None,
        help="Scan B pull directory (scan_b_cascade)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output plot_metrics_table.json path",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write table even when required fields are missing (pre-phone E2E)",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Offline metrics only (no device scan directories required)",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    registry = load_registry(root)
    latest_dir = (
        args.offline_dir or root / "Shared_pipeline_Files/results/offline/latest"
    ).resolve()
    scan_a_dir = (
        args.scan_a_dir or root / "Shared_pipeline_Files/results/device/scan_a_all_models"
    ).resolve()
    scan_b_dir = (
        args.scan_b_dir or root / "Shared_pipeline_Files/results/device/scan_b_cascade"
    ).resolve()
    out = (
        args.out or root / "Shared_pipeline_Files/results/figures/plot_metrics_table.json"
    ).resolve()

    offline_by_model = load_offline_latest(latest_dir, registry)
    if not offline_by_model:
        print(f"No offline JSON in {latest_dir} — run collect_offline_test_metrics.py", file=sys.stderr)
        return 1

    scan_a_path = resolve_device_metrics_path(scan_a_dir)
    scan_b_path = resolve_device_metrics_path(scan_b_dir)

    if args.offline_only or scan_a_path is None:
        if scan_a_path is None and not args.offline_only and not args.allow_partial:
            print(f"Scan A metrics not found under {scan_a_dir}", file=sys.stderr)
            return 1
        table = build_offline_only_table(
            registry=registry,
            offline_by_model=offline_by_model,
            root=root,
        )
        table["sources"] = {
            "offline_dir": str(latest_dir),
            "scan_a": str(scan_a_path) if scan_a_path else None,
            "scan_b": str(scan_b_path) if scan_b_path else None,
        }
    else:
        scan_a_scans, scan_a_sessions = load_device_records(scan_a_path)
        scan_b_scans: list[dict[str, Any]] | None = None
        scan_b_sessions: list[dict[str, Any]] | None = None
        if scan_b_path is not None:
            scan_b_scans, scan_b_sessions = load_device_records(scan_b_path)
        else:
            print(f"Note: Scan B metrics not found under {scan_b_dir} (optional).")

        table = build_aggregate_table(
            registry=registry,
            offline_by_model=offline_by_model,
            scan_a_scans=scan_a_scans,
            scan_a_sessions=scan_a_sessions,
            scan_b_scans=scan_b_scans,
            scan_b_sessions=scan_b_sessions,
            root=root,
        )
        table["sources"] = {
            "offline_dir": str(latest_dir),
            "scan_a": str(scan_a_path),
            "scan_b": str(scan_b_path) if scan_b_path else None,
        }

    errors = missing_required_fields(table)
    if errors and not args.allow_partial:
        print("Aggregation incomplete:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(
        f"  models={len(table.get('models', []))}, "
        f"scan_a={table.get('n_scans_scan_a')}, "
        f"scan_b={table.get('n_scans_scan_b', 0)}, "
        f"per_apk={len(table.get('per_apk_series', []))}"
    )
    if errors:
        print(f"  warnings: {len(errors)} missing field(s) (used --allow-partial or partial device data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
