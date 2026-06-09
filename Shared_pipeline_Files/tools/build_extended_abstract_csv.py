#!/usr/bin/env python3
"""Build Illustrations_templates extended-abstract CSV from registry + offline/device metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from plot_registry_lib import (  # noqa: E402
    build_comments,
    csv_models,
    load_registry,
    repo_root,
)


CSV_COLUMNS = [
    "Method",
    "Features",
    "Accuracy",
    "F1",
    "ROC-AUC",
    "CPU",
    "Memory",
    "Battery",
    "Total Time  (Avg)",
    "Device Feasibility",
    "Comments",
]

# Scan A battery drain is ~10^-3 %; 4 d.p. + leading ' keeps Excel from rounding to 0.0%.
BATTERY_PCT_DECIMALS = 4


def _format_battery_pct(pct: float) -> str:
    text = f"{pct:.{BATTERY_PCT_DECIMALS}f}%"
    return f"'{text}"


def _round4(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _load_latest_offline(latest_dir: Path, model_id: str) -> dict[str, Any] | None:
    path = latest_dir / f"{model_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plot_table(plot_table: Path) -> dict[str, Any] | None:
    if not plot_table.is_file():
        return None
    return json.loads(plot_table.read_text(encoding="utf-8"))


def _device_row(table: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for row in table.get("models", []):
        if row.get("model_id") == model_id:
            return row
    return None


def _table_has_device_data(table: dict[str, Any]) -> bool:
    """True when plot_metrics_table includes Scan A device medians."""
    if table.get("offline_only"):
        return False
    if int(table.get("n_scans_scan_a") or 0) > 0:
        return True
    for row in table.get("models", []):
        scan_a = row.get("device_scan_a") or {}
        if int(scan_a.get("n_stage_samples") or 0) > 0:
            return True
    return False


def build_rows(
    registry: dict[str, Any],
    *,
    latest_dir: Path,
    plot_table: Path | None,
    offline_only: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    table = _load_plot_table(plot_table) if plot_table is not None else None
    use_device = table is not None and _table_has_device_data(table)
    for entry in csv_models(registry):
        model_id = entry["model_id"]
        offline = _load_latest_offline(latest_dir, model_id)
        if offline is None:
            raise FileNotFoundError(
                f"Missing offline latest JSON for {model_id}: {latest_dir / (model_id + '.json')}"
            )

        metrics = offline.get("metrics", {})
        comments = build_comments(registry, entry, offline)

        cpu = memory = battery = total_time = feasibility = ""
        if use_device:
            device = _device_row(table, model_id)
            if device:
                scan_a = device.get("device_scan_a") or {}
                derived = device.get("derived") or {}
                if scan_a.get("cpu_ms") is not None:
                    cpu = f"{float(scan_a['cpu_ms']):.1f} ms"
                if scan_a.get("mem_mb") is not None:
                    memory = f"{float(scan_a['mem_mb']):.2f} MB"
                if scan_a.get("battery_pct_delta") is not None:
                    battery = _format_battery_pct(float(scan_a["battery_pct_delta"]))
                if scan_a.get("stage_total_ms") is not None:
                    total_time = f"{float(scan_a['stage_total_ms']):.1f} ms"
                feasibility = derived.get("device_feasibility", "")

        rows.append(
            {
                "Method": entry["method"],
                "Features": entry["features"],
                "Accuracy": _round4(metrics.get("accuracy")),
                "F1": _round4(metrics.get("f1")),
                "ROC-AUC": _round4(metrics.get("roc_auc")),
                "CPU": cpu,
                "Memory": memory,
                "Battery": battery,
                "Total Time  (Avg)": total_time,
                "Device Feasibility": feasibility,
                "Comments": comments,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--latest-dir",
        type=Path,
        default=None,
        help="Directory with per-model latest offline JSON",
    )
    parser.add_argument(
        "--plot-table",
        type=Path,
        default=None,
        help="plot_metrics_table.json for device columns (Phase 5)",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Fill ACC/F1/ROC-AUC/Comments only; leave device columns empty",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    registry = load_registry(root)
    latest_dir = (args.latest_dir or root / "Shared_pipeline_Files/results/offline/latest").resolve()
    out = args.out or (
        root / "Illustrations_templates/On-Device ML-Experiments - Sheet1-generated.csv"
    )
    plot_table = args.plot_table
    default_table = root / "Shared_pipeline_Files/results/figures/plot_metrics_table.json"
    if plot_table is None and default_table.is_file():
        plot_table = default_table

    rows = build_rows(
        registry,
        latest_dir=latest_dir,
        plot_table=plot_table,
        offline_only=args.offline_only,
    )
    write_csv(out.resolve(), rows)
    table = _load_plot_table(plot_table) if plot_table and plot_table.is_file() else None
    device_note = ""
    if table and _table_has_device_data(table):
        device_note = f" (device cols from {plot_table.name})"
    elif args.offline_only:
        device_note = " (offline-only — no Scan A table)"
    print(f"Wrote {len(rows)} row(s) → {out}{device_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
