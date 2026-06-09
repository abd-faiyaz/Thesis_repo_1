#!/usr/bin/env python3
"""Validate Scan A (ablation) device metrics pulled from phone.

Usage:
  validate_scan_a.py path/to/all_scan_metrics.jsonl [--min-scans 400] [--write-table]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from device_metrics_lib import (  # noqa: E402
    build_plot_metrics_table_scan_a,
    expected_ablation_stage_ids,
    load_scan_records,
    plot_order_model_ids,
    repo_root,
)


def _active_scans(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in scans if not s.get("dedup_skipped")]


def validate_scan_a(
    scans: list[dict[str, Any]],
    *,
    min_scans: int,
    min_stages: int,
    require_battery: bool,
) -> list[str]:
    errors: list[str] = []
    active = _active_scans(scans)
    if len(active) < min_scans:
        errors.append(f"expected at least {min_scans} scans, got {len(active)} active")

    cascade_on = [s.get("scan_id", "?") for s in active if s.get("cascade_enabled") is True]
    if cascade_on:
        errors.append(
            f"{len(cascade_on)} scan(s) have cascade_enabled=true (Scan B mixed into Scan A pull); "
            f"example scan_id={cascade_on[0]}"
        )

    missing_cascade_flag = sum(1 for s in active if "cascade_enabled" not in s)
    if missing_cascade_flag and active:
        errors.append(
            f"{missing_cascade_flag} scan(s) lack cascade_enabled field — redeploy Phase 2+ APK"
        )

    skipped_models = 0
    short_stages = 0
    missing_model_id = 0
    missing_cpu = 0
    for scan in active:
        cascade = scan.get("cascade") or {}
        skipped = cascade.get("models_skipped") or []
        if skipped:
            skipped_models += 1
        stages = [st for st in (scan.get("stages") or []) if st.get("status") == "ok"]
        if len(stages) < min_stages:
            short_stages += 1
        for st in stages:
            if not st.get("model_id"):
                missing_model_id += 1
            if st.get("cpu_ms") is None:
                missing_cpu += 1

    if skipped_models:
        errors.append(f"{skipped_models} scan(s) report cascade.models_skipped (ablation should run all)")
    if short_stages:
        errors.append(
            f"{short_stages} scan(s) have fewer than {min_stages} ok stages "
            f"(expected {min_stages} for full ablation)"
        )
    if missing_model_id:
        errors.append(f"{missing_model_id} ok stage(s) missing model_id")
    if missing_cpu:
        errors.append(f"{missing_cpu} ok stage(s) missing cpu_ms")

    expected_stages = set(expected_ablation_stage_ids())
    if active:
        seen: set[str] = set()
        for scan in active:
            for st in scan.get("stages") or []:
                if st.get("status") == "ok" and st.get("model_id"):
                    seen.add(st["model_id"])
        missing = sorted(expected_stages - seen)
        if missing:
            errors.append(f"never saw ok stages for model_id(s): {', '.join(missing)}")

    if require_battery:
        no_bat = sum(
            1
            for s in active
            if s.get("totals", {}).get("battery_pct_delta") is None
        )
        if no_bat == len(active) and active:
            errors.append(
                "totals.battery_pct_delta is null on all scans "
                "(device on charger or pre-Phase-2 APK?)"
            )

    # Ranked models need device samples for Phase 3 exit.
    table = build_plot_metrics_table_scan_a(active)
    for row in table["models"]:
        model_id = row["model_id"]
        n = row["device_scan_a"].get("n_stage_samples", 0)
        if n == 0:
            errors.append(f"no ok stage samples for plot model {model_id}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path, help="all_scan_metrics.jsonl or .json")
    parser.add_argument(
        "--min-scans",
        type=int,
        default=1,
        help="Minimum active (non-dedup) scans (thesis default: 400+)",
    )
    parser.add_argument(
        "--min-stages",
        type=int,
        default=11,
        help="Minimum ok stages per scan (11 = full ablation)",
    )
    parser.add_argument(
        "--require-battery",
        action="store_true",
        help="Fail if totals.battery_pct_delta is null everywhere",
    )
    parser.add_argument(
        "--write-table",
        type=Path,
        default=None,
        help="Write plot_metrics_table.json (device_scan_a section) to this path",
    )
    args = parser.parse_args(argv)

    path = args.metrics_path.resolve()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 2

    scans = load_scan_records(path)
    active = _active_scans(scans)
    print(f"Loaded {len(scans)} record(s), {len(active)} active scans from {path}")

    errors = validate_scan_a(
        scans,
        min_scans=args.min_scans,
        min_stages=args.min_stages,
        require_battery=args.require_battery,
    )

    table = build_plot_metrics_table_scan_a(active)
    if args.write_table:
        out = args.write_table.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    else:
        default = repo_root() / "Shared_pipeline_Files/results/figures/plot_metrics_table.json"
        if not errors:
            default.parent.mkdir(parents=True, exist_ok=True)
            default.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {default}")

    print(f"Plot models ({len(plot_order_model_ids())}):")
    for row in table["models"]:
        sa = row["device_scan_a"]
        print(
            f"  {row['model_id']}: n={sa.get('n_stage_samples')}, "
            f"stage_total_ms={sa.get('stage_total_ms')}, cpu_ms={sa.get('cpu_ms')}"
        )

    if errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nVALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
