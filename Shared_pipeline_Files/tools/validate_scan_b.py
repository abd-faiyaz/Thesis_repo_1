#!/usr/bin/env python3
"""Validate Scan B (cascade deployed) device metrics.

Usage:
  validate_scan_b.py path/to/all_scan_metrics.jsonl [--min-scans 400] [--scan-a path]
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
    apk_keys,
    effective_cascade_enabled,
    load_device_records,
    load_scan_records,
    merge_plot_metrics_scan_b,
    repo_root,
    sessions_by_id,
    summarize_scan_b,
)


def _active_scans(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in scans if not s.get("dedup_skipped")]


def validate_scan_b(
    scans: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    *,
    min_scans: int,
    scan_a_scans: list[dict[str, Any]] | None,
) -> list[str]:
    errors: list[str] = []
    session_index = sessions_by_id(sessions)
    active = _active_scans(scans)

    if len(active) < min_scans:
        errors.append(f"expected at least {min_scans} cascade scans, got {len(active)}")

    ablation = [
        s.get("scan_id", "?")
        for s in active
        if effective_cascade_enabled(s, session_index) is False
    ]
    if ablation:
        errors.append(
            f"{len(ablation)} scan(s) are ablation (cascade_enabled=false); "
            f"example scan_id={ablation[0]}"
        )

    missing_flag = sum(
        1 for s in active if effective_cascade_enabled(s, session_index) is None
    )
    if missing_flag:
        errors.append(
            f"{missing_flag} scan(s) lack cascade_enabled on scan and session — redeploy Phase 2+ APK"
        )

    no_cascade_block = 0
    no_exit_tier = 0
    no_skipped = 0
    for scan in active:
        cascade = scan.get("cascade")
        if not cascade:
            no_cascade_block += 1
            continue
        if cascade.get("exit_tier") is None:
            no_exit_tier += 1
        skipped = cascade.get("models_skipped")
        if not isinstance(skipped, list) or len(skipped) == 0:
            no_skipped += 1

    if no_cascade_block:
        errors.append(f"{no_cascade_block} scan(s) missing cascade block")
    if no_exit_tier:
        errors.append(f"{no_exit_tier} scan(s) missing cascade.exit_tier")

    # Early exit is expected in cascade; only warn-level if none skipped (all tier 4).
    if no_skipped == len(active) and active:
        errors.append(
            "no scan reports models_skipped — cascade may not be exiting early "
            "(check cascade_policy.json thresholds)"
        )

    if scan_a_scans is not None:
        a_keys: set[str] = set()
        for scan in _active_scans(scan_a_scans):
            a_keys.update(apk_keys(scan))
        overlap = 0
        for scan in active:
            if apk_keys(scan) & a_keys:
                overlap += 1
        if overlap == 0 and a_keys:
            errors.append(
                "no APK overlap with Scan A pull — Scan B should rescan the same eval manifest"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path, help="all_scan_metrics.jsonl or .json")
    parser.add_argument(
        "--min-scans",
        type=int,
        default=1,
        help="Minimum active scans (thesis default: 400+)",
    )
    parser.add_argument(
        "--scan-a",
        type=Path,
        default=None,
        help="Scan A metrics path to check APK overlap (optional)",
    )
    parser.add_argument(
        "--write-table",
        type=Path,
        default=None,
        help="Merge device_scan_b into plot_metrics_table.json",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Write cascade device comparison report JSON",
    )
    args = parser.parse_args(argv)

    path = args.metrics_path.resolve()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 2

    scans, sessions = load_device_records(path)
    session_index = sessions_by_id(sessions)
    cascade_scans = [
        s
        for s in _active_scans(scans)
        if effective_cascade_enabled(s, session_index) is True
    ]
    print(
        f"Loaded {len(scans)} record(s), {len(cascade_scans)} cascade scans "
        f"({len(sessions)} session(s)) from {path}"
    )

    scan_a_scans = None
    if args.scan_a is not None and args.scan_a.is_file():
        scan_a_scans = load_scan_records(args.scan_a)

    errors = validate_scan_b(
        cascade_scans,
        sessions,
        min_scans=args.min_scans,
        scan_a_scans=scan_a_scans,
    )

    summary = summarize_scan_b(cascade_scans)
    print(f"median_wall_ms: {summary.get('median_wall_ms')}")
    print(f"exit_tier_histogram: {summary.get('exit_tier_histogram')}")
    print(f"early_exit_rate: {summary.get('early_exit_rate', 0):.2%}")

    table_path = args.write_table or (
        repo_root() / "Shared_pipeline_Files/results/figures/plot_metrics_table.json"
    )
    if not errors:
        merge_plot_metrics_scan_b(table_path, summary)
        print(f"Merged device_scan_b → {table_path}")

    report_path = args.write_report or (
        repo_root() / "Shared_pipeline_Files/results/device/scan_b_cascade/cascade_device_report.json"
    )
    if not errors:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "version": 1,
            "source": str(path),
            "n_sessions": len(sessions),
            **summary,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {report_path}")

    if errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nVALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
