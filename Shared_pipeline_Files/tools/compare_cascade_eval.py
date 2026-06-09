#!/usr/bin/env python3
"""Compare Scan A (ablation) vs Scan B (cascade) device eval runs.

Joins scan.session_id → session.cascade_enabled (authoritative when scan field missing).
Prints tier histogram, median wall_ms, and optional accuracy comparison on labeled APKs.

Usage:
  compare_cascade_eval.py scan_b.jsonl
  compare_cascade_eval.py scan_b.jsonl --scan-a scan_a.jsonl --write-report out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from device_metrics_lib import (  # noqa: E402
    apk_keys,
    effective_cascade_enabled,
    load_device_records,
    sessions_by_id,
    summarize_scan_b,
)


def _active(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in scans if not s.get("dedup_skipped")]


def ground_truth(scan: dict[str, Any]) -> str | None:
    label = scan.get("ground_truth")
    if label in {"benign", "malware"}:
        return label
    name = (scan.get("apk") or {}).get("name", "")
    lower = name.lower()
    if lower.endswith("_benign.apk"):
        return "benign"
    if lower.endswith("_malware.apk"):
        return "malware"
    return None


def prediction(scan: dict[str, Any]) -> str | None:
    cascade = scan.get("cascade") or {}
    decision = cascade.get("decision")
    if decision in {"benign", "malware", "uncertain"}:
        return decision
    ensemble = scan.get("ensemble") or {}
    decision = ensemble.get("decision")
    if decision in {"benign", "malware", "uncertain"}:
        return decision
    return None


def confusion_metrics(scans: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for scan in scans:
        truth = ground_truth(scan)
        pred = prediction(scan)
        if truth is None or pred is None or pred == "uncertain":
            continue
        counts[f"{truth}_{pred}"] += 1
    tp = counts.get("malware_malware", 0)
    tn = counts.get("benign_benign", 0)
    fp = counts.get("benign_malware", 0)
    fn = counts.get("malware_benign", 0)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_labeled": float(total),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def median_wall_ms(scans: list[dict[str, Any]]) -> float | None:
    walls = [
        float((s.get("totals") or {})["wall_ms"])
        for s in scans
        if (s.get("totals") or {}).get("wall_ms") is not None
    ]
    if not walls:
        return None
    return float(statistics.median(walls))


def partition_by_mode(
    scans: list[dict[str, Any]], sessions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_index = sessions_by_id(sessions)
    ablation: list[dict[str, Any]] = []
    cascade: list[dict[str, Any]] = []
    for scan in _active(scans):
        mode = effective_cascade_enabled(scan, session_index)
        if mode is True:
            cascade.append(scan)
        elif mode is False:
            ablation.append(scan)
    return ablation, cascade


def paired_wall_comparison(
    ablation: list[dict[str, Any]], cascade: list[dict[str, Any]]
) -> dict[str, Any]:
    cascade_by_key: dict[str, dict[str, Any]] = {}
    for scan in cascade:
        for key in apk_keys(scan):
            cascade_by_key[key] = scan

    paired_a: list[float] = []
    paired_b: list[float] = []
    for scan in ablation:
        keys = apk_keys(scan)
        match = None
        for key in keys:
            if key in cascade_by_key:
                match = cascade_by_key[key]
                break
        if match is None:
            continue
        wall_a = (scan.get("totals") or {}).get("wall_ms")
        wall_b = (match.get("totals") or {}).get("wall_ms")
        if wall_a is None or wall_b is None:
            continue
        paired_a.append(float(wall_a))
        paired_b.append(float(wall_b))

    if not paired_a:
        return {"n_paired": 0}

    ratios = [b / a if a > 0 else 0.0 for a, b in zip(paired_a, paired_b)]
    return {
        "n_paired": len(paired_a),
        "median_wall_ms_ablation": float(statistics.median(paired_a)),
        "median_wall_ms_cascade": float(statistics.median(paired_b)),
        "median_speedup_ratio": float(statistics.median(ratios)),
        "mean_speedup_ratio": float(statistics.mean(ratios)),
    }


def build_report(
    *,
    path_b: Path,
    scans_b: list[dict[str, Any]],
    sessions_b: list[dict[str, Any]],
    scans_a: list[dict[str, Any]] | None,
    sessions_a: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    session_index_b = sessions_by_id(sessions_b)
    cascade_scans = [
        s
        for s in _active(scans_b)
        if effective_cascade_enabled(s, session_index_b) is True
    ]
    scan_b_summary = summarize_scan_b(cascade_scans)

    report: dict[str, Any] = {
        "version": 1,
        "scan_b_path": str(path_b),
        "cascade": {
            **scan_b_summary,
            "labeled_metrics": confusion_metrics(cascade_scans),
        },
    }

    if scans_a is not None:
        sessions_a = sessions_a or []
        ablation_from_b, cascade_from_b = partition_by_mode(scans_b, sessions_b)
        ablation = _active(scans_a)
        if not ablation:
            ablation = ablation_from_b
        report["scan_a"] = {
            "n_scans": len(ablation),
            "median_wall_ms": median_wall_ms(ablation),
            "labeled_metrics": confusion_metrics(ablation),
        }
        report["paired_apk_comparison"] = paired_wall_comparison(ablation, cascade_scans)
        if report["scan_a"]["median_wall_ms"] and scan_b_summary.get("median_wall_ms"):
            report["wall_ms_reduction_pct"] = (
                1.0
                - float(scan_b_summary["median_wall_ms"])
                / float(report["scan_a"]["median_wall_ms"])
            ) * 100.0

    return report


def print_report(report: dict[str, Any]) -> None:
    cascade = report.get("cascade", {})
    print(f"\n=== cascade (n={cascade.get('n_scans', 0)}) ===")
    print(f"  median_wall_ms:     {cascade.get('median_wall_ms')}")
    print(f"  early_exit_rate:    {cascade.get('early_exit_rate', 0):.2%}")
    print(f"  exit_tier_histogram: {cascade.get('exit_tier_histogram')}")
    print(f"  exit_reason_counts:  {cascade.get('exit_reason_counts')}")
    labeled = cascade.get("labeled_metrics") or {}
    if labeled.get("n_labeled", 0) > 0:
        print(f"  labeled n:          {int(labeled['n_labeled'])}")
        print(f"  accuracy:           {labeled.get('accuracy', 0):.4f}")
        print(f"  f1:                 {labeled.get('f1', 0):.4f}")

    scan_a = report.get("scan_a")
    if scan_a:
        print(f"\n=== ablation / Scan A (n={scan_a.get('n_scans', 0)}) ===")
        print(f"  median_wall_ms:     {scan_a.get('median_wall_ms')}")
        labeled_a = scan_a.get("labeled_metrics") or {}
        if labeled_a.get("n_labeled", 0) > 0:
            print(f"  f1:                 {labeled_a.get('f1', 0):.4f}")

    paired = report.get("paired_apk_comparison") or {}
    if paired.get("n_paired", 0) > 0:
        print(f"\n=== paired APK wall_ms (n={paired['n_paired']}) ===")
        print(f"  median ablation:    {paired.get('median_wall_ms_ablation'):.1f} ms")
        print(f"  median cascade:     {paired.get('median_wall_ms_cascade'):.1f} ms")
        print(f"  median speedup:     {paired.get('median_speedup_ratio', 0):.2%} of ablation time")
    if "wall_ms_reduction_pct" in report:
        print(f"  wall_ms reduction:  {report['wall_ms_reduction_pct']:.1f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_b",
        type=Path,
        nargs="?",
        help="Scan B metrics JSONL/JSON (cascade)",
    )
    parser.add_argument(
        "--scan-a",
        type=Path,
        default=None,
        help="Scan A metrics for paired comparison",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Write full comparison report JSON",
    )
    args = parser.parse_args(argv)

    default_b = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "device"
        / "scan_b_cascade"
        / "scan_b_cascade.jsonl"
    )
    path_b = args.scan_b or default_b
    if not path_b.is_file():
        print(f"Metrics not found: {path_b}", file=sys.stderr)
        return 1

    scans_b, sessions_b = load_device_records(path_b)
    scans_a = sessions_a = None
    if args.scan_a is not None:
        if not args.scan_a.is_file():
            print(f"Scan A not found: {args.scan_a}", file=sys.stderr)
            return 1
        scans_a, sessions_a = load_device_records(args.scan_a)

    report = build_report(
        path_b=path_b,
        scans_b=scans_b,
        sessions_b=sessions_b,
        scans_a=scans_a,
        sessions_a=sessions_a,
    )
    print(f"Loaded {len(scans_b)} record(s) from {path_b}")
    print_report(report)

    out = args.write_report
    if out is None and not args.scan_b:
        out = path_b.parent / "cascade_device_report.json"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
