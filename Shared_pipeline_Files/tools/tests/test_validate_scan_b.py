#!/usr/bin/env python3
"""Tests for Scan B validation and cascade comparison."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compare_cascade_eval import build_report  # noqa: E402
from device_metrics_lib import (  # noqa: E402
    filter_jsonl_by_mode,
    load_device_records,
    summarize_scan_b,
)
from validate_scan_b import validate_scan_b  # noqa: E402

_FIXTURE_B = Path(__file__).resolve().parent / "fixtures" / "scan_b_valid.jsonl"
_FIXTURE_A = Path(__file__).resolve().parent / "fixtures" / "scan_a_valid.jsonl"


class ValidateScanBTests(unittest.TestCase):
    def test_valid_fixture_passes(self) -> None:
        scans, sessions = load_device_records(_FIXTURE_B)
        cascade = [s for s in scans if s.get("cascade_enabled")]
        errors = validate_scan_b(
            cascade,
            sessions,
            min_scans=2,
            scan_a_scans=None,
        )
        self.assertEqual(errors, [], errors)

    def test_summarize_exit_tiers(self) -> None:
        scans, _ = load_device_records(_FIXTURE_B)
        summary = summarize_scan_b(scans)
        self.assertEqual(summary["n_scans"], 2)
        self.assertEqual(summary["exit_tier_histogram"]["1"], 1)
        self.assertEqual(summary["exit_tier_histogram"]["4"], 1)
        self.assertEqual(summary["median_wall_ms"], 285.0)

    def test_filter_cascade_mode(self) -> None:
        scans, sessions = filter_jsonl_by_mode(_FIXTURE_A, cascade_enabled=False)
        self.assertEqual(len(scans), 2)
        self.assertTrue(all(not s.get("cascade_enabled") for s in scans))

    def test_paired_wall_with_scan_a(self) -> None:
        scans_a, _ = load_device_records(_FIXTURE_A)
        scans_b, sessions_b = load_device_records(_FIXTURE_B)
        report = build_report(
            path_b=_FIXTURE_B,
            scans_b=scans_b,
            sessions_b=sessions_b,
            scans_a=scans_a,
            sessions_a=[],
        )
        self.assertIn("paired_apk_comparison", report)
        self.assertGreater(report["cascade"]["n_scans"], 0)

    def test_cli_validate_scan_b(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_TOOLS / "validate_scan_b.py"),
                str(_FIXTURE_B),
                "--min-scans",
                "2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_cli_compare_cascade(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_TOOLS / "compare_cascade_eval.py"),
                str(_FIXTURE_B),
                "--scan-a",
                str(_FIXTURE_A),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
