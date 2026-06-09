#!/usr/bin/env python3
"""Tests for Scan A validation and device summary helpers."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from device_metrics_lib import (  # noqa: E402
    build_plot_metrics_table_scan_a,
    load_scan_records,
    stage_total_ms,
)
from validate_scan_a import validate_scan_a  # noqa: E402


_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scan_a_valid.jsonl"


class ValidateScanATests(unittest.TestCase):
    def test_stage_total_ms(self) -> None:
        stage = {"parse_ms": 1.0, "vectorize_ms": 2.0, "inference_ms": 3.0}
        self.assertAlmostEqual(stage_total_ms(stage), 6.0)

    def test_valid_fixture_passes(self) -> None:
        scans = load_scan_records(_FIXTURE)
        errors = validate_scan_a(scans, min_scans=2, min_stages=11, require_battery=False)
        self.assertEqual(errors, [], errors)

    def test_cascade_enabled_fails(self) -> None:
        scans = load_scan_records(_FIXTURE)
        scans[0]["cascade_enabled"] = True
        errors = validate_scan_a(scans, min_scans=1, min_stages=11, require_battery=False)
        self.assertTrue(any("cascade_enabled=true" in e for e in errors))

    def test_plot_table_has_all_models(self) -> None:
        scans = load_scan_records(_FIXTURE)
        table = build_plot_metrics_table_scan_a(scans)
        self.assertEqual(table["n_scans"], 2)
        self.assertEqual(len(table["models"]), 10)
        for row in table["models"]:
            self.assertGreater(row["device_scan_a"]["n_stage_samples"], 0)
            self.assertIsNotNone(row["device_scan_a"]["stage_total_ms"])

    def test_cli_exit_zero_on_fixture(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(_TOOLS / "validate_scan_a.py"),
                str(_FIXTURE),
                "--min-scans",
                "2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
