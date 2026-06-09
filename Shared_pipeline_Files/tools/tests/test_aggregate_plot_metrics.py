#!/usr/bin/env python3
"""Tests for aggregate_plot_metrics (Phase 5)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from aggregate_plot_metrics import (  # noqa: E402
    build_aggregate_table,
    load_offline_latest,
    missing_required_fields,
)
from device_metrics_lib import (  # noqa: E402
    SCAN_A_JSONL,
    SCAN_B_JSONL,
    SIZE_BUCKET_EDGES_MB,
    apk_size_mb,
    compute_cost_scores,
    compute_session_battery_per_model,
    jsonl_filename_for_mode,
    load_device_records,
    resolve_device_metrics_path,
    size_bucket_label,
)
from plot_registry_lib import load_registry, repo_root  # noqa: E402

_FIXTURE_A = Path(__file__).resolve().parent / "fixtures" / "scan_a_valid.jsonl"
_FIXTURE_B = Path(__file__).resolve().parent / "fixtures" / "scan_b_valid.jsonl"


class AggregatePlotMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_root()
        cls.registry = load_registry(cls.root)

    def test_split_jsonl_filenames(self) -> None:
        self.assertEqual(jsonl_filename_for_mode(False), SCAN_A_JSONL)
        self.assertEqual(jsonl_filename_for_mode(True), SCAN_B_JSONL)

    def test_resolve_device_metrics_path_by_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan_a = root / "scan_a_all_models"
            scan_b = root / "scan_b_cascade"
            scan_a.mkdir()
            scan_b.mkdir()
            (scan_a / SCAN_A_JSONL).write_text('{"record_type":"scan","stages":[]}\n', encoding="utf-8")
            (scan_b / SCAN_B_JSONL).write_text('{"record_type":"scan","stages":[]}\n', encoding="utf-8")
            self.assertEqual(resolve_device_metrics_path(scan_a).name, SCAN_A_JSONL)
            self.assertEqual(resolve_device_metrics_path(scan_b).name, SCAN_B_JSONL)

    def test_size_bucket_labels(self) -> None:
        self.assertEqual(size_bucket_label(512 * 1024), "1")
        self.assertEqual(size_bucket_label(3 * 1024 * 1024), "5")
        self.assertEqual(size_bucket_label(150 * 1024 * 1024), "100+")

    def test_cost_scores_in_range(self) -> None:
        rows = [
            {
                "model_id": "a",
                "device_scan_a": {"stage_total_ms": 10.0, "cpu_ms": 5.0, "mem_mb": 2.0},
            },
            {
                "model_id": "b",
                "device_scan_a": {"stage_total_ms": 100.0, "cpu_ms": 50.0, "mem_mb": 20.0},
            },
            {
                "model_id": "c",
                "device_scan_a": {"stage_total_ms": 50.0, "cpu_ms": 25.0, "mem_mb": 10.0},
            },
        ]
        scores = compute_cost_scores(rows)
        self.assertEqual(len(scores), 3)
        for val in scores.values():
            self.assertGreaterEqual(val, 0.1)
            self.assertLessEqual(val, 100.0)

    def test_build_aggregate_with_fixtures(self) -> None:
        latest = self.root / "Shared_pipeline_Files/results/offline/latest"
        if not latest.is_dir():
            self.skipTest("offline/latest not populated — run collect_offline_test_metrics.py")

        offline = load_offline_latest(latest, self.registry)
        if len(offline) < 2:
            self.skipTest("need offline latest JSON for aggregate test")

        scans_a, sessions_a = load_device_records(_FIXTURE_A)
        scans_b, sessions_b = load_device_records(_FIXTURE_B)
        table = build_aggregate_table(
            registry=self.registry,
            offline_by_model=offline,
            scan_a_scans=scans_a,
            scan_a_sessions=sessions_a,
            scan_b_scans=scans_b,
            scan_b_sessions=sessions_b,
            root=self.root,
        )

        self.assertEqual(table["n_scans_scan_a"], 2)
        self.assertEqual(table["n_scans_scan_b"], 2)
        self.assertIn("device_scan_b", table)
        self.assertEqual(len(table["per_apk_series"]), 2)
        self.assertEqual(table["size_bucket_edges_mb"], SIZE_BUCKET_EDGES_MB)
        self.assertIn("cost_score", table)

        for row in table["models"]:
            self.assertIn("offline", row)
            self.assertIn("derived", row)
            if row["device_scan_a"].get("n_stage_samples", 0) > 0:
                self.assertIn("f1_pct", row["derived"])
                self.assertIn("cost_score", row["derived"])

    def test_session_battery_allocation_from_charge_counter(self) -> None:
        sessions = [
            {
                "record_type": "session",
                "session_id": "sess-1",
                "battery": {
                    "capacity_pct_delta": 0,
                    "charge_counter_uah_used": 1000,
                    "charge_counter_uah_start": 100000,
                    "capacity_pct_start": 100,
                },
            }
        ]
        scans = [
            {
                "session_id": "sess-1",
                "cascade_enabled": False,
                "stages": [
                    {
                        "model_id": "bytecnn",
                        "status": "ok",
                        "parse_ms": 1.0,
                        "vectorize_ms": 0.0,
                        "inference_ms": 1.0,
                    },
                    {
                        "model_id": "manifest_xgb",
                        "status": "ok",
                        "parse_ms": 3.0,
                        "vectorize_ms": 0.0,
                        "inference_ms": 1.0,
                    },
                ],
            }
        ]
        shares = compute_session_battery_per_model(scans, sessions, root=self.root)
        self.assertAlmostEqual(shares["bytecnn"], 1.0 * (2.0 / 6.0), places=4)
        self.assertAlmostEqual(shares["manifest_xgb"], 1.0 * (4.0 / 6.0), places=4)

    def test_missing_required_fields_detects_gaps(self) -> None:
        table = {
            "models": [
                {
                    "model_id": "x",
                    "offline": {},
                    "device_scan_a": {"n_stage_samples": 0},
                    "derived": {},
                }
            ]
        }
        errors = missing_required_fields(table)
        self.assertTrue(any("offline.accuracy" in e for e in errors))
        self.assertTrue(any("no Scan A stage samples" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
