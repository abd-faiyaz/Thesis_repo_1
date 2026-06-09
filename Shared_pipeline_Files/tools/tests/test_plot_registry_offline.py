#!/usr/bin/env python3
"""Phase 0/1 tests: registry validation and offline CSV regression."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from plot_registry_lib import (  # noqa: E402
    load_registry,
    normalize_offline_payload,
    repo_root,
    validate_registry,
)


class PlotRegistryOfflineTests(unittest.TestCase):
    def test_registry_validates(self) -> None:
        errors = validate_registry()
        self.assertEqual(errors, [], f"registry errors: {errors}")

    def test_legacy_xgb_metrics_roundtrip(self) -> None:
        root = repo_root()
        registry = load_registry(root)
        entry = next(m for m in registry["models"] if m["model_id"] == "manifest_xgb")
        src = root / entry["test_results_candidates"][0]
        payload = json.loads(src.read_text(encoding="utf-8"))
        norm = normalize_offline_payload(payload, entry, source_path=src)
        self.assertAlmostEqual(norm["metrics"]["accuracy"], 0.93717277486911, places=6)
        self.assertAlmostEqual(norm["metrics"]["f1"], 0.8883720930232558, places=6)
        self.assertAlmostEqual(norm["metrics"]["roc_auc"], 0.9913369197051206, places=6)
        self.assertEqual(norm["n_samples"], 764)

    def test_legacy_bytecnn_metrics_roundtrip(self) -> None:
        root = repo_root()
        registry = load_registry(root)
        entry = next(m for m in registry["models"] if m["model_id"] == "bytecnn")
        src = root / entry["test_results_candidates"][0]
        payload = json.loads(src.read_text(encoding="utf-8"))
        norm = normalize_offline_payload(payload, entry, source_path=src)
        self.assertAlmostEqual(norm["metrics"]["roc_auc"], 0.9592269376369795, places=6)

    def test_cascade_mode_a_normalization(self) -> None:
        root = repo_root()
        registry = load_registry(root)
        entry = next(m for m in registry["models"] if m["model_id"] == "mldp_dexheader_cascade")
        src = root / entry["test_results_candidates"][0]
        payload = json.loads(src.read_text(encoding="utf-8"))
        norm = normalize_offline_payload(payload, entry, source_path=src)
        self.assertIn("roc_auc", norm["metrics"])
        self.assertAlmostEqual(norm["metrics"]["f1"], payload["mode_a"]["f1"], places=9)

    def test_collect_and_csv_offline_only(self) -> None:
        root = repo_root()
        out_latest = root / "Shared_pipeline_Files/results/offline/latest"

        collect = subprocess.run(
            [sys.executable, str(_TOOLS / "collect_offline_test_metrics.py"), "--out-dir", str(out_latest)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(collect.returncode, 0, collect.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "sheet-offline.csv"
            build = subprocess.run(
                [
                    sys.executable,
                    str(_TOOLS / "build_extended_abstract_csv.py"),
                    "--offline-only",
                    "--latest-dir",
                    str(out_latest),
                    "--out",
                    str(out_csv),
                    "--plot-table",
                    str(root / "Shared_pipeline_Files/results/figures/__missing__.json"),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)

            with out_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            xgb = next(r for r in rows if r["Method"] == "XGBoost")
            self.assertEqual(xgb["ROC-AUC"], "0.9913")
            self.assertEqual(xgb["Accuracy"], "0.9372")
            self.assertEqual(xgb["F1"], "0.8884")
            self.assertEqual(xgb["CPU"], "")

            cnn = next(r for r in rows if "CNN" in r["Method"])
            self.assertEqual(cnn["ROC-AUC"], "0.9592")

            registry = load_registry(root)
            expected = len([m for m in registry["models"] if m.get("include_in_csv", True)])
            self.assertEqual(len(rows), expected)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
