#!/usr/bin/env python3
"""Phase 6/8 tests: thesis plot scripts and sufficiency report."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

try:
    from generate_plotting_sufficiency_report import build_report  # noqa: E402
    from thesis_plot_lib import ALL_PLOTS, _ensure_matplotlib, load_table, run_all_plots  # noqa: E402

    _HAS_MPL = True
except SystemExit:
    _HAS_MPL = False

_FIXTURE_TABLE = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "figures"
    / "plot_metrics_table_fixture.json"
)


@unittest.skipUnless(_HAS_MPL, "matplotlib not installed")
class ThesisPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_matplotlib()

    def test_all_plots_on_fixture(self) -> None:
        if not _FIXTURE_TABLE.is_file():
            self.skipTest("fixture table missing — run aggregate on test fixtures first")
        table = load_table(_FIXTURE_TABLE)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = run_all_plots(table, out)
            names = {p.name for p in written}
            self.assertIn("apk_size_vs_detection_time.png", names)
            self.assertIn("plot3_accuracy_vs_ram.png", names)
            self.assertIn("performance_res_usage_tradeoff_plot.jpeg", names)
            self.assertIn("figure_index.json", names)
            self.assertIn("inferenceTime_vs_apkSize.png", names)
            self.assertEqual(len(ALL_PLOTS), 8)

    def test_csv_with_fixture_table(self) -> None:
        if not _FIXTURE_TABLE.is_file():
            self.skipTest("fixture table missing")
        root = _TOOLS.parents[1]
        latest = root / "Shared_pipeline_Files/results/offline/latest"
        if not latest.is_dir():
            self.skipTest("offline/latest missing")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sheet.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_TOOLS / "build_extended_abstract_csv.py"),
                    "--plot-table",
                    str(_FIXTURE_TABLE),
                    "--latest-dir",
                    str(latest),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
            xgb = next(r for r in rows if r["Method"] == "XGBoost")
            self.assertEqual(xgb["ROC-AUC"], "0.9913")
            self.assertTrue(xgb["CPU"].endswith("ms"))
            self.assertTrue(xgb["Device Feasibility"])
            self.assertTrue(xgb["Comments"])

    def test_sufficiency_report_fixture(self) -> None:
        if not _FIXTURE_TABLE.is_file():
            self.skipTest("fixture table missing")
        with tempfile.TemporaryDirectory() as tmp:
            fig_dir = Path(tmp) / "figs"
            table = load_table(_FIXTURE_TABLE)
            run_all_plots(table, fig_dir)
            report, blockers = build_report(
                table_path=_FIXTURE_TABLE,
                figures_dir=fig_dir,
                csv_path=None,
            )
            self.assertIn("apk_size_vs_detection_time.png", report)
            self.assertIn("| Yes |", report)
            self.assertEqual(blockers, [])


if __name__ == "__main__":
    unittest.main()
