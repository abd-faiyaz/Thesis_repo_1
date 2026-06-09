"""Tests for export_manifest validation metric stamping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shared_calibration.manifest_metrics import stamp_export_manifest


class ManifestMetricsTest(unittest.TestCase):
    def test_stamp_writes_val_f1_and_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_scores = root / "val_scores.json"
            manifest = root / "export_manifest.json"
            val_scores.write_text(
                json.dumps(
                    {
                        "metrics": {"f1": 0.91, "accuracy": 0.95},
                        "rows": [],
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(json.dumps({"model_id": "test"}), encoding="utf-8")
            result = stamp_export_manifest(manifest, val_scores)
            self.assertTrue(result["changed"])
            updated = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertAlmostEqual(updated["val_f1"], 0.91)
            self.assertAlmostEqual(updated["val_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
