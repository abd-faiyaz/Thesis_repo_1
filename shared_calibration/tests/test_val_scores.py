"""Tests for val score dump helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shared_calibration.val_scores import (
    build_split_scores_payload,
    filter_rows_to_canonical,
    load_canonical_val_ids,
    write_split_scores_bundle,
)


class ValScoresTest(unittest.TestCase):
    def test_canonical_filter(self) -> None:
        rows = [
            {"apk_id": "aaa", "label": 0, "score": 0.1},
            {"apk_id": "bbb", "label": 1, "score": 0.9},
        ]
        filtered = filter_rows_to_canonical(rows, {"aaa"})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["apk_id"], "aaa")

    def test_build_split_scores_payload(self) -> None:
        payload = build_split_scores_payload(
            model_id="test_model",
            split="val",
            apk_ids=["a" * 64, "b" * 64],
            labels=np.array([0, 1]),
            scores=np.array([0.2, 0.8]),
            threshold=0.5,
        )
        self.assertEqual(payload["model_id"], "test_model")
        self.assertEqual(payload["n_samples"], 2)
        self.assertIn("f1", payload["metrics"])

    def test_write_split_scores_bundle_syncs_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "Shared_pipeline_Files"
            (shared / "data" / "splits").mkdir(parents=True)
            (shared / "calibration").mkdir(parents=True)
            canonical = shared / "data" / "splits" / "canonical_val.txt"
            apk = "c" * 64
            canonical.write_text(f"{apk}\n", encoding="utf-8")

            metrics_dir = root / "pipeline" / "artifacts" / "metrics"
            out = write_split_scores_bundle(
                model_id="test_model",
                split="val",
                metrics_dir=metrics_dir,
                apk_ids=[apk],
                labels=np.array([1]),
                scores=np.array([0.7]),
                threshold=0.5,
                repo_root=root,
            )
            self.assertTrue(out.is_file())
            synced = shared / "calibration" / "test_model_val_scores.json"
            self.assertTrue(synced.is_file())
            payload = json.loads(synced.read_text(encoding="utf-8"))
            self.assertEqual(payload["n_aligned"], 1)

    def test_load_canonical_skips_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "canonical_val.txt"
            path.write_text("# comment\nabc\n\n", encoding="utf-8")
            ids = load_canonical_val_ids(path)
            self.assertEqual(ids, {"abc"})


if __name__ == "__main__":
    unittest.main()
