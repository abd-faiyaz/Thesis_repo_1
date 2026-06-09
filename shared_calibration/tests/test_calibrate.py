"""Smoke tests for shared calibration math."""

from __future__ import annotations

import unittest

import numpy as np

from shared_calibration import (
    build_thresholds_payload,
    build_val_thresholds_payload,
    calibrate_cascade_thresholds,
    tune_threshold,
)


class SharedCalibrationTest(unittest.TestCase):
    def test_tune_threshold_prefers_separating_cutoff(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        t = tune_threshold(y, scores)
        self.assertGreaterEqual(t, 0.3)
        self.assertLessEqual(t, 0.7)

    def test_cascade_band_is_ordered(self) -> None:
        rng = np.random.default_rng(42)
        y = np.concatenate([np.zeros(200), np.ones(200)]).astype(int)
        scores = np.concatenate(
            [rng.uniform(0.0, 0.4, 200), rng.uniform(0.6, 1.0, 200)]
        )
        bands = calibrate_cascade_thresholds(y, scores)
        self.assertLess(bands["stage1_t_low"], bands["stage1_t_high"])
        self.assertGreaterEqual(bands["val_step1_exit_rate"], 0.0)
        self.assertLessEqual(bands["val_step1_exit_rate"], 1.0)

    def test_build_val_thresholds_payload_includes_cascade(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.85])
        payload = build_val_thresholds_payload(
            model_id="test_model",
            y_true=y,
            scores=scores,
        )
        self.assertIn("cascade", payload)
        self.assertLess(payload["cascade"]["t_low"], payload["cascade"]["t_high"])

    def test_build_thresholds_payload_shape(self) -> None:
        bands = calibrate_cascade_thresholds(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
        )
        payload = build_thresholds_payload(
            model_id="test_model",
            tuned_val=0.55,
            cascade=bands,
        )
        self.assertEqual(payload["model_id"], "test_model")
        self.assertEqual(payload["malware_threshold"], payload["tuned_val"])
        self.assertIn("cascade", payload)
        self.assertIn("t_low", payload["cascade"])
        self.assertIn("t_high", payload["cascade"])


if __name__ == "__main__":
    unittest.main()
