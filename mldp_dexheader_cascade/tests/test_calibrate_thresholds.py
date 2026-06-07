"""Unit tests for cascade threshold calibration (M11)."""

from __future__ import annotations

import numpy as np

from src.training.calibrate_thresholds import (
    calibrate_cascade_thresholds,
    false_alarm_rate,
    false_omission_rate,
)


def test_for_and_fa_rates() -> None:
    y = np.array([0, 0, 1, 1, 1], dtype=int)
    s1 = np.array([0.1, 0.9, 0.05, 0.5, 0.95], dtype=float)
    assert false_omission_rate(y, s1, 0.1) == 1 / 3
    assert false_alarm_rate(y, s1, 0.85) == 0.5


def test_calibrate_respects_targets() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=500)
    s1 = rng.random(500)
    out = calibrate_cascade_thresholds(
        y, s1, target_false_omission_rate=0.02, target_false_alarm_at_thigh=0.02
    )
    assert out["stage1_t_low"] < out["stage1_t_high"]
    assert out["val_false_omission_rate_at_t_low"] <= 0.02 + 1e-9
    assert out["val_false_alarm_rate_at_t_high"] <= 0.02 + 1e-9
