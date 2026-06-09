"""P6 — calibrate Mode A threshold and Mode B cascade t_low/t_high on val only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared_calibration import (
    calibrate_cascade_thresholds,
    calibrate_t_high,
    calibrate_t_low,
    false_alarm_rate,
    false_omission_rate,
    tune_threshold,
    write_thresholds as _write_thresholds,
)

from src.config import PipelineConfig

__all__ = [
    "tune_threshold",
    "false_omission_rate",
    "false_alarm_rate",
    "calibrate_t_low",
    "calibrate_t_high",
    "calibrate_cascade_thresholds",
    "build_thresholds_payload",
    "write_thresholds",
]


def build_thresholds_payload(
    cfg: PipelineConfig,
    *,
    mode_a_default: float,
    mode_a_tuned: float,
    cascade: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_id": cfg.model_id,
        "mode_a": {
            "default": mode_a_default,
            "tuned_val": mode_a_tuned,
        },
        "mode_b": {
            "stage1_t_low": cascade["stage1_t_low"],
            "stage1_t_high": cascade["stage1_t_high"],
            "val_false_omission_rate_at_t_low": cascade["val_false_omission_rate_at_t_low"],
            "val_false_alarm_rate_at_t_high": cascade["val_false_alarm_rate_at_t_high"],
            "val_step1_exit_rate": cascade["val_step1_exit_rate"],
        },
        "cascade_targets": {
            "target_false_omission_rate": float(
                cfg.cascade.get("target_false_omission_rate", 0.02)
            ),
            "target_false_alarm_at_thigh": float(
                cfg.cascade.get("target_false_alarm_at_thigh", 0.02)
            ),
        },
    }


def write_thresholds(path: Path, payload: dict[str, Any]) -> None:
    _write_thresholds(path, payload)
