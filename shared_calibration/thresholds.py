"""Canonical thresholds.json payload builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from shared_calibration.calibrate import calibrate_cascade_thresholds
from shared_calibration.metrics import tune_threshold


def cascade_band_from_calibration(calibration: dict[str, float]) -> dict[str, float]:
    """Map calibrate_cascade_thresholds keys to canonical cascade band keys."""
    return {
        "t_low": float(calibration["stage1_t_low"]),
        "t_high": float(calibration["stage1_t_high"]),
        "val_false_omission_rate_at_t_low": float(
            calibration["val_false_omission_rate_at_t_low"]
        ),
        "val_false_alarm_rate_at_t_high": float(
            calibration["val_false_alarm_rate_at_t_high"]
        ),
        "val_step1_exit_rate": float(calibration["val_step1_exit_rate"]),
    }


def build_thresholds_payload(
    *,
    model_id: str,
    default: float = 0.5,
    tuned_val: float,
    cascade: dict[str, float] | None = None,
    cascade_targets: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical thresholds.json for a single-model bundle."""
    targets = cascade_targets or {
        "target_false_omission_rate": 0.02,
        "target_false_alarm_at_thigh": 0.02,
    }
    payload: dict[str, Any] = {
        "model_id": model_id,
        "default": float(default),
        "tuned_val": float(tuned_val),
        "malware_threshold": float(tuned_val),
        "cascade_targets": {
            "target_false_omission_rate": float(
                targets.get("target_false_omission_rate", 0.02)
            ),
            "target_false_alarm_at_thigh": float(
                targets.get("target_false_alarm_at_thigh", 0.02)
            ),
        },
    }
    if cascade is not None:
        if "t_low" in cascade:
            band = {k: float(v) for k, v in cascade.items()}
        else:
            band = cascade_band_from_calibration(cascade)
        payload["cascade"] = band
    if extra:
        payload.update(extra)
    return payload


def write_thresholds(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_cascade_targets(cascade_cfg: dict[str, float] | None) -> dict[str, float]:
    cfg = cascade_cfg or {}
    return {
        "target_false_omission_rate": float(cfg.get("target_false_omission_rate", 0.02)),
        "target_false_alarm_at_thigh": float(cfg.get("target_false_alarm_at_thigh", 0.02)),
    }


def build_val_thresholds_payload(
    *,
    model_id: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    default: float = 0.5,
    tune: bool = True,
    calibrate_bands: bool = True,
    cascade_targets: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune decision cutoff and cascade bands on validation scores."""
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = get_cascade_targets(cascade_targets)
    tuned_val = tune_threshold(y_true, scores) if tune else float(default)
    cascade = None
    if calibrate_bands:
        cascade = calibrate_cascade_thresholds(
            y_true,
            scores,
            target_false_omission_rate=targets["target_false_omission_rate"],
            target_false_alarm_at_thigh=targets["target_false_alarm_at_thigh"],
        )
    return build_thresholds_payload(
        model_id=model_id,
        default=float(default),
        tuned_val=tuned_val,
        cascade=cascade,
        cascade_targets=targets,
        extra=extra,
    )


def read_thresholds_payload(
    path: Path,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(fallback or {})


def write_export_thresholds(
    metrics_path: Path,
    out_path: Path,
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Copy full thresholds.json into an export bundle (preserves cascade block)."""
    payload = read_thresholds_payload(metrics_path, fallback=fallback)
    write_thresholds(out_path, payload)
    return payload


def format_cascade_band_summary(payload: dict[str, Any]) -> str:
    cascade = payload.get("cascade")
    if not cascade:
        return ""
    return (
        f"t_low={cascade['t_low']:.4f} t_high={cascade['t_high']:.4f} "
        f"val_step1_exit_rate={cascade['val_step1_exit_rate']:.3f}"
    )


def read_saved_thresholds(
    path: Path,
    *,
    default: float = 0.5,
) -> dict[str, float]:
    """Load default + tuned_val from artifacts/metrics/thresholds.json if present."""
    path = Path(path)
    if not path.is_file():
        return {"default": float(default), "tuned_val": float(default)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    default_val = float(payload.get("default", default))
    tuned = float(
        payload.get(
            "tuned_val",
            payload.get("malware_threshold", default_val),
        )
    )
    return {"default": default_val, "tuned_val": tuned}
