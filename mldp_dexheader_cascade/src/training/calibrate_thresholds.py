"""P6 — calibrate Mode A threshold and Mode B cascade t_low/t_high on val only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PipelineConfig
from src.training.metrics import tune_threshold


def false_omission_rate(y_true: np.ndarray, s1: np.ndarray, t_low: float) -> float:
    """Fraction of malware samples with s1 <= t_low (benign early-exit mistakes)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    malware = y_true == 1
    if not malware.any():
        return 0.0
    return float((s1[malware] <= t_low).mean())


def false_alarm_rate(y_true: np.ndarray, s1: np.ndarray, t_high: float) -> float:
    """Fraction of benign samples with s1 >= t_high (malware early-exit mistakes)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    benign = y_true == 0
    if not benign.any():
        return 0.0
    return float((s1[benign] >= t_high).mean())


def calibrate_t_low(
    y_true: np.ndarray,
    s1: np.ndarray,
    *,
    target_for: float = 0.02,
) -> tuple[float, float]:
    """Pick highest t_low with false-omission rate <= target on val malware."""
    y_true = np.asarray(y_true).astype(int).ravel()
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    malware_scores = np.sort(s1[y_true == 1])
    if malware_scores.size == 0:
        return 0.5, 0.0

    best_t = 0.0
    best_for = 1.0
    for t in np.unique(malware_scores):
        for_rate = false_omission_rate(y_true, s1, float(t))
        if for_rate <= target_for and t >= best_t:
            best_t = float(t)
            best_for = for_rate
    if best_t == 0.0 and best_for > target_for:
        best_t = float(np.quantile(malware_scores, target_for))
        best_for = false_omission_rate(y_true, s1, best_t)
    return best_t, best_for


def calibrate_t_high(
    y_true: np.ndarray,
    s1: np.ndarray,
    *,
    target_fa: float = 0.02,
    t_low: float = 0.0,
) -> tuple[float, float]:
    """Pick lowest t_high with false-alarm rate <= target on val benign; enforce t_high > t_low."""
    y_true = np.asarray(y_true).astype(int).ravel()
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    benign_scores = np.sort(s1[y_true == 0])
    if benign_scores.size == 0:
        return 0.5, 0.0

    best_t = 1.0
    best_fa = 1.0
    for t in np.unique(benign_scores):
        if t <= t_low:
            continue
        fa_rate = false_alarm_rate(y_true, s1, float(t))
        if fa_rate <= target_fa and t <= best_t:
            best_t = float(t)
            best_fa = fa_rate
    if best_t >= 1.0 or best_t <= t_low:
        candidates = benign_scores[benign_scores > t_low]
        if candidates.size:
            best_t = float(np.quantile(candidates, 1.0 - target_fa))
            best_t = max(best_t, t_low + 1e-4)
        else:
            best_t = min(1.0, t_low + 0.05)
        best_fa = false_alarm_rate(y_true, s1, best_t)
    return best_t, best_fa


def calibrate_cascade_thresholds(
    y_true: np.ndarray,
    s1: np.ndarray,
    *,
    target_false_omission_rate: float = 0.02,
    target_false_alarm_at_thigh: float = 0.02,
) -> dict[str, float]:
    t_low, val_for = calibrate_t_low(
        y_true, s1, target_for=target_false_omission_rate
    )
    t_high, val_fa = calibrate_t_high(
        y_true, s1, target_fa=target_false_alarm_at_thigh, t_low=t_low
    )
    if t_high <= t_low:
        t_high = min(1.0, t_low + 0.05)

    uncertain = (s1 > t_low) & (s1 < t_high)
    step1_exit_rate_val = float(1.0 - uncertain.mean()) if s1.size else 0.0

    return {
        "stage1_t_low": t_low,
        "stage1_t_high": t_high,
        "val_false_omission_rate_at_t_low": val_for,
        "val_false_alarm_rate_at_t_high": val_fa,
        "val_step1_exit_rate": step1_exit_rate_val,
    }


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
