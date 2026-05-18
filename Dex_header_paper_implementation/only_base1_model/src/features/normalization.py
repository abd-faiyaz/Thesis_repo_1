"""Min-max normalization for Dex header feature vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.dex_header import FEATURE_DIM


def fit_minmax(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension min and max from shape (N, D)."""
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected (N, {FEATURE_DIM}), got {features.shape}")
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    return mins, maxs


def transform_minmax(
    features: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    """Apply min-max scaling to [0, 1]; constant dimensions map to 0."""
    denom = maxs - mins
    denom = np.where(denom == 0, 1.0, denom)
    return (features - mins) / denom


def save_normalization_stats(
    path: Path,
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    feature_dim: int = FEATURE_DIM,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "feature_dim": feature_dim,
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_normalization_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["mins"], dtype=np.float64), np.array(data["maxs"], dtype=np.float64)
