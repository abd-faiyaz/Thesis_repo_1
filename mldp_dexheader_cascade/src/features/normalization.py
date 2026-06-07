"""Min-max normalization for Dex header feature vectors."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import DEX_FEATURE_DIM


def fit_minmax(features: np.ndarray, *, feature_dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    dim = feature_dim if feature_dim is not None else DEX_FEATURE_DIM
    if features.ndim != 2 or features.shape[1] != dim:
        raise ValueError(f"Expected (N, {dim}), got {features.shape}")
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    return mins, maxs


def transform_minmax(
    features: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    denom = maxs - mins
    denom = np.where(denom == 0, 1.0, denom)
    return (features - mins) / denom


def save_normalization_header(
    path: Path,
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    feature_dim: int = DEX_FEATURE_DIM,
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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_normalization_header(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mins = np.array(data["mins"], dtype=np.float64)
    maxs = np.array(data["maxs"], dtype=np.float64)
    return mins, maxs, data


def copy_deployed_normalization(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def transform_vector(raw: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return transform_minmax(raw.reshape(1, -1), mins, maxs).reshape(-1)
