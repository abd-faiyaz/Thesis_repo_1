"""Aggregate per-Dex header vectors from multi-Dex APKs."""

from __future__ import annotations

import re

import numpy as np

from src.features.dex_header import FEATURE_DIM

DEFAULT_DEX_PATTERN = r"^classes(\d*)\.dex$"
DEFAULT_MULTIDEX_MODE = "sum"


class MultidexError(ValueError):
    """Invalid multidex configuration or aggregation mode."""


def dex_suffix_sort_key(basename: str) -> tuple[int, str]:
    match = re.match(r"^classes(\d*)\.dex$", basename)
    if not match:
        return (999_999, basename)
    suffix = match.group(1)
    order = 0 if suffix == "" else int(suffix)
    return (order, basename)


def aggregate_header_vectors(
    vectors: list[np.ndarray],
    mode: str,
    *,
    max_dex: int = 3,
) -> np.ndarray:
    if not vectors:
        raise MultidexError("No header vectors to aggregate")

    for i, vec in enumerate(vectors):
        if vec.shape != (FEATURE_DIM,):
            raise MultidexError(
                f"Vector {i} has shape {vec.shape}, expected ({FEATURE_DIM},)"
            )

    stacked = np.stack(vectors, axis=0)

    if mode == "sum":
        return stacked.sum(axis=0)
    if mode == "mean":
        return stacked.mean(axis=0)
    if mode == "primary_only":
        return vectors[0].copy()
    if mode == "concat":
        out = np.zeros(FEATURE_DIM * max_dex, dtype=np.float64)
        for i, vec in enumerate(vectors[:max_dex]):
            out[i * FEATURE_DIM : (i + 1) * FEATURE_DIM] = vec
        return out

    raise MultidexError(f"Unknown multidex mode: {mode!r}")
