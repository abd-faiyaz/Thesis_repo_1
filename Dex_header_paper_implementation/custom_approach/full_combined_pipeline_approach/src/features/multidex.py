"""Aggregate per-Dex header vectors from multi-Dex APKs."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.constants import DEX_HEADER_FEATURE_DIM

DEFAULT_DEX_PATTERN = r"^classes(\d*)\.dex$"
DEFAULT_MULTIDEX_MODE = "sum"


class MultidexError(ValueError):
    """Invalid multidex configuration or aggregation mode."""


def multidex_settings(preprocessing: dict[str, Any]) -> dict[str, Any]:
    """Resolve multidex options from config preprocessing section."""
    md = preprocessing.get("multidex") or {}
    return {
        "mode": str(md.get("mode", DEFAULT_MULTIDEX_MODE)),
        "dex_pattern": str(md.get("dex_pattern", DEFAULT_DEX_PATTERN)),
        "max_dex": int(md.get("max_dex", 3)),
    }


def dex_suffix_sort_key(basename: str) -> tuple[int, str]:
    """Sort classes.dex before classes2.dex, classes3.dex, …"""
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
    """Combine per-Dex 104-d vectors into one header feature vector."""
    if not vectors:
        raise MultidexError("No header vectors to aggregate")

    for i, vec in enumerate(vectors):
        if vec.shape != (DEX_HEADER_FEATURE_DIM,):
            raise MultidexError(
                f"Vector {i} has shape {vec.shape}, expected ({DEX_HEADER_FEATURE_DIM},)"
            )

    stacked = np.stack(vectors, axis=0)

    if mode == "sum":
        return stacked.sum(axis=0)
    if mode == "mean":
        return stacked.mean(axis=0)
    if mode == "primary_only":
        return vectors[0].copy()
    if mode == "concat":
        dim = DEX_HEADER_FEATURE_DIM
        out = np.zeros(dim * max_dex, dtype=np.float64)
        for i, vec in enumerate(vectors[:max_dex]):
            out[i * dim : (i + 1) * dim] = vec
        return out

    raise MultidexError(f"Unknown multidex mode: {mode!r}")
