"""Build x_S, H, and fused x vectors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from src.constants import DEX_FEATURE_DIM


def build_binary_block(present: Iterable[str], vocab: Sequence[str]) -> np.ndarray:
    present_set = set(present)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for i, token in enumerate(vocab):
        if token in present_set:
            vec[i] = 1.0
    return vec


def vectorize_mldp(permissions: Iterable[str], mldp_vocab: Sequence[str]) -> np.ndarray:
    return build_binary_block(permissions, mldp_vocab)


def vectorize_cascade(
    permissions: Iterable[str],
    h_normalized: np.ndarray,
    *,
    mldp_vocab: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_s = vectorize_mldp(permissions, mldp_vocab)
    h = np.asarray(h_normalized, dtype=np.float32).reshape(DEX_FEATURE_DIM)
    x = np.concatenate([x_s, h], axis=0)
    return x_s, h, x
