"""Concatenate MLDP permission bits and receiver-action bits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def build_binary_block(
    present: Iterable[str],
    vocab: Sequence[str],
) -> np.ndarray:
    present_set = set(present)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for i, token in enumerate(vocab):
        if token in present_set:
            vec[i] = 1.0
    return vec


def vectorize_hybrid(
    permissions: Iterable[str],
    receiver_actions: Iterable[str],
    *,
    mldp_vocab: Sequence[str],
    receiver_vocab: Sequence[str],
) -> np.ndarray:
    x_s = build_binary_block(permissions, mldp_vocab)
    x_r = build_binary_block(receiver_actions, receiver_vocab)
    return np.concatenate([x_s, x_r], axis=0)
