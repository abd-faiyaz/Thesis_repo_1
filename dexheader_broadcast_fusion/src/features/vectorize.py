"""Binary receiver-action vectorization against frozen vocab A."""

from __future__ import annotations

import numpy as np


def vectorize_receiver_actions(
    actions: list[str] | tuple[str, ...],
    *,
    receiver_vocab: list[str],
) -> np.ndarray:
    """R[k] = 1.0 iff action_k in vocab and present in APK (set semantics)."""
    present = set(actions)
    vec = np.zeros(len(receiver_vocab), dtype=np.float32)
    for idx, token in enumerate(receiver_vocab):
        if token in present:
            vec[idx] = 1.0
    return vec
