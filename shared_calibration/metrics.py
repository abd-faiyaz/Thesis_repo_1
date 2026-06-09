"""Classification threshold tuning."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pick threshold on val that maximizes malware F1."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (y_score >= t).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t
