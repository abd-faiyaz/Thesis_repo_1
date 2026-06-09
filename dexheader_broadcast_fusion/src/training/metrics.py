"""Classification metrics for malware detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    y_true = y_true.astype(int).ravel()
    y_pred = y_pred.astype(int).ravel()
    y_score = y_score.astype(float).ravel()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"acc={metrics.get('accuracy', 0):.4f} "
        f"f1={metrics.get('f1', 0):.4f} "
        f"auc={metrics.get('roc_auc', float('nan')):.4f}"
    )
