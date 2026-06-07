"""Classification metrics for training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    cm = confusion_matrix(
        np.asarray(y_true).astype(int).ravel(),
        np.asarray(y_pred).astype(int).ravel(),
        labels=[0, 1],
    )
    return cm.astype(int).tolist()


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def format_metrics(metrics: dict[str, float]) -> str:
    auc = metrics.get("roc_auc", float("nan"))
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "n/a"
    return (
        f"ACC={metrics['accuracy']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"AUC={auc_str}"
    )
