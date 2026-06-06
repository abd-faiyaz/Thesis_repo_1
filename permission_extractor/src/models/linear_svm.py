"""Linear SVM with sigmoid calibration."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def train_linear_svc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    cv: int = 3,
) -> CalibratedClassifierCV:
    base = LinearSVC(C=C, class_weight=class_weight, max_iter=10_000, dual="auto")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    model.fit(X, y)
    return model


def malware_probabilities(model: CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
