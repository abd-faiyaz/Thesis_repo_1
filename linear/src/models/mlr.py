"""LinRegDroid MLR + LinRegDroid1 decision rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression


@dataclass
class MLRFitResult:
    sklearn_model: LinearRegression
    feature_dim: int
    intercept: float
    coefficients: np.ndarray


class LinRegDroidModule(nn.Module):
    """
    PyTorch mirror of sklearn MLR.
    Output: malware_probability in [0, 1] via clamp(linear_score, 0, 1).
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.linear(x)
        return torch.clamp(raw, 0.0, 1.0)

    @classmethod
    def from_sklearn(cls, model: LinearRegression, feature_dim: int) -> "LinRegDroidModule":
        module = cls(feature_dim)
        coef = model.coef_.reshape(-1).astype(np.float32)
        bias = float(model.intercept_.reshape(-1)[0])
        with torch.no_grad():
            module.linear.weight.copy_(torch.from_numpy(coef).view(1, -1))
            module.linear.bias.copy_(torch.tensor([bias], dtype=torch.float32))
        return module


def fit_mlr(X: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True) -> MLRFitResult:
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)
    coef = model.coef_.reshape(-1)
    intercept = float(model.intercept_.reshape(-1)[0])
    return MLRFitResult(
        sklearn_model=model,
        feature_dim=X.shape[1],
        intercept=intercept,
        coefficients=coef,
    )


def linregdroid1_predict(malware_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return binary labels: 0 benign, 1 malware."""
    return (malware_prob >= threshold).astype(np.int64)


def linregdroid2_predict(raw_scores: np.ndarray) -> np.ndarray:
    """
    Nearest-class rule on raw linear scores using paper encodings (benign=1, malware=0),
    mapped back to thesis labels (0 benign, 1 malware).
    """
    paper_class = np.where(
        np.abs(raw_scores - 0.0) <= np.abs(raw_scores - 1.0),
        0,
        1,
    )
    return np.where(paper_class == 1, 0, 1).astype(np.int64)


def raw_linear_scores(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(X).reshape(-1)
