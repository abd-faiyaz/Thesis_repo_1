"""Tests for MLR fit and LinRegDroid decision rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.mlr import (
    LinRegDroidModule,
    fit_mlr,
    linregdroid1_predict,
    linregdroid2_predict,
    predict_variant,
    raw_linear_scores,
)


class TestMLRFit(unittest.TestCase):
    def test_fit_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.integers(0, 2, size=(40, 5)).astype(np.float64)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
        fit = fit_mlr(X, y)
        self.assertEqual(fit.feature_dim, 5)
        self.assertEqual(fit.coefficients.shape, (5,))
        preds = raw_linear_scores(fit.sklearn_model, X)
        self.assertEqual(preds.shape, (40,))

    def test_torch_module_matches_sklearn(self) -> None:
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        y = np.array([1.0, 0.0], dtype=np.float64)
        fit = fit_mlr(X, y)
        module = LinRegDroidModule.from_sklearn(fit.sklearn_model, fit.feature_dim)
        import torch

        with torch.no_grad():
            out = module(torch.from_numpy(X.astype(np.float32))).numpy().reshape(-1)
        expected = np.clip(raw_linear_scores(fit.sklearn_model, X), 0.0, 1.0)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


class TestDecisionRules(unittest.TestCase):
    def test_linregdroid1_threshold(self) -> None:
        scores = np.array([0.2, 0.5, 0.9])
        self.assertEqual(linregdroid1_predict(scores, threshold=0.5).tolist(), [0, 1, 1])

    def test_linregdroid2_nearest_class(self) -> None:
        raw = np.array([0.2, 0.8, 0.45, 0.55])
        self.assertEqual(linregdroid2_predict(raw).tolist(), [1, 0, 1, 0])

    def test_predict_variant_switch(self) -> None:
        model = LinearRegression()
        model.coef_ = np.array([1.0, -1.0])
        model.intercept_ = np.array([0.0])
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        pred1, scores1 = predict_variant(model, X, variant="linregdroid1", threshold=0.5)
        pred2, scores2 = predict_variant(model, X, variant="linregdroid2")
        self.assertEqual(pred1.tolist(), [1, 0])
        self.assertEqual(pred2.tolist(), [0, 1])
        np.testing.assert_allclose(scores1, np.clip([1.0, -1.0], 0, 1))
        np.testing.assert_allclose(scores2, [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
