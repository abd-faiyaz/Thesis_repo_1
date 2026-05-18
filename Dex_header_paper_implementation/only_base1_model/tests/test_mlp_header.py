"""Unit tests for MLP(H) architecture (Phase 4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features.dex_header import FEATURE_DIM
from src.models.mlp_header import MLPHeader, build_mlp_header_from_config


class TestMLPHeader(unittest.TestCase):
    def test_output_shape_batch(self) -> None:
        model = MLPHeader(input_dim=FEATURE_DIM, hidden_dim=128)
        x = torch.randn(16, FEATURE_DIM)
        y = model(x)
        self.assertEqual(y.shape, (16, 1))

    def test_output_shape_single(self) -> None:
        model = MLPHeader(input_dim=FEATURE_DIM, hidden_dim=64)
        model.eval()  # BatchNorm needs eval mode for batch size 1
        x = torch.randn(FEATURE_DIM)
        y = model(x)
        self.assertEqual(y.shape, (1, 1))

    def test_output_range_sigmoid(self) -> None:
        model = MLPHeader(input_dim=FEATURE_DIM, hidden_dim=32)
        y = model(torch.randn(8, FEATURE_DIM))
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))

    def test_wrong_input_dim_raises(self) -> None:
        model = MLPHeader(input_dim=FEATURE_DIM, hidden_dim=32)
        with self.assertRaises(ValueError):
            model(torch.randn(4, FEATURE_DIM - 1))

    def test_from_config(self) -> None:
        cfg = load_config()
        model = build_mlp_header_from_config(cfg, input_dim=FEATURE_DIM)
        self.assertEqual(model.input_dim, FEATURE_DIM)
        self.assertEqual(model.hidden_dim, int(cfg.model.get("hidden_dim", 128)))

    def test_parameter_count_positive(self) -> None:
        model = MLPHeader(input_dim=104, hidden_dim=128)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 0)


if __name__ == "__main__":
    unittest.main()
