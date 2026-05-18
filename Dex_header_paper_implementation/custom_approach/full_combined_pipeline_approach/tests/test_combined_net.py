"""Unit tests for CombinedNet and submodules (Phase 4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DEX_HEADER_FEATURE_DIM, DEFAULT_COMBINED_INPUT_LEN
from src.models.adaptive_shrinkage_unit import AdaptiveShrinkageUnit
from src.models.ascnn_combined import ASCNNCombined
from src.models.classifier_head import ClassifierHead
from src.models.combined_net import CombinedNet, build_combined_net_from_config

BOW_DIM = 4381
COMBINED_LEN = DEFAULT_COMBINED_INPUT_LEN
PADDED_LEN = 4488
EMBED_DIM = 128


class TestSubmodules(unittest.TestCase):
    def test_asu_output_rank(self) -> None:
        asu = AdaptiveShrinkageUnit(1, 64, kernel_size=3, stride=2)
        x = torch.randn(4, 1, PADDED_LEN)
        y = asu(x)
        self.assertEqual(y.dim(), 3)
        self.assertEqual(y.shape[0], 4)
        self.assertEqual(y.shape[1], 64)

    def test_ascnn_combined_embedding(self) -> None:
        ascnn = ASCNNCombined(combined_padded_len=PADDED_LEN, embed_dim=EMBED_DIM)
        e = ascnn(torch.randn(8, COMBINED_LEN))
        self.assertEqual(e.shape, (8, EMBED_DIM))

    def test_ascnn_pads_shorter_input(self) -> None:
        ascnn = ASCNNCombined(combined_padded_len=PADDED_LEN, embed_dim=EMBED_DIM)
        e = ascnn(torch.randn(4, COMBINED_LEN))
        self.assertEqual(e.shape, (4, EMBED_DIM))

    def test_classifier_logit(self) -> None:
        head = ClassifierHead(embed_dim=128, hidden_dim=128)
        logit = head(torch.randn(8, 128))
        self.assertEqual(logit.shape, (8, 1))


class TestCombinedNet(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_combined_net_from_config(load_config())

    def test_forward_shapes(self) -> None:
        header = torch.randn(16, DEX_HEADER_FEATURE_DIM)
        bow = torch.randn(16, BOW_DIM)
        logits = self.model(header, bow)
        self.assertEqual(logits.shape, (16, 1))

    def test_predict_proba_range(self) -> None:
        probs = self.model.predict_proba(
            torch.randn(4, DEX_HEADER_FEATURE_DIM),
            torch.randn(4, BOW_DIM),
        )
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))

    def test_wrong_header_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.randn(4, 100), torch.randn(4, BOW_DIM))

    def test_concat_order_header_first(self) -> None:
        """CombinedNet uses [H || I] — spot-check via internal concat length."""
        self.assertEqual(self.model.combined_dim, DEX_HEADER_FEATURE_DIM + BOW_DIM)

    def test_parameter_count_positive(self) -> None:
        n = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n, 0)

    def test_single_sample(self) -> None:
        self.model.eval()
        logits = self.model(
            torch.randn(DEX_HEADER_FEATURE_DIM),
            torch.randn(BOW_DIM),
        )
        self.assertEqual(logits.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
