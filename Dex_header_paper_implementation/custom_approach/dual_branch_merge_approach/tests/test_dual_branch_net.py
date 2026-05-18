"""Unit tests for DualBranchNet and submodules (Phase 4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DEX_HEADER_FEATURE_DIM
from src.models.adaptive_shrinkage_unit import AdaptiveShrinkageUnit
from src.models.ascnn_manifest import ASCNNManifest
from src.models.dual_branch_net import DualBranchNet, build_dual_branch_net_from_config
from src.models.fusion_head import FusionHead
from src.models.mlp_header import MLPHeaderBranch

BOW_DIM = 4381
EMBED_DIM = 128


class TestSubmodules(unittest.TestCase):
    def test_asu_output_rank(self) -> None:
        asu = AdaptiveShrinkageUnit(1, 64, kernel_size=3, stride=2)
        x = torch.randn(4, 1, BOW_DIM)
        y = asu(x)
        self.assertEqual(y.dim(), 3)
        self.assertEqual(y.shape[0], 4)
        self.assertEqual(y.shape[1], 64)

    def test_mlp_header_embedding(self) -> None:
        mlp = MLPHeaderBranch(input_dim=104, embed_dim=128)
        e = mlp(torch.randn(8, 104))
        self.assertEqual(e.shape, (8, 128))

    def test_ascnn_embedding(self) -> None:
        ascnn = ASCNNManifest(bow_dim=BOW_DIM, embed_dim=128)
        e = ascnn(torch.randn(8, BOW_DIM))
        self.assertEqual(e.shape, (8, 128))

    def test_fusion_logit(self) -> None:
        head = FusionHead(input_dim=256, hidden_dim=128)
        logit = head(torch.randn(8, 256))
        self.assertEqual(logit.shape, (8, 1))


class TestDualBranchNet(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_dual_branch_net_from_config(load_config())

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
