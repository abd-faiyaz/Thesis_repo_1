"""Tests for PRNR ranking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mldp.prnr import compute_prnr


class TestPRNR(unittest.TestCase):
    def test_ranks_malware_skewed_permissions_higher(self) -> None:
        transactions = [
            {"permissions::sms", "permissions::internet"},
            {"permissions::sms", "permissions::camera"},
            {"permissions::internet"},
            {"permissions::internet"},
        ]
        labels = [1, 1, 0, 0]
        result = compute_prnr(transactions, labels, min_rate_delta=0.0, top_k=10)
        self.assertIn("permissions::sms", result.ranked)
        self.assertGreater(
            result.scores["permissions::sms"],
            result.scores.get("permissions::internet", 0.0),
        )

    def test_requires_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            compute_prnr([{"a"}], [1])


if __name__ == "__main__":
    unittest.main()
