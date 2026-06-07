"""Tests for support filtering and association rule mining."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mldp.association_rules import mine_rule_permissions
from src.mldp.support_filter import filter_by_support


class TestSupportFilter(unittest.TestCase):
    def test_filters_by_support_bounds(self) -> None:
        txs = [
            {"a", "b"},
            {"a"},
            {"b"},
            set(),
        ]
        kept, stats = filter_by_support(["a", "b", "c"], txs, min_support=0.25, max_support=0.9)
        self.assertEqual(kept, ["a", "b"])
        self.assertAlmostEqual(stats["a"], 0.5)
        self.assertAlmostEqual(stats["c"], 0.0)


class TestAssociationRules(unittest.TestCase):
    def test_mines_permissions_from_malware_transactions(self) -> None:
        malware_tx = [
            {"permissions::sms", "permissions::internet"},
            {"permissions::sms", "permissions::camera"},
            {"permissions::sms", "permissions::internet"},
            {"permissions::sms"},
        ]
        candidates = ["permissions::sms", "permissions::internet", "permissions::camera"]
        selected, rules = mine_rule_permissions(
            malware_tx,
            candidates,
            min_support=0.5,
            min_confidence=0.5,
            min_lift=1.0,
        )
        self.assertIn("permissions::sms", selected)
        self.assertGreater(len(rules), 0)

    def test_empty_inputs(self) -> None:
        selected, rules = mine_rule_permissions([], ["a"], min_support=0.1)
        self.assertEqual(selected, set())
        self.assertEqual(rules, [])


if __name__ == "__main__":
    unittest.main()
