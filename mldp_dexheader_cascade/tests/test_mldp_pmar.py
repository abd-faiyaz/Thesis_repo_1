"""Smoke tests for OOM-safe PMAR miner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.mldp.pmar import collapse_by_implications, mine_association_rules


class TestPairwisePmar(unittest.TestCase):
    def test_mines_rules_without_mlxtend(self) -> None:
        malware_tx = [
            {"permissions::a", "permissions::b"},
            {"permissions::a", "permissions::b", "permissions::c"},
            {"permissions::a", "permissions::b"},
            {"permissions::b", "permissions::c"},
        ]
        candidates = [
            "permissions::a",
            "permissions::b",
            "permissions::c",
        ]
        rules, implications = mine_association_rules(
            malware_tx,
            candidates,
            min_support=0.5,
            min_confidence=0.9,
        )
        self.assertTrue(rules)
        self.assertTrue(implications)

    def test_collapse_removes_consequents(self) -> None:
        kept, removed = collapse_by_implications(
            ["permissions::a", "permissions::b", "permissions::c"],
            [("permissions::a", "permissions::c")],
            {},
        )
        self.assertIn("permissions::a", kept)
        self.assertIn("permissions::c", removed)


if __name__ == "__main__":
    unittest.main()
