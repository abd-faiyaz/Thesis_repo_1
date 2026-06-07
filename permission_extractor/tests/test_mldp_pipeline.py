"""Tests for frozen set validation, vectors, and model selection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features.permission_vector import build_binary_vector, save_selected_permissions
from src.mldp.validation import validate_selected_set
from src.training.train import _val_f1_from_probs


class TestFrozenSetValidation(unittest.TestCase):
    def test_in_range_passes(self) -> None:
        cfg = load_config()
        metadata = {"fallback_used": False, "n_from_rules": 30}
        result = validate_selected_set(["p"] * 30, metadata, cfg)
        self.assertTrue(result["in_expected_range"])
        self.assertTrue(result["passed"])

    def test_small_s_warns(self) -> None:
        cfg = load_config()
        result = validate_selected_set(["p"] * 10, {"fallback_used": False}, cfg)
        self.assertFalse(result["in_expected_range"])
        self.assertFalse(result["passed"])

    def test_fallback_fails_validation(self) -> None:
        cfg = load_config()
        result = validate_selected_set(["p"] * 25, {"fallback_used": True}, cfg)
        self.assertFalse(result["passed"])
        self.assertTrue(any("fallback" in w for w in result["warnings"]))

    def test_existing_smoke_run_selection(self) -> None:
        selected_path = ROOT / "artifacts/mldp/selected_permissions.json"
        if not selected_path.is_file():
            self.skipTest("no selected_permissions.json from prior run")
        import json

        cfg = load_config()
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        perms = data["permissions"]
        metadata = {k: v for k, v in data.items() if k != "permissions"}
        result = validate_selected_set(perms, metadata, cfg)
        self.assertEqual(result["s_size"], data["S"])
        self.assertTrue(result["in_expected_range"])


class TestPrunedVectors(unittest.TestCase):
    def test_vector_matches_frozen_set_size(self) -> None:
        perms = ["permissions::a", "permissions::b", "permissions::c"]
        token_to_index = {p: i for i, p in enumerate(perms)}
        vec = build_binary_vector(["permissions::b"], token_to_index, vector_size=len(perms))
        np.testing.assert_array_equal(vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def test_save_selected_permissions_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.json"
            save_selected_permissions(
                path,
                ["permissions::sms"],
                metadata={"association_rule_mode": "malware_only_itemsets"},
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("malware_only_itemsets", text)
            self.assertIn('"S": 1', text)


class TestModelSelection(unittest.TestCase):
    def test_val_f1_from_probs(self) -> None:
        y = np.array([0, 1, 1, 0], dtype=np.float64)
        probs = np.array([0.1, 0.9, 0.8, 0.2])
        f1 = _val_f1_from_probs(y, probs, threshold=0.5)
        self.assertGreater(f1, 0.5)

    def test_candidate_selection_by_val_f1(self) -> None:
        scores = {"linear_svc": 0.7612, "tiny_mlp": 0.7424}
        best = max(scores, key=scores.get)
        self.assertEqual(best, "linear_svc")


if __name__ == "__main__":
    unittest.main()
