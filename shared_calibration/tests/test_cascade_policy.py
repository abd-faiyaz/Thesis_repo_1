"""Tests for cross-model cascade policy builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shared_calibration.cascade_policy import (
    build_cascade_policy,
    inner_join_val_scores,
    load_mode_b_bands_from_thresholds,
    simulate_cumulative_exits,
    write_json,
)


def _make_val_scores(model_id: str, apk_ids: list[str], scores: list[float], labels: list[int]) -> dict:
    return {
        "model_id": model_id,
        "split": "val",
        "alignment_key": "sha256",
        "n_samples": len(apk_ids),
        "n_aligned": len(apk_ids),
        "threshold": 0.5,
        "metrics": {"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.95},
        "rows": [
            {"apk_id": apk_id, "label": label, "score": score}
            for apk_id, label, score in zip(apk_ids, labels, scores)
        ],
    }


class CascadePolicyBuilderTest(unittest.TestCase):
    def test_inner_join_keeps_apk_intersection_only(self) -> None:
        apk_a = "a" * 64
        apk_b = "b" * 64
        payloads = {
            "mldp_pruned_permission": _make_val_scores(
                "mldp_pruned_permission", [apk_a, apk_b], [0.1, 0.9], [0, 1]
            ),
            "broadcast_mldp_hybrid": _make_val_scores(
                "broadcast_mldp_hybrid", [apk_a], [0.2], [0]
            ),
        }
        aligned = inner_join_val_scores(
            payloads,
            required_models=["mldp_pruned_permission", "broadcast_mldp_hybrid"],
        )
        self.assertEqual(aligned.apk_ids, [apk_a])

    def test_build_policy_calibrates_tier_one(self) -> None:
        apk_ids = [("a" * 64), ("b" * 64), ("c" * 64), ("d" * 64)]
        labels = [0, 0, 1, 1]
        payloads = {
            "mldp_pruned_permission": _make_val_scores(
                "mldp_pruned_permission",
                apk_ids,
                [0.05, 0.15, 0.85, 0.95],
                labels,
            ),
            "broadcast_mldp_hybrid": _make_val_scores(
                "broadcast_mldp_hybrid",
                apk_ids,
                [0.1, 0.2, 0.8, 0.9],
                labels,
            ),
        }
        tier_specs = [
            {
                "tier": 1,
                "models": ["mldp_pruned_permission", "broadcast_mldp_hybrid"],
                "conservative_malware_or": True,
            }
        ]
        policy, report = build_cascade_policy(tier_specs=tier_specs, payloads=payloads)
        tier1 = policy["tiers"][0]
        self.assertLess(tier1["t_low"], tier1["t_high"])
        self.assertIn("model_weights", policy)
        self.assertFalse(policy["enabled"])
        self.assertEqual(report["n_aligned_apks"], 4)

    def test_load_mode_b_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(
                json.dumps(
                    {
                        "mode_b": {
                            "stage1_t_low": 0.083,
                            "stage1_t_high": 0.832,
                            "val_step1_exit_rate": 0.52,
                        }
                    }
                ),
                encoding="utf-8",
            )
            bands = load_mode_b_bands_from_thresholds(path)
            self.assertAlmostEqual(bands["t_low"], 0.083)
            self.assertAlmostEqual(bands["t_high"], 0.832)

    def test_simulate_cumulative_exits_monotonic(self) -> None:
        apk_ids = ["x" * 64, "y" * 64]
        aligned = inner_join_val_scores(
            {
                "mldp_pruned_permission": _make_val_scores(
                    "mldp_pruned_permission", apk_ids, [0.1, 0.9], [0, 1]
                )
            },
            required_models=["mldp_pruned_permission"],
        )
        tier_specs = [{"tier": 1, "models": ["mldp_pruned_permission"]}]
        tier_bands = {
            1: {
                "t_low": 0.2,
                "t_high": 0.8,
                "val_step1_exit_rate": 1.0,
            }
        }
        tier_scores = {1: np.array([0.1, 0.9])}
        report = simulate_cumulative_exits(aligned, tier_specs, tier_bands, tier_scores)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["cumulative_exit_rate"], 1.0)

    def test_write_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            write_json(path, {"policy_name": "test", "enabled": False, "tiers": []})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["policy_name"], "test")


if __name__ == "__main__":
    unittest.main()
