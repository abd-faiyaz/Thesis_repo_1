"""Tests for offline cascade simulation."""

from __future__ import annotations

import unittest

from shared_calibration.cascade_simulate import simulate_cascade_batch, summarize_outcomes


def _val_scores(model_id: str, apk_ids: list[str], scores: list[float], labels: list[int]) -> dict:
    return {
        "model_id": model_id,
        "split": "val",
        "alignment_key": "sha256",
        "n_samples": len(apk_ids),
        "n_aligned": len(apk_ids),
        "threshold": 0.5,
        "metrics": {"accuracy": 0.9, "f1": 0.88},
        "rows": [
            {"apk_id": apk_id, "label": label, "score": score}
            for apk_id, label, score in zip(apk_ids, labels, scores)
        ],
    }


class CascadeSimulateTest(unittest.TestCase):
    def test_tier_one_early_benign_exit(self) -> None:
        apk_ids = ["a" * 64, "b" * 64]
        labels = [0, 1]
        payloads = {
            "mldp_pruned_permission": _val_scores(
                "mldp_pruned_permission", apk_ids, [0.01, 0.99], labels
            ),
            "broadcast_mldp_hybrid": _val_scores(
                "broadcast_mldp_hybrid", apk_ids, [0.02, 0.98], labels
            ),
            "mldp_dexheader_cascade_mode_b": _val_scores(
                "mldp_dexheader_cascade", apk_ids, [0.5, 0.5], labels
            ),
            "early_fusion_dex_manifest": _val_scores(
                "early_fusion_dex_manifest", apk_ids, [0.5, 0.5], labels
            ),
        }
        policy = {
            "policy_name": "test",
            "tier3_pattern_model": "early_fusion_dex_manifest",
            "model_weights": {
                "mldp_pruned_permission": 1.0,
                "broadcast_mldp_hybrid": 1.0,
                "mldp_dexheader_cascade_mode_b": 1.0,
                "early_fusion_dex_manifest": 1.0,
            },
            "fusion_weights": {"early_fusion_dex_manifest": 1.0},
            "tiers": [
                {
                    "tier": 1,
                    "models": ["mldp_pruned_permission", "broadcast_mldp_hybrid"],
                    "t_low": 0.15,
                    "t_high": 0.85,
                    "conservative_malware_or": False,
                },
                {
                    "tier": 2,
                    "models": ["mldp_dexheader_cascade_mode_b"],
                    "t_low": 0.2,
                    "t_high": 0.8,
                },
                {
                    "tier": 3,
                    "models": ["early_fusion_dex_manifest"],
                    "t_low": 0.3,
                    "t_high": 0.7,
                },
                {"tier": 4, "models": ["bytecnn"], "t_low": 0.5, "t_high": 0.5, "final": True},
            ],
        }
        outcomes = simulate_cascade_batch(policy, payloads)
        self.assertEqual(outcomes[0].exit_tier, 1)
        self.assertEqual(outcomes[0].decision, "benign")
        self.assertEqual(outcomes[1].exit_tier, 1)
        self.assertEqual(outcomes[1].decision, "malware")

    def test_summarize_outcomes_reports_f1(self) -> None:
        outcomes = [
            type("O", (), {"label": 0, "decision": "benign", "models_run": ["a"], "exit_tier": 1})(),
            type("O", (), {"label": 1, "decision": "malware", "models_run": ["a", "b"], "exit_tier": 4})(),
        ]
        stats = summarize_outcomes(outcomes)
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["accuracy"], 1.0)
        self.assertAlmostEqual(stats["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
