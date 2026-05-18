"""Tests for evaluation metrics (Phase 6)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DEX_HEADER_FEATURE_DIM
from src.models.dual_branch_net import build_dual_branch_net_from_config
from src.preprocessing.common import write_shard_manifest
from src.training.evaluate import compute_metrics, format_metrics

BOW_DIM = 4381


def _write_manifest(tmp: Path, split: str, n: int) -> None:
    shard_dir = tmp / "processed" / "shards" / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n):
        apk_id = f"{split}_{i:04d}"
        shard_path = shard_dir / f"{apk_id}.npz"
        np.savez_compressed(
            shard_path,
            header=np.random.rand(DEX_HEADER_FEATURE_DIM).astype(np.float32),
            bow=np.zeros(BOW_DIM, dtype=np.float32),
            label=np.int64(i % 2),
        )
        entries.append(
            {
                "apk_id": apk_id,
                "shard_path": str(shard_path.resolve()),
                "apk_path": f"/fake/{apk_id}.apk",
                "label": i % 2,
            }
        )
    write_shard_manifest(
        tmp / "processed" / f"manifest_{split}.json",
        entries,
        header_dim=DEX_HEADER_FEATURE_DIM,
        bow_dim=BOW_DIM,
    )


class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        y_score = np.array([0.1, 0.9, 0.4, 0.2])
        m = compute_metrics(y_true, y_pred, y_score)
        self.assertIn("accuracy", m)
        self.assertIn("f1", m)
        self.assertTrue(format_metrics(m).startswith("ACC="))


class TestEvaluateRun(unittest.TestCase):
    def test_run_evaluation_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "artifacts" / "processed"
            ckpt = tmp_path / "artifacts" / "checkpoints"
            ckpt.mkdir(parents=True)
            _write_manifest(tmp_path / "artifacts", "train", 32)
            _write_manifest(tmp_path / "artifacts", "val", 16)

            cfg_data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
            cfg_data["paths"]["processed_dir"] = str(processed)
            cfg_data["paths"]["checkpoint_dir"] = str(ckpt)
            cfg_data["paths"]["latest_checkpoint"] = str(ckpt / "latest.pt")
            cfg_data["paths"]["best_checkpoint"] = str(ckpt / "best.pt")
            cfg_data["paths"]["class_balance"] = str(tmp_path / "artifacts" / "class_balance.json")
            cfg_data["training"]["device"] = "cpu"
            cfg_data["data"]["num_workers"] = 0

            cfg_path = tmp_path / "cfg.yaml"
            cfg_path.write_text(yaml.dump(cfg_data))

            from src.training.train import run_training

            run_training(load_config(cfg_path), epochs_override=1, fresh_start=True)

            from src.training.evaluate import run_evaluation

            result = run_evaluation(load_config(cfg_path), split="val")
            self.assertIn("accuracy", result)


if __name__ == "__main__":
    unittest.main()
