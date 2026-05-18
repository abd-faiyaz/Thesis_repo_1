"""Tests for training loop and checkpoint resume (Phase 5)."""

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
from src.training.checkpoint import (
    build_checkpoint_state,
    load_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
)
from src.training.losses import build_criterion, resolve_pos_weight
from src.training.setup import build_training_objects

BOW_DIM = 4381
N_SAMPLES = 64


def _write_split_shards(base: Path, split: str, n: int) -> Path:
    shard_dir = base / "processed" / "shards" / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for i in range(n):
        apk_id = f"{split}_{i:04d}"
        shard_path = shard_dir / f"{apk_id}.npz"
        np.savez_compressed(
            shard_path,
            header=np.random.rand(DEX_HEADER_FEATURE_DIM).astype(np.float32),
            bow=np.random.rand(BOW_DIM).astype(np.float32),
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
    manifest_path = base / "processed" / f"manifest_{split}.json"
    write_shard_manifest(
        manifest_path,
        entries,
        header_dim=DEX_HEADER_FEATURE_DIM,
        bow_dim=BOW_DIM,
    )
    return manifest_path


def _make_test_config(tmp: Path) -> Path:
    cfg_data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    processed = tmp / "artifacts" / "processed"
    ckpt = tmp / "artifacts" / "checkpoints"
    ckpt.mkdir(parents=True)

    _write_split_shards(tmp / "artifacts", "train", N_SAMPLES)
    _write_split_shards(tmp / "artifacts", "val", 16)

    cfg_data["paths"]["processed_dir"] = str(processed)
    cfg_data["paths"]["checkpoint_dir"] = str(ckpt)
    cfg_data["paths"]["latest_checkpoint"] = str(ckpt / "latest.pt")
    cfg_data["paths"]["best_checkpoint"] = str(ckpt / "best.pt")
    cfg_data["training"]["device"] = "cpu"
    cfg_data["training"]["epochs"] = 2
    cfg_data["data"]["batch_size"] = 16
    cfg_data["data"]["num_workers"] = 0
    cfg_data["data"]["pin_memory"] = False

    cfg_path = tmp / "test_config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    return cfg_path


class TestLosses(unittest.TestCase):
    def test_pos_weight_from_ratio(self) -> None:
        cfg = load_config()
        raw = dict(cfg.raw)
        raw["training"] = dict(cfg.training)
        raw["training"]["pos_weight"] = None
        raw["training"]["benign_to_malware_ratio"] = 1.5
        from src.config import PipelineConfig

        cfg2 = PipelineConfig(root=cfg.root, paths=cfg.paths, raw=raw)
        self.assertEqual(resolve_pos_weight(cfg2), 1.5)


class TestCheckpoint(unittest.TestCase):
    def test_save_restore_roundtrip(self) -> None:
        cfg = load_config()
        model = build_dual_branch_net_from_config(cfg)
        optimizer, scheduler, device = build_training_objects(cfg, model)
        criterion = build_criterion(cfg, device)

        state = build_checkpoint_state(
            next_epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=0.5,
            val_loss=0.6,
            best_val_loss=0.6,
            global_step=10,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            save_checkpoint(path, state)
            loaded = load_checkpoint(path)
            assert loaded is not None

            model2 = build_dual_branch_net_from_config(cfg)
            opt2, sch2, _ = build_training_objects(cfg, model2)
            nxt, step, best = restore_from_checkpoint(loaded, model2, opt2, sch2)
            self.assertEqual(nxt, 2)
            self.assertEqual(step, 10)
            self.assertEqual(best, 0.6)
            for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
                self.assertTrue(torch.allclose(p1, p2))


class TestTrainingRun(unittest.TestCase):
    def test_short_training_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _make_test_config(Path(tmp))
            from src.training.train import run_training

            run_training(load_config(cfg_path), epochs_override=2, fresh_start=True)
            ckpt = load_checkpoint(Path(tmp) / "artifacts" / "checkpoints" / "latest.pt")
            self.assertIsNotNone(ckpt)
            self.assertEqual(ckpt["next_epoch"], 2)

            run_training(load_config(cfg_path), epochs_override=3, fresh_start=False)


if __name__ == "__main__":
    unittest.main()
