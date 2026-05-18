"""Tests for training loop and checkpoint resume (Phase 5)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.store import ProcessedBundle
from src.features.dex_header import FEATURE_DIM
from src.training.checkpoint import (
    build_checkpoint_state,
    load_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
)
from src.training.setup import build_training_objects
from src.models.mlp_header import MLPHeader


class TestCheckpoint(unittest.TestCase):
    def test_save_restore_roundtrip(self) -> None:
        model = MLPHeader(FEATURE_DIM, 32)
        cfg = load_config()
        _, optimizer, scheduler, _ = build_training_objects(cfg, model)

        state = build_checkpoint_state(
            next_epoch=3,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=0.42,
            val_loss=0.51,
            feature_dim=FEATURE_DIM,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest_checkpoint.pth"
            save_checkpoint(path, state)
            loaded = load_checkpoint(path)
            assert loaded is not None

            model2 = MLPHeader(FEATURE_DIM, 32)
            _, opt2, sch2, _ = build_training_objects(cfg, model2)
            nxt = restore_from_checkpoint(loaded, model2, opt2, sch2)
            self.assertEqual(nxt, 3)
            for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
                self.assertTrue(torch.allclose(p1, p2))


class TestTrainingRun(unittest.TestCase):
    def test_short_training_with_synthetic_bundle(self) -> None:
        bundle = ProcessedBundle(
            features=torch.rand(64, FEATURE_DIM),
            labels=torch.randint(0, 2, (64,)).float(),
            paths=[f"s_{i}.apk" for i in range(64)],
            feature_dim=FEATURE_DIM,
            source_path=Path("synthetic.pt"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "artifacts" / "processed"
            processed.mkdir(parents=True)
            ckpt_dir = tmp_path / "artifacts" / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            torch.save(
                {
                    "features": bundle.features,
                    "labels": bundle.labels,
                    "paths": bundle.paths,
                    "feature_dim": FEATURE_DIM,
                },
                processed / "dex_header_features.pt",
            )

            cfg_data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
            cfg_data["paths"]["processed_dir"] = str(processed)
            cfg_data["paths"]["checkpoint_dir"] = str(ckpt_dir)
            cfg_data["paths"]["latest_checkpoint"] = str(ckpt_dir / "latest_checkpoint.pth")
            cfg_data["training"]["device"] = "cpu"
            cfg_data["training"]["epochs"] = 2
            cfg_data["data"]["num_workers"] = 0
            cfg_data["data"]["pin_memory"] = False

            cfg_path = tmp_path / "test_config.yaml"
            cfg_path.write_text(yaml.dump(cfg_data))

            from src.training.train import run_training

            run_training(load_config(cfg_path), epochs_override=2, fresh_start=True)
            ckpt = load_checkpoint(ckpt_dir / "latest_checkpoint.pth")
            self.assertIsNotNone(ckpt)
            self.assertEqual(ckpt["next_epoch"], 2)

            # Resume should not raise
            run_training(load_config(cfg_path), epochs_override=3, fresh_start=False)


if __name__ == "__main__":
    unittest.main()
