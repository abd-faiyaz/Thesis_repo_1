"""Unit tests for Phase 6 evaluation metrics."""

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
from src.features.dex_header import FEATURE_DIM
from src.models.mlp_header import MLPHeader
from src.training.evaluate import compute_metrics, format_metrics, run_evaluation
from src.training.setup import build_training_objects
from src.data.dataloaders import build_dataloaders_from_bundle
from src.data.store import ProcessedBundle
from src.training.checkpoint import build_checkpoint_state, save_checkpoint
from src.training.evaluate import validation_epoch


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_metrics(y_true, y_pred, y_score)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertAlmostEqual(m["roc_auc"], 1.0)

    def test_format_metrics(self) -> None:
        s = format_metrics({"accuracy": 0.9, "f1": 0.85, "roc_auc": 0.88})
        self.assertIn("ACC=0.9000", s)
        self.assertIn("F1=0.8500", s)


class TestValidationEpoch(unittest.TestCase):
    def test_validation_epoch_returns_metrics(self) -> None:
        bundle = ProcessedBundle(
            features=torch.rand(32, FEATURE_DIM),
            labels=torch.tensor([0.0] * 16 + [1.0] * 16),
            paths=[f"s_{i}.apk" for i in range(32)],
            feature_dim=FEATURE_DIM,
            source_path=Path("synthetic.pt"),
        )
        _, val_loader, _ = build_dataloaders_from_bundle(
            bundle, batch_size=8, num_workers=0, val_fraction=0.25
        )
        model = MLPHeader(FEATURE_DIM, 32)
        cfg = load_config()
        criterion, _, _, device = build_training_objects(cfg, model)
        model.to(device)

        loss, metrics = validation_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
            show_progress=False,
        )
        self.assertGreaterEqual(loss, 0.0)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)


class TestRunEvaluation(unittest.TestCase):
    def test_run_evaluation_from_checkpoint(self) -> None:
        bundle = ProcessedBundle(
            features=torch.rand(40, FEATURE_DIM),
            labels=torch.randint(0, 2, (40,)).float(),
            paths=[f"s_{i}.apk" for i in range(40)],
            feature_dim=FEATURE_DIM,
            source_path=Path("synthetic.pt"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed"
            processed.mkdir()
            ckpt_dir = tmp_path / "checkpoints"
            ckpt_dir.mkdir()

            torch.save(
                {
                    "features": bundle.features,
                    "labels": bundle.labels,
                    "paths": bundle.paths,
                    "feature_dim": FEATURE_DIM,
                },
                processed / "dex_header_features.pt",
            )

            cfg = load_config()
            hidden_dim = int(cfg.model.get("hidden_dim", 128))
            model = MLPHeader(FEATURE_DIM, hidden_dim)
            _, optimizer, scheduler, device = build_training_objects(cfg, model)
            ckpt_path = ckpt_dir / "latest_checkpoint.pth"
            save_checkpoint(
                ckpt_path,
                build_checkpoint_state(
                    next_epoch=1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=0.5,
                    val_loss=0.5,
                    feature_dim=FEATURE_DIM,
                    hidden_dim=hidden_dim,
                ),
            )

            cfg_data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
            cfg_data["paths"]["processed_dir"] = str(processed)
            cfg_data["paths"]["checkpoint_dir"] = str(ckpt_dir)
            cfg_data["paths"]["latest_checkpoint"] = str(ckpt_path)
            cfg_data["training"]["device"] = "cpu"
            cfg_data["data"]["num_workers"] = 0
            cfg_path = tmp_path / "cfg.yaml"
            cfg_path.write_text(yaml.dump(cfg_data))

            result = run_evaluation(load_config(cfg_path), checkpoint_path=ckpt_path)
            self.assertIn("accuracy", result)
            self.assertIn("f1", result)


if __name__ == "__main__":
    unittest.main()
