"""Smoke test for Phase 6 helpers (balance + package, no real APKs)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DEX_HEADER_FEATURE_DIM
from src.preprocessing.common import write_shard_manifest

BOW_DIM = 4381


def _setup_artifacts(tmp: Path) -> Path:
    for split, n in (("train", 40), ("val", 10)):
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
                label=np.int64(1 if i % 3 == 0 else 0),
            )
            entries.append(
                {
                    "apk_id": apk_id,
                    "shard_path": str(shard_path.resolve()),
                    "apk_path": f"/fake/{apk_id}.apk",
                    "label": 1 if i % 3 == 0 else 0,
                }
            )
        write_shard_manifest(
            tmp / "processed" / f"manifest_{split}.json",
            entries,
            header_dim=DEX_HEADER_FEATURE_DIM,
            bow_dim=BOW_DIM,
        )

    cfg_data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    cfg_data["paths"]["processed_dir"] = str(tmp / "processed")
    cfg_data["paths"]["checkpoint_dir"] = str(tmp / "checkpoints")
    cfg_data["paths"]["latest_checkpoint"] = str(tmp / "checkpoints" / "latest.pt")
    cfg_data["paths"]["best_checkpoint"] = str(tmp / "checkpoints" / "best.pt")
    cfg_data["paths"]["class_balance"] = str(tmp / "class_balance.json")
    cfg_data["paths"]["artifacts_bundle"] = str(tmp / "pattern_b_bundle.tar.gz")
    cfg_data["paths"]["manifest_train"] = str(tmp / "processed" / "manifest_train.json")
    cfg_data["paths"]["vocab"] = str(tmp / "vocab.json")
    cfg_data["paths"]["normalization_stats"] = str(tmp / "normalization_header.json")

    (tmp / "vocab.json").write_text('{"token_to_index": {}, "unk_index": 0, "vector_size": 1}')
    (tmp / "normalization_header.json").write_text(
        '{"mins": [0], "maxs": [1], "feature_dim": 104}'
    )

    cfg_path = tmp / "test_config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    return cfg_path


class TestPhase6Helpers(unittest.TestCase):
    def test_compute_balance_and_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = _setup_artifacts(tmp_path)
            cfg = load_config(cfg_path)

            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "compute_class_balance.py"), "--config", str(cfg_path)],
                check=True,
                cwd=ROOT,
            )
            self.assertTrue(cfg.paths.class_balance.is_file())
            data = json.loads(cfg.paths.class_balance.read_text())
            self.assertIn("pos_weight", data)

            from src.training.train import run_training

            run_training(cfg, epochs_override=1, fresh_start=True)

            artifacts_root = tmp_path
            env = {
                **os.environ,
                "BUNDLE": str(cfg.paths.artifacts_bundle),
                "ARTIFACTS_ROOT": str(artifacts_root),
            }
            subprocess.run(
                [str(ROOT / "scripts" / "package_artifacts.sh")],
                check=True,
                cwd=ROOT,
                env=env,
            )
            self.assertTrue(cfg.paths.artifacts_bundle.is_file())


if __name__ == "__main__":
    unittest.main()
