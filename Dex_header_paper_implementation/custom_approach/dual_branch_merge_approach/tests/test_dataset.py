"""Unit tests for DualBranchDataset and DataLoaders (synthetic shards)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import DEX_HEADER_FEATURE_DIM
from src.data.dataloaders import build_dataloaders_from_manifests, build_eval_loader, build_train_loader
from src.data.dataset import DualBranchDataset
from src.data.store import load_shard_manifest
from src.preprocessing.common import write_shard_manifest

BOW_DIM = 4381
N_SAMPLES = 32


def _write_synthetic_manifest(tmp: Path) -> Path:
    shard_dir = tmp / "shards" / "train"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for i in range(N_SAMPLES):
        apk_id = f"sample_{i:04d}"
        shard_path = shard_dir / f"{apk_id}.npz"
        np.savez_compressed(
            shard_path,
            header=np.random.rand(DEX_HEADER_FEATURE_DIM).astype(np.float32),
            bow=np.random.randint(0, 2, BOW_DIM).astype(np.float32),
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

    manifest_path = tmp / "manifest_train.json"
    write_shard_manifest(
        manifest_path,
        entries,
        header_dim=DEX_HEADER_FEATURE_DIM,
        bow_dim=BOW_DIM,
    )
    return manifest_path


class TestDualBranchDataset(unittest.TestCase):
    def test_len_and_item_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_synthetic_manifest(Path(tmp))
            ds = DualBranchDataset.from_manifest(manifest)
            self.assertEqual(len(ds), N_SAMPLES)
            header, bow, label = ds[0]
            self.assertEqual(header.shape, (DEX_HEADER_FEATURE_DIM,))
            self.assertEqual(bow.shape, (BOW_DIM,))
            self.assertEqual(label.dim(), 0)

    def test_subset_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_synthetic_manifest(Path(tmp))
            loaded = load_shard_manifest(manifest)
            ds = DualBranchDataset(
                loaded.entries,
                header_dim=loaded.header_dim,
                bow_dim=loaded.bow_dim,
                indices=[0, 2, 4],
            )
            self.assertEqual(len(ds), 3)


class TestDataLoaders(unittest.TestCase):
    def test_train_loader_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_synthetic_manifest(Path(tmp))
            ds = DualBranchDataset.from_manifest(manifest)
            loader = build_train_loader(ds, batch_size=16, num_workers=0)
            header, bow, labels = next(iter(loader))
            self.assertEqual(header.shape, (16, DEX_HEADER_FEATURE_DIM))
            self.assertEqual(bow.shape, (16, BOW_DIM))
            self.assertEqual(labels.shape, (16,))

    def test_manifest_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_synthetic_manifest(Path(tmp))
            train_loader, val_loader, header_dim, bow_dim = build_dataloaders_from_manifests(
                manifest,
                manifest,
                batch_size=16,
                num_workers=0,
            )
            self.assertEqual(header_dim, DEX_HEADER_FEATURE_DIM)
            self.assertEqual(bow_dim, BOW_DIM)
            th, tb, _ = next(iter(train_loader))
            vh, vb, _ = next(iter(val_loader))
            self.assertEqual(th.shape[1], DEX_HEADER_FEATURE_DIM)
            self.assertEqual(tb.shape[1], BOW_DIM)
            self.assertLessEqual(vh.shape[0], 16)


if __name__ == "__main__":
    unittest.main()
