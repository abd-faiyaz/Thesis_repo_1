"""Unit tests for DexDataset and DataLoaders (synthetic .pt bundle)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import (
    build_dataloaders_from_bundle,
    build_eval_loader,
    build_train_loader,
    split_train_val_indices,
)
from src.data.dataset import DexDataset
from src.data.store import ProcessedBundle, load_processed_bundle
from src.features.dex_header import FEATURE_DIM


def _make_bundle(n: int = 32) -> ProcessedBundle:
    features = torch.rand(n, FEATURE_DIM)
    labels = torch.randint(0, 2, (n,)).float()
    paths = [f"/fake/sample_{i}.apk" for i in range(n)]
    return ProcessedBundle(
        features=features,
        labels=labels,
        paths=paths,
        feature_dim=FEATURE_DIM,
        source_path=Path("synthetic.pt"),
    )


class TestDexDataset(unittest.TestCase):
    def test_len_and_item_shape(self) -> None:
        bundle = _make_bundle(10)
        ds = DexDataset.from_bundle(bundle)
        self.assertEqual(len(ds), 10)
        x, y = ds[0]
        self.assertEqual(x.shape, (FEATURE_DIM,))
        self.assertEqual(y.dim(), 0)

    def test_subset_indices(self) -> None:
        bundle = _make_bundle(10)
        ds = DexDataset.from_bundle(bundle, indices=[0, 2, 4])
        self.assertEqual(len(ds), 3)

    def test_save_and_load_pt(self) -> None:
        bundle = _make_bundle(8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dex_header_features.pt"
            torch.save(
                {
                    "features": bundle.features,
                    "labels": bundle.labels,
                    "paths": bundle.paths,
                    "feature_dim": bundle.feature_dim,
                },
                path,
            )
            loaded = load_processed_bundle(path)
            ds = DexDataset.from_processed_file(path)
            self.assertEqual(len(ds), 8)
            self.assertEqual(loaded.feature_dim, FEATURE_DIM)


class TestDataLoaders(unittest.TestCase):
    def test_train_loader_batch(self) -> None:
        bundle = _make_bundle(32)
        ds = DexDataset.from_bundle(bundle)
        loader = build_train_loader(ds, batch_size=16, num_workers=0)
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.shape, (16, FEATURE_DIM))
        self.assertEqual(batch_y.shape, (16,))

    def test_eval_loader_sequential(self) -> None:
        bundle = _make_bundle(4)
        ds = DexDataset.from_bundle(bundle, indices=[0, 1, 2, 3])
        loader = build_eval_loader(ds, batch_size=2, num_workers=0)
        batches = list(loader)
        self.assertEqual(len(batches), 2)

    def test_split_and_build_pair(self) -> None:
        bundle = _make_bundle(100)
        train_loader, val_loader, dim = build_dataloaders_from_bundle(
            bundle,
            val_fraction=0.2,
            batch_size=16,
            num_workers=0,
        )
        self.assertEqual(dim, FEATURE_DIM)
        tx, _ = next(iter(train_loader))
        vx, _ = next(iter(val_loader))
        self.assertEqual(tx.shape[1], FEATURE_DIM)
        self.assertLessEqual(vx.shape[0], 16)

    def test_split_indices_sizes(self) -> None:
        train_idx, val_idx = split_train_val_indices(100, val_fraction=0.2, seed=42)
        self.assertEqual(train_idx.numel() + val_idx.numel(), 100)


if __name__ == "__main__":
    unittest.main()
