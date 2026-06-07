#!/usr/bin/env python3
"""Phase 3: verify DexDataset / DataLoaders against processed file or synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import (
    build_dataloaders_from_bundle,
    build_dataloaders_from_config,
    resolve_processed_path,
    resolve_split_settings,
)
from src.data.store import load_processed_bundle
from src.features.dex_header import FEATURE_DIM


def _synthetic_bundle(n: int = 32):
    from src.data.store import ProcessedBundle

    return ProcessedBundle(
        features=torch.rand(n, FEATURE_DIM),
        labels=torch.randint(0, 2, (n,)).float(),
        paths=[f"synthetic_{i}.apk" for i in range(n)],
        feature_dim=FEATURE_DIM,
        source_path=Path("synthetic"),
    )


def main() -> int:
    cfg = load_config()
    processed_path = resolve_processed_path(cfg)

    if processed_path.is_file():
        print(f"Loading processed bundle: {processed_path}")
        bundle = load_processed_bundle(processed_path)
    else:
        print(f"Processed file not found ({processed_path}); using synthetic bundle.")
        bundle = _synthetic_bundle()

    if processed_path.is_file():
        train_loader, val_loader, feature_dim = build_dataloaders_from_config(cfg)
    else:
        split = resolve_split_settings(cfg)
        train_loader, val_loader, feature_dim = build_dataloaders_from_bundle(
            bundle,
            split_mode=split["split_mode"],
            train_years=split["train_years"],
            test_years=split["test_years"],
            val_fraction=split["val_fraction"],
            seed=split["seed"],
            batch_size=int(cfg.data.get("batch_size", 16)),
            num_workers=0,
            pin_memory=False,
        )

    tx, ty = next(iter(train_loader))
    vx, vy = next(iter(val_loader))

    print("Phase 3 DataLoader check")
    print(f"  feature_dim: {feature_dim}")
    print(f"  train batches (approx): {len(train_loader)}  batch shape: {tuple(tx.shape)}")
    print(f"  val batches (approx):   {len(val_loader)}  batch shape: {tuple(vx.shape)}")
    print(f"  train shuffle: True   val shuffle: False")
    print(f"  label range: train [{ty.min():.0f}, {ty.max():.0f}]")
    print("\nPhase 3 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
