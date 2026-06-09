#!/usr/bin/env python3
"""Smoke-test PermissionDataset and split manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import make_dataloader
from src.preprocessing.common import read_dataset_index, split_counts


def main() -> int:
    cfg = load_config()
    index_path = cfg.paths.dataset_index
    if not index_path.is_file():
        print(f"Missing dataset index: {index_path}\nRun scan_dataset.py first.")
        return 1

    rows = read_dataset_index(index_path)
    counts = split_counts(rows)
    print("Split counts:")
    for name in ("train", "val", "test", "other"):
        print(f"  {name}: {counts.get(name, 0)}")

    for split in ("train", "val"):
        manifest = cfg.paths.processed / f"manifest_{split}.json"
        if not manifest.is_file():
            print(f"Missing {manifest}; run extract_to_cache.py first.")
            return 1
        loader = make_dataloader(cfg, split, batch_size=32)
        batch_x, batch_y = next(iter(loader))
        print(f"{split} batch: x={tuple(batch_x.shape)} y={tuple(batch_y.shape)}")
    print("DataLoader OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
