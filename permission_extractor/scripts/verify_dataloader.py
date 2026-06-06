#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import make_dataloader
from src.preprocessing.common import read_dataset_index, split_counts


def main() -> int:
    cfg = load_config()
    if not cfg.paths.dataset_index.is_file():
        print("Run scan_dataset.py first.")
        return 1

    rows = read_dataset_index(cfg.paths.dataset_index)
    for name, count in split_counts(rows).items():
        print(f"  {name}: {count}")

    for split in ("train", "val"):
        manifest = cfg.paths.processed / f"manifest_{split}.json"
        if not manifest.is_file():
            print(f"Missing {manifest}")
            return 1
        batch_x, batch_y = next(iter(make_dataloader(cfg, split, batch_size=32)))
        print(f"{split}: x={tuple(batch_x.shape)} y={tuple(batch_y.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
