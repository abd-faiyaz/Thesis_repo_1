#!/usr/bin/env python3
"""P3 — smoke test HybridManifestDataset / DataLoaders."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import build_dataloaders


def main() -> int:
    cfg = load_config()
    train_loader, val_loader, test_loader, feature_dim, balance_stats = build_dataloaders(cfg)

    tx, ty = next(iter(train_loader))
    vx, vy = next(iter(val_loader))
    sx, sy = next(iter(test_loader))

    print("\nP3 DataLoader smoke test")
    print(f"  feature_dim d: {feature_dim}")
    print(f"  train batch: x={tuple(tx.shape)}  y={tuple(ty.shape)}  dtype=({tx.dtype}, {ty.dtype})")
    print(f"  val   batch: x={tuple(vx.shape)}  y={tuple(vy.shape)}")
    print(f"  test  batch: x={tuple(sx.shape)}  y={tuple(sy.shape)}")

    for name, labels in (("train", ty), ("val", vy), ("test", sy)):
        uniq = sorted({int(v) for v in labels.tolist()})
        if not set(uniq).issubset({0, 1}):
            print(f"ERROR: {name} labels not in {{0,1}}: {uniq}")
            return 1

    if tx.shape[1] != feature_dim:
        print(f"ERROR: batch feature dim {tx.shape[1]} != {feature_dim}")
        return 1

    pos_weight = balance_stats.get("pos_weight", {}).get("value")
    if pos_weight is None:
        print("ERROR: pos_weight not computed from train split")
        return 1

    print(f"  pos_weight: {pos_weight:.4f}")
    print("\nP3 exit criteria met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
