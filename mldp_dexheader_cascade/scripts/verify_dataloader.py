#!/usr/bin/env python3
"""P3 — smoke test CascadeDataset / DataLoaders."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import build_dataloaders


def main() -> int:
    cfg = load_config()
    train_loader, val_loader, test_loader, feature_dims, balance_stats = build_dataloaders(cfg)

    tx_s, th, tx, ty = next(iter(train_loader))
    vx_s, vh, vx, vy = next(iter(val_loader))
    sx_s, sh, sx, sy = next(iter(test_loader))

    print("\nP3 DataLoader smoke test")
    print(f"  feature_dims: {feature_dims}")
    print(
        f"  train batch: x_S={tuple(tx_s.shape)}  H={tuple(th.shape)}  "
        f"x={tuple(tx.shape)}  y={tuple(ty.shape)}"
    )
    print(
        f"  val   batch: x_S={tuple(vx_s.shape)}  H={tuple(vh.shape)}  "
        f"x={tuple(vx.shape)}  y={tuple(vy.shape)}"
    )
    print(
        f"  test  batch: x_S={tuple(sx_s.shape)}  H={tuple(sh.shape)}  "
        f"x={tuple(sx.shape)}  y={tuple(sy.shape)}"
    )

    for name, labels in (("train", ty), ("val", vy), ("test", sy)):
        uniq = sorted({int(v) for v in labels.tolist()})
        if not set(uniq).issubset({0, 1}):
            print(f"ERROR: {name} labels not in {{0,1}}: {uniq}")
            return 1

    s_dim = feature_dims["S"]
    h_dim = feature_dims["H"]
    d_dim = feature_dims["d"]
    if tx_s.shape[1] != s_dim or th.shape[1] != h_dim or tx.shape[1] != d_dim:
        print(
            f"ERROR: train batch dims mismatch: "
            f"x_S={tx_s.shape[1]} H={th.shape[1]} x={tx.shape[1]}"
        )
        return 1
    if s_dim + h_dim != d_dim:
        print(f"ERROR: S + H != d: {s_dim} + {h_dim} != {d_dim}")
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
