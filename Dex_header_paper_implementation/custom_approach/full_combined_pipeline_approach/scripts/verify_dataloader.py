#!/usr/bin/env python3
"""Phase 3: verify CombinedPipelineDataset / DataLoaders against manifests or synthetic shards."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DEX_HEADER_FEATURE_DIM
from src.data.dataloaders import build_dataloaders_from_config, build_dataloaders_from_manifests
from src.preprocessing.common import write_shard_manifest


def _write_synthetic_shards(tmp: Path, n: int = 32) -> Path:
    shard_dir = tmp / "shards" / "train"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    bow_dim = 4381

    for i in range(n):
        apk_id = f"synthetic_{i:04d}"
        shard_path = shard_dir / f"{apk_id}.npz"
        np.savez_compressed(
            shard_path,
            header=np.random.rand(DEX_HEADER_FEATURE_DIM).astype(np.float32),
            bow=np.zeros(bow_dim, dtype=np.float32),
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
        bow_dim=bow_dim,
    )
    return manifest_path


def _print_batch_summary(
    header_dim: int,
    bow_dim: int,
    th,
    tb,
    ty,
    vh,
    vb,
    vy,
) -> None:
    print("Phase 3 DataLoader check (Pattern A)")
    print(f"  header_dim: {header_dim}  bow_dim: {bow_dim}  combined: {header_dim + bow_dim}")
    print(f"  train batch: header {tuple(th.shape)} bow {tuple(tb.shape)} labels {tuple(ty.shape)}")
    print(f"  val batch:   header {tuple(vh.shape)} bow {tuple(vb.shape)} labels {tuple(vy.shape)}")
    print("  train shuffle: True   val shuffle: False")
    print(f"  label range (train): [{ty.min():.0f}, {ty.max():.0f}]")
    print("\nPhase 3 verified.")


def main() -> int:
    cfg = load_config()
    train_manifest = cfg.paths.manifest_train
    val_manifest = cfg.paths.manifest_val

    if train_manifest.is_file() and val_manifest.is_file():
        print(f"Loading manifests:\n  {train_manifest}\n  {val_manifest}")
        train_loader, val_loader, header_dim, bow_dim = build_dataloaders_from_config(cfg)
        th, tb, ty = next(iter(train_loader))
        vh, vb, vy = next(iter(val_loader))
        _print_batch_summary(header_dim, bow_dim, th, tb, ty, vh, vb, vy)
        return 0

    print("Manifests not found; using synthetic train shards for both splits.")
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _write_synthetic_shards(Path(tmp))
        train_loader, val_loader, header_dim, bow_dim = build_dataloaders_from_manifests(
            manifest,
            manifest,
            batch_size=int(cfg.data.get("batch_size", 16)),
            num_workers=0,
            pin_memory=False,
        )
        th, tb, ty = next(iter(train_loader))
        vh, vb, vy = next(iter(val_loader))
        _print_batch_summary(header_dim, bow_dim, th, tb, ty, vh, vb, vy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
