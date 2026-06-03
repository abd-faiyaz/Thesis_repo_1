#!/usr/bin/env python3
"""Count train-split labels and write recommended BCE pos_weight (Phase 6)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.store import load_shard_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute class balance from train manifest.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    manifest = load_shard_manifest(cfg.paths.manifest_train)

    counts = Counter(entry.label for entry in manifest.entries)
    n_benign = int(counts.get(0, 0))
    n_malware = int(counts.get(1, 0))
    if n_malware == 0:
        raise RuntimeError("No malware (label=1) samples in train manifest")

    pos_weight = n_benign / n_malware
    ratio = pos_weight

    payload = {
        "n_benign": n_benign,
        "n_malware": n_malware,
        "benign_to_malware_ratio": ratio,
        "pos_weight": pos_weight,
        "source_manifest": str(cfg.paths.manifest_train),
    }

    out_path = cfg.paths.class_balance
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Class balance (train split)")
    print(f"  benign:   {n_benign}")
    print(f"  malware:  {n_malware}")
    print(f"  pos_weight (n_benign / n_malware): {pos_weight:.4f}")
    print(f"  written → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
