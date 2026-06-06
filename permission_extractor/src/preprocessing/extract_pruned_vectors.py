"""P2c — encode pruned vectors using frozen MLDP set S."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import ensure_artifact_dirs, load_config
from src.features.permission_vector import build_binary_vector, load_selected_permissions
from src.preprocessing.common import read_dataset_index

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_manifest(processed: Path, split: str, entries: list[dict]) -> None:
    path = processed / f"manifest_{split}.json"
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract MLDP-pruned vector shards.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "dev_test", "temporal_holdout"],
    )
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    if not cfg.paths.selected_permissions.is_file():
        raise SystemExit("Missing selected_permissions.json; run run_mldp_selection.py first")

    permissions, token_to_index = load_selected_permissions(cfg.paths.selected_permissions)
    vector_size = len(permissions)
    rows = read_dataset_index(cfg.paths.dataset_index)

    for split in args.splits:
        split_rows = [r for r in rows if r.split == split]
        shard_root = cfg.paths.processed / "shards" / split
        shard_root.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []

        for row in tqdm(split_rows, desc=f"shards:{split}"):
            tx_path = cfg.paths.transactions_dir / split / f"{row.apk_id}.json"
            if not tx_path.is_file():
                continue
            data = json.loads(tx_path.read_text(encoding="utf-8"))
            vec = build_binary_vector(
                data["permissions"], token_to_index, vector_size=vector_size
            )
            shard_path = shard_root / f"{row.apk_id}.npz"
            np.savez_compressed(
                shard_path,
                x=vec.astype(np.float32),
                label=np.int64(row.label),
                apk_id=row.apk_id,
            )
            entries.append(
                {
                    "apk_id": row.apk_id,
                    "path": str(row.apk_path),
                    "label": row.label,
                    "shard": str(shard_path.relative_to(cfg.paths.processed)),
                }
            )

        _write_manifest(cfg.paths.processed, split, entries)
        print(f"{split}: {len(entries)} shards (dim={vector_size})")

    meta = {
        "S": vector_size,
        "selected_permissions": str(cfg.paths.selected_permissions.resolve()),
    }
    (cfg.paths.processed / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
