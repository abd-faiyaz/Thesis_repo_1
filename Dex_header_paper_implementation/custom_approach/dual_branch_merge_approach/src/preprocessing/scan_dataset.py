"""Walk apk_root, infer labels from folders, write index and train/val splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ensure_artifact_dirs, load_config
from src.pipeline_integration import (
    get_pipeline_settings,
    load_shared_train_val_paths,
    partition_rows_from_shared_paths,
)
from src.preprocessing.common import (
    scan_apk_rows,
    stratified_split,
    write_dataset_index,
    write_split_file,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan APK tree and build dataset index.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--apk-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Index only first N APKs")
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    pre = cfg.preprocessing

    apk_root = Path(args.apk_root or cfg.paths.apk_root)
    rows = scan_apk_rows(cfg, apk_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    settings = get_pipeline_settings(cfg)
    shared_paths = load_shared_train_val_paths(settings)
    if shared_paths is not None:
        train_paths, val_paths = shared_paths
        train_rows, val_rows, rows = partition_rows_from_shared_paths(
            rows, train_paths, val_paths, apk_root
        )
        print(
            f"Using shared splits from {settings.shared_splits_dir} "
            f"(train={len(train_rows)}, val={len(val_rows)})"
        )
    else:
        train_rows, val_rows = stratified_split(
            rows,
            train_ratio=float(pre.get("train_ratio", 0.9)),
            seed=int(pre.get("seed", 42)),
        )

    write_dataset_index(cfg.paths.dataset_index, rows)
    write_split_file(cfg.paths.splits_dir / "train.txt", train_rows)
    write_split_file(cfg.paths.splits_dir / "val.txt", val_rows)

    print(f"APK root: {apk_root}")
    print(f"Indexed {len(rows)} APKs → {cfg.paths.dataset_index}")
    print(f"  train: {len(train_rows)} → {cfg.paths.splits_dir / 'train.txt'}")
    print(f"  val:   {len(val_rows)} → {cfg.paths.splits_dir / 'val.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
