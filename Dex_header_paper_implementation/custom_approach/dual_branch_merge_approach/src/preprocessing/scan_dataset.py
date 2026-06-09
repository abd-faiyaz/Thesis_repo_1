"""Walk apk_root, infer labels from folders, write index and train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ensure_artifact_dirs, load_config
from src.pipeline_integration import (
    get_pipeline_settings,
    load_shared_split_paths,
    partition_rows_from_shared_paths,
)
from src.preprocessing.common import (
    scan_apk_rows,
    stratified_split,
    temporal_three_way_split,
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

    split_mode = str(pre.get("split_mode", "stratified_random"))
    test_rows: list | None = None

    settings = get_pipeline_settings(cfg)
    shared_paths = load_shared_split_paths(settings)
    if shared_paths is not None:
        train_paths, val_paths, test_paths = shared_paths
        train_rows, val_rows, test_rows, rows = partition_rows_from_shared_paths(
            rows, train_paths, val_paths, apk_root, test_paths=test_paths
        )
        print(
            f"Using shared splits from {settings.shared_splits_dir} "
            f"(train={len(train_rows)}, val={len(val_rows)}, "
            f"test={len(test_rows) if test_rows else 0})"
        )
    else:
        if split_mode in {"temporal_holdout", "temporal_year"}:
            train_years = pre.get("train_years", [2020, 2021])
            holdout_years = pre.get("holdout_years", pre.get("test_years", [2022, 2023]))
            val_fraction = float(
                pre.get("val_fraction_of_holdout", pre.get("val_fraction", 0.5))
            )
            seed = int(pre.get("random_seed", pre.get("seed", 42)))
            train_rows, val_rows, test_rows = temporal_three_way_split(
                rows,
                train_years=train_years,
                holdout_years=holdout_years,
                val_fraction_of_holdout=val_fraction,
                seed=seed,
            )
            print(
                f"Temporal holdout split: train_years={train_years} holdout_years={holdout_years} "
                f"val_fraction_of_holdout={val_fraction} "
                f"(train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)})"
            )
        elif split_mode == "stratified_random":
            train_rows, val_rows = stratified_split(
                rows,
                train_ratio=float(pre.get("train_ratio", 0.9)),
                seed=int(pre.get("seed", 42)),
            )
        else:
            raise ValueError(
                f"Unknown preprocessing.split_mode={split_mode!r}; "
                "use 'temporal_holdout' or 'stratified_random'"
            )

    write_dataset_index(cfg.paths.dataset_index, rows)
    write_split_file(cfg.paths.splits_dir / "train.txt", train_rows)
    write_split_file(cfg.paths.splits_dir / "val.txt", val_rows)
    if test_rows is not None:
        write_split_file(cfg.paths.splits_dir / "test.txt", test_rows)

    print(f"APK root: {apk_root}")
    print(f"Indexed {len(rows)} APKs → {cfg.paths.dataset_index}")
    print(f"  train: {len(train_rows)} → {cfg.paths.splits_dir / 'train.txt'}")
    print(f"  val:   {len(val_rows)} → {cfg.paths.splits_dir / 'val.txt'}")
    if test_rows is not None:
        print(f"  test:  {len(test_rows)} → {cfg.paths.splits_dir / 'test.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
