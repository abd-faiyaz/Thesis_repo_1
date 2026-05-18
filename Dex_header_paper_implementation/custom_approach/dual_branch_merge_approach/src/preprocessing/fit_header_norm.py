"""Fit corpus min-max normalization on train-split Dex header features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import ensure_artifact_dirs, load_config
from src.features.apk_extract import ApkExtractError, extract_apk_raw_header
from src.features.dex_header import DexHeaderError
from src.features.multidex import multidex_settings
from src.features.normalization import fit_minmax, save_normalization_stats
from src.preprocessing.common import (
    log_failure,
    read_dataset_index,
    read_split_ids,
    rows_for_split,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Dex header min-max stats (train split).")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    pre = cfg.preprocessing
    md = multidex_settings(pre)

    all_rows = read_dataset_index(cfg.paths.dataset_index)
    train_ids = read_split_ids(cfg.paths.splits_dir / "train.txt")
    train_rows = rows_for_split(all_rows, train_ids)

    raw_features: list[np.ndarray] = []
    failed = 0
    for row in tqdm(train_rows, desc="Fitting header normalization", unit="apk"):
        try:
            raw_features.append(
                extract_apk_raw_header(
                    row.apk_path,
                    mode=md["mode"],
                    pattern=md["dex_pattern"],
                    max_dex=md["max_dex"],
                )
            )
        except (ApkExtractError, DexHeaderError) as exc:
            failed += 1
            log_failure(cfg.paths.failed_apks_log, row.apk_path, str(exc))
            continue

    if not raw_features:
        raise RuntimeError("No Dex headers collected for normalization")

    matrix = np.stack(raw_features, axis=0)
    mins, maxs = fit_minmax(matrix)
    save_normalization_stats(
        cfg.paths.normalization_stats,
        mins,
        maxs,
        extra={
            "num_train_apks": len(train_rows),
            "dex_failures": failed,
            "multidex_mode": md["mode"],
        },
    )

    print(f"Fitted on {matrix.shape[0]} train APKs → {cfg.paths.normalization_stats}")
    if failed:
        print(f"  {failed} train APK(s) skipped (logged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
