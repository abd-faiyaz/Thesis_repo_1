"""Extract per-APK .npz shards (header H, manifest BoW I, label) with resume support."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.constants import DEX_HEADER_FEATURE_DIM
from src.features.apk_extract import ApkExtractError, extract_apk_raw_header
from src.features.dex_header import DexHeaderError
from src.features.multidex import multidex_settings
from src.features.manifest_bow import (
    ManifestBoWError,
    build_multihot_vector,
    extract_manifest_tokens,
    load_vocab,
)
from src.features.normalization import load_normalization_stats, transform_minmax
from src.preprocessing.common import (
    DatasetRow,
    append_processed_id,
    load_processed_ids,
    log_failure,
    read_dataset_index,
    read_split_ids,
    rows_for_split,
    write_shard_manifest,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _shard_dir(cfg: PipelineConfig, split: str) -> Path:
    if split == "train":
        return cfg.paths.shards_train_dir
    if split == "val":
        return cfg.paths.shards_val_dir
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")


def _manifest_path(cfg: PipelineConfig, split: str) -> Path:
    if split == "train":
        return cfg.paths.manifest_train
    return cfg.paths.manifest_val


def extract_split(cfg: PipelineConfig, split: str, *, limit: int | None = None) -> dict[str, int]:
    pre = cfg.preprocessing
    md = multidex_settings(pre)

    token_to_index, unk_index, vector_size = load_vocab(cfg.paths.vocab)
    mins, maxs = load_normalization_stats(cfg.paths.normalization_stats)

    all_rows = read_dataset_index(cfg.paths.dataset_index)
    split_ids = read_split_ids(cfg.paths.splits_dir / f"{split}.txt")
    rows = rows_for_split(all_rows, split_ids)
    if limit is not None:
        rows = rows[:limit]

    shard_dir = _shard_dir(cfg, split)
    shard_dir.mkdir(parents=True, exist_ok=True)
    done_ids = load_processed_ids(cfg.paths.processed_ids_log)

    failed = 0
    skipped = 0
    newly_written = 0

    for row in tqdm(rows, desc=f"Extracting shards ({split})", unit="apk"):
        shard_path = shard_dir / f"{row.apk_id}.npz"
        if row.apk_id in done_ids or shard_path.is_file():
            skipped += 1
            continue

        try:
            header, bow = _extract_features(
                row,
                multidex=md,
                token_to_index=token_to_index,
                unk_index=unk_index,
                vector_size=vector_size,
                mins=mins,
                maxs=maxs,
            )
        except (ApkExtractError, DexHeaderError, ManifestBoWError) as exc:
            failed += 1
            log_failure(cfg.paths.failed_apks_log, row.apk_path, str(exc))
            continue
        except Exception as exc:
            failed += 1
            log_failure(cfg.paths.failed_apks_log, row.apk_path, f"unexpected: {exc}")
            continue

        # np.savez_compressed always appends ".npz" — use ".tmp.npz" not ".npz.tmp".
        tmp_path = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            header=header.astype(np.float32),
            bow=bow,
            label=np.int64(row.label),
        )
        tmp_path.replace(shard_path)
        append_processed_id(cfg.paths.processed_ids_log, row.apk_id)
        newly_written += 1

    entries: list[dict[str, object]] = []
    for row in rows:
        shard_path = shard_dir / f"{row.apk_id}.npz"
        if shard_path.is_file():
            entries.append(_entry_from_row(row, shard_path))

    if not entries:
        raise RuntimeError(f"No shards for split={split}; see failed_apks.log")

    write_shard_manifest(
        _manifest_path(cfg, split),
        entries,
        header_dim=DEX_HEADER_FEATURE_DIM,
        bow_dim=vector_size,
        multidex_mode=md["mode"],
    )

    return {
        "split": split,
        "total": len(rows),
        "shards": len(entries),
        "newly_written": newly_written,
        "skipped_existing": skipped,
        "failed": failed,
        "manifest": str(_manifest_path(cfg, split)),
    }


def _entry_from_row(row: DatasetRow, shard_path: Path) -> dict[str, object]:
    return {
        "apk_id": row.apk_id,
        "shard_path": str(shard_path.resolve()),
        "apk_path": str(row.apk_path),
        "label": row.label,
    }


def _extract_features(
    row: DatasetRow,
    *,
    multidex: dict[str, object],
    token_to_index: dict[str, int],
    unk_index: int,
    vector_size: int,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_header = extract_apk_raw_header(
        row.apk_path,
        mode=str(multidex["mode"]),
        pattern=str(multidex["dex_pattern"]),
        max_dex=int(multidex["max_dex"]),
    )
    header = transform_minmax(raw_header.reshape(1, -1), mins, maxs)[0]

    tokens = extract_manifest_tokens(row.apk_path)
    bow = build_multihot_vector(
        tokens,
        token_to_index,
        vector_size=vector_size,
        unk_index=unk_index,
    )
    return header, bow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract per-APK feature shards.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=("train", "val", "both"),
        default="both",
        help="Which split to extract (default: both)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N APKs per split")
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        summary = extract_split(cfg, split, limit=args.limit)
        print(f"\n{split}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
