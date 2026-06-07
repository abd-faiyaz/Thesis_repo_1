#!/usr/bin/env python3
"""P1 — Index APK corpus and assign BM1 train / val / test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ensure_artifact_dirs, load_config
from src.indexing.build_manifest import (
    append_log,
    assign_splits,
    scan_apk_rows,
    split_summary,
    write_index_csv,
    write_index_json,
    write_split_lists,
    year_label_split_summary,
    year_split_crosscheck,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build apk_index.csv with split assignments.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--apk-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max APKs after dedupe (smoke test)")
    parser.add_argument(
        "--no-shared",
        action="store_true",
        help="Force fresh scan even if Shared_pipeline_Files manifest exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    apk_root = args.apk_root or cfg.paths.apk_root
    shared_used = (
        not args.no_shared
        and cfg.paths.shared_manifest_csv is not None
        and cfg.paths.shared_manifest_csv.is_file()
    )
    rows, failed, dup_lines = scan_apk_rows(
        cfg,
        apk_root=apk_root,
        limit=args.limit,
        use_shared=not args.no_shared,
    )
    rows = assign_splits(cfg, rows)

    cross_errors = year_split_crosscheck(rows, cfg)
    if cross_errors:
        for msg in cross_errors[:10]:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    write_index_csv(cfg.paths.dataset_index, rows)
    write_index_json(cfg.paths.manifests_dir / "apk_index_summary.json", rows, cfg)
    write_split_lists(cfg.paths.splits_dir, rows)

    cfg.paths.failed_index_log.write_text("", encoding="utf-8")
    append_log(cfg.paths.failed_index_log, failed)
    append_log(cfg.paths.failed_index_log, dup_lines)

    summary = split_summary(rows)
    print(f"Indexed {len(rows)} APKs → {cfg.paths.dataset_index}")
    if shared_used:
        print(f"  source: shared manifest {cfg.paths.shared_manifest_csv}")
    else:
        print(f"  source: fresh scan under {apk_root}")

    print("\nBy split:")
    for split_name in ("train", "val", "test", "other"):
        if split_name not in summary:
            continue
        s = summary[split_name]
        print(
            f"  {split_name:5s}: total={s['total']:5d}  "
            f"benign={s['benign']:5d}  malware={s['malware']:5d}"
        )

    print("\nBy year / split:")
    for year, split, benign, malware, total in year_label_split_summary(rows):
        year_s = str(year) if year is not None else "?"
        print(
            f"  year={year_s} split={split:5s} total={total:5d}  "
            f"benign={benign:5d}  malware={malware:5d}"
        )

    if failed:
        print(f"\n  failures logged: {len(failed)} → {cfg.paths.failed_index_log}")
    if dup_lines:
        print(f"  duplicate sha256 skipped: {len(dup_lines)}")

    print("\nP1 exit criteria:")
    print("  [x] Row counts per year/label/split printed")
    print("  [x] No 2022/2023 APK marked train or val")
    print("  [x] Duplicate-hash report + failed_index.log produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
