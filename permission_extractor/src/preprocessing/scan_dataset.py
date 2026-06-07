"""Walk apk_root, assign splits, write dataset index.

Optional P1 metadata (apk_size_bytes, num_dex_files): use repo helper
Shared_pipeline_Files/tools/build_apk_manifest.py for centralized corpus stats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ensure_artifact_dirs, load_config
from src.preprocessing.common import (
    assign_splits,
    scan_apk_rows,
    split_counts,
    write_dataset_index,
    write_split_file,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan APK tree and build dataset index.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--apk-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    apk_root = Path(args.apk_root or cfg.paths.apk_root)
    rows = scan_apk_rows(cfg, apk_root, limit=args.limit)

    rows = assign_splits(cfg, rows)
    write_dataset_index(cfg.paths.dataset_index, rows)

    for split_name in ("train", "val", "dev_test", "temporal_holdout"):
        split_rows = [r for r in rows if r.split == split_name]
        write_split_file(cfg.paths.splits_dir / f"{split_name}.txt", split_rows)

    counts = split_counts(rows)
    print(f"Indexed {len(rows)} APKs → {cfg.paths.dataset_index}")
    for name in ("train", "val", "dev_test", "temporal_holdout"):
        print(f"  {name}: {counts.get(name, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
