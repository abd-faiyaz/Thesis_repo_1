#!/usr/bin/env python3
"""Histogram Dex file counts per APK (Phase 7 multi-dex telemetry)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.store import load_shard_manifest
from src.features.apk_extract import ApkExtractError, read_all_dex_from_apk
from src.features.multidex import multidex_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count classes*.dex files per APK (train manifest by default)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which manifest to scan",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max APKs to scan")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: artifacts/dex_stats.json)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    md = multidex_settings(cfg.preprocessing)
    pattern = str(md["dex_pattern"])

    manifest_path = (
        cfg.paths.manifest_train if args.split == "train" else cfg.paths.manifest_val
    )
    manifest = load_shard_manifest(manifest_path)
    entries = manifest.entries
    if args.limit is not None:
        entries = entries[: args.limit]

    counts: Counter[int] = Counter()
    failed = 0
    for entry in tqdm(entries, desc=f"Dex count ({args.split})", unit="apk"):
        apk_path = Path(entry.apk_path)
        try:
            dex_list = read_all_dex_from_apk(apk_path, pattern=pattern)
            counts[len(dex_list)] += 1
        except (ApkExtractError, OSError) as exc:
            failed += 1
            if failed <= 5:
                print(f"  skip {apk_path}: {exc}", file=sys.stderr)

    histogram = {str(k): v for k, v in sorted(counts.items())}
    payload = {
        "split": args.split,
        "n_apks_scanned": len(entries),
        "n_failed": failed,
        "dex_pattern": pattern,
        "multidex_mode": md["mode"],
        "histogram_n_dex": histogram,
        "max_n_dex": max((int(k) for k in histogram), default=0),
        "pct_multi_dex": round(
            100.0
            * sum(v for k, v in counts.items() if k > 1)
            / max(len(entries) - failed, 1),
            2,
        ),
    }

    out_path = args.out or (cfg.paths.failed_apks_log.parent / "dex_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Dex stats ({args.split})")
    print(f"  scanned: {len(entries)}  failed: {failed}")
    print(f"  histogram (n_dex → count): {histogram}")
    print(f"  multi-dex APKs: {payload['pct_multi_dex']}%")
    print(f"  written → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
