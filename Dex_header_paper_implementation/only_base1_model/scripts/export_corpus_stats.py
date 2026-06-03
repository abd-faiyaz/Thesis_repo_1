#!/usr/bin/env python3
"""Export label / dex-count stats from dex_header_features.pt to JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import resolve_processed_path
from src.data.store import load_processed_bundle
from src.training.run_logging import archive_run_dir, metrics_dir, write_json


def _year_from_path(path: str) -> str | None:
    match = re.search(r"/(20\d{2})/", path.replace("\\", "/"))
    return match.group(1) if match else None


def export_corpus_stats(cfg, *, processed_path: Path | None = None) -> dict:
    path = processed_path or resolve_processed_path(cfg)
    bundle = load_processed_bundle(path)
    labels = bundle.labels.int()
    label_dist = {
        "total": int(labels.numel()),
        "benign": int((labels == 0).sum().item()),
        "malware": int((labels == 1).sum().item()),
    }

    dex_counts: dict[str, int] = {}
    pt_meta = __import__("torch").load(path, map_location="cpu", weights_only=False)
    if isinstance(pt_meta, dict) and pt_meta.get("dex_file_counts"):
        raw = pt_meta["dex_file_counts"]
        dex_counts = {str(k): int(v) for k, v in raw.items()}

    year_counts: Counter[str] = Counter()
    for p in bundle.paths:
        year = _year_from_path(p)
        if year:
            year_counts[year] += 1

    payload = {
        "source": str(path),
        "label_distribution": label_dist,
        "dex_file_counts": dex_counts,
        "year_folder_counts": dict(sorted(year_counts.items())),
    }

    out = metrics_dir(cfg) / "corpus_stats.json"
    write_json(out, {"timestamp": datetime.now(timezone.utc).isoformat(), **payload})

    archive = archive_run_dir(cfg)
    if archive is not None:
        write_json(archive / "corpus_stats" / "label_distribution.json", label_dist)
        write_json(archive / "corpus_stats" / "dex_file_counts.json", dex_counts)
        write_json(archive / "corpus_stats" / "year_folder_counts.json", dict(sorted(year_counts.items())))

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BM1 corpus statistics JSON.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    stats = export_corpus_stats(cfg)
    print(json.dumps(stats, indent=2))
    print(f"Wrote corpus stats under {metrics_dir(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
