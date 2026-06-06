"""Build permission vocabulary from train split only."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.config import load_config
from src.features.permission_vector import extract_permission_tokens, save_vocab
from src.preprocessing.common import read_dataset_index, rows_for_split

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build permission vocab from train APKs.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    rows = read_dataset_index(cfg.paths.dataset_index)
    train_rows = rows_for_split(rows, "train")
    if not train_rows:
        raise SystemExit("No train rows in dataset index; run scan_dataset.py first")

    counter: Counter[str] = Counter()
    failed = 0
    for row in train_rows:
        try:
            for token in extract_permission_tokens(row.apk_path):
                counter[token] += 1
        except Exception:
            failed += 1

    permissions = sorted(counter.keys())
    save_vocab(cfg.paths.permission_vocab, permissions)
    print(f"Vocabulary size M={len(permissions)} (train APKs={len(train_rows)}, failed={failed})")
    print(f"Saved → {cfg.paths.permission_vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
