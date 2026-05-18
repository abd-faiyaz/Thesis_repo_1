"""Build manifest vocabulary from the train split only."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.config import ensure_artifact_dirs, load_config
from src.features.manifest_bow import (
    ManifestBoWError,
    build_lexicon_from_counts,
    extract_manifest_tokens,
    save_vocab,
)
from src.preprocessing.common import (
    log_failure,
    read_dataset_index,
    read_split_ids,
    rows_for_split,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build manifest BoW lexicon (train split).")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    pre = cfg.preprocessing

    all_rows = read_dataset_index(cfg.paths.dataset_index)
    train_ids = read_split_ids(cfg.paths.splits_dir / "train.txt")
    train_rows = rows_for_split(all_rows, train_ids)

    counts: Counter[str] = Counter()
    failed = 0
    for row in tqdm(train_rows, desc="Collecting manifest tokens", unit="apk"):
        try:
            tokens = extract_manifest_tokens(row.apk_path)
        except ManifestBoWError as exc:
            failed += 1
            log_failure(cfg.paths.failed_apks_log, row.apk_path, str(exc))
            continue
        counts.update(tokens)

    if not counts:
        raise RuntimeError("No manifest tokens collected; see failed_apks.log")

    lexicon_size = int(pre.get("lexicon_size", 4380))
    min_freq = int(pre.get("min_token_freq", 2))
    token_to_index, unk_index = build_lexicon_from_counts(
        counts,
        lexicon_size=lexicon_size,
        min_token_freq=min_freq,
    )
    save_vocab(
        cfg.paths.vocab,
        token_to_index,
        lexicon_size=lexicon_size,
        unk_index=unk_index,
        min_token_freq=min_freq,
        extra={"num_train_apks": len(train_rows), "token_failures": failed},
    )

    print(f"Vocab size: {len(token_to_index)} (+ UNK @ {unk_index})")
    print(f"Saved → {cfg.paths.vocab}")
    if failed:
        print(f"  {failed} train APK(s) skipped (logged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
