"""Extract permission vectors to per-APK shard files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import ensure_artifact_dirs, load_config
from src.features.permission_vector import (
    build_binary_vector,
    extract_permission_tokens,
    load_vocab,
)
from src.preprocessing.common import read_dataset_index

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _shard_dir(processed: Path, split: str) -> Path:
    return processed / "shards" / split


def _processed_ids_path(processed: Path) -> Path:
    return processed / "processed_ids.txt"


def _load_processed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append_processed_id(path: Path, apk_id: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(apk_id + "\n")


def _write_manifest(processed: Path, split: str, entries: list[dict]) -> None:
    manifest_path = processed / f"manifest_{split}.json"
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def extract_split(
    cfg,
    *,
    split: str,
    permissions: list[str],
    token_to_index: dict[str, int],
    resume: bool = True,
) -> list[dict]:
    rows = [r for r in read_dataset_index(cfg.paths.dataset_index) if r.split == split]
    if not rows:
        return []

    processed = cfg.paths.processed
    shard_root = _shard_dir(processed, split)
    shard_root.mkdir(parents=True, exist_ok=True)
    processed_ids_path = _processed_ids_path(processed)
    done = _load_processed_ids(processed_ids_path) if resume else set()
    failed_log = cfg.paths.failed_apks_log
    vector_size = len(permissions)
    entries: list[dict] = []

    for row in tqdm(rows, desc=f"extract:{split}"):
        if row.apk_id in done:
            shard_path = shard_root / f"{row.apk_id}.npz"
            if shard_path.is_file():
                entries.append(
                    {
                        "apk_id": row.apk_id,
                        "path": str(row.apk_path),
                        "label": row.label,
                        "shard": str(shard_path.relative_to(processed)),
                    }
                )
            continue

        try:
            tokens = extract_permission_tokens(row.apk_path)
            vec = build_binary_vector(tokens, token_to_index, vector_size=vector_size)
        except Exception as exc:
            with failed_log.open("a", encoding="utf-8") as f:
                f.write(f"{row.apk_path}\t{exc}\n")
            continue

        shard_path = shard_root / f"{row.apk_id}.npz"
        # np.savez_compressed always appends ".npz" — use ".tmp.npz" not ".npz.tmp".
        tmp_path = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            p=vec.astype(np.float32),
            label=np.int64(row.label),
            apk_id=row.apk_id,
        )
        tmp_path.replace(shard_path)
        _append_processed_id(processed_ids_path, row.apk_id)
        entries.append(
            {
                "apk_id": row.apk_id,
                "path": str(row.apk_path),
                "label": row.label,
                "shard": str(shard_path.relative_to(processed)),
            }
        )

    _write_manifest(processed, split, entries)
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract permission shards for all splits.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    if not cfg.paths.permission_vocab.is_file():
        raise SystemExit(f"Missing vocab: {cfg.paths.permission_vocab}; run build_permission_vocab.py")

    permissions, token_to_index = load_vocab(cfg.paths.permission_vocab)
    for split in args.splits:
        entries = extract_split(
            cfg,
            split=split,
            permissions=permissions,
            token_to_index=token_to_index,
            resume=not args.no_resume,
        )
        print(f"{split}: {len(entries)} shards")

    meta = {
        "M": len(permissions),
        "splits": args.splits,
        "vocab": str(cfg.paths.permission_vocab.resolve()),
    }
    meta_path = cfg.paths.processed / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Meta → {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
