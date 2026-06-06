"""Load preprocessed permission shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShardRecord:
    apk_id: str
    path: str
    label: int
    shard_path: Path


def load_manifest(processed_dir: Path, split: str) -> list[ShardRecord]:
    manifest_path = processed_dir / f"manifest_{split}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[ShardRecord] = []
    for entry in entries:
        records.append(
            ShardRecord(
                apk_id=entry["apk_id"],
                path=entry["path"],
                label=int(entry["label"]),
                shard_path=(processed_dir / entry["shard"]).resolve(),
            )
        )
    return records


def load_shard_vector(shard_path: Path) -> tuple[np.ndarray, int]:
    data = np.load(shard_path)
    return data["p"].astype(np.float32), int(data["label"])
