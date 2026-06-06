"""Load pruned permission shards."""

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
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        ShardRecord(
            apk_id=e["apk_id"],
            path=e["path"],
            label=int(e["label"]),
            shard_path=(processed_dir / e["shard"]).resolve(),
        )
        for e in entries
    ]


def load_shard_vector(shard_path: Path) -> tuple[np.ndarray, int]:
    data = np.load(shard_path)
    return data["x"].astype(np.float32), int(data["label"])
