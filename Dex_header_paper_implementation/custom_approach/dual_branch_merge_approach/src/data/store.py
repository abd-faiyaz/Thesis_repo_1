"""Load shard manifest JSON written by preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShardEntry:
    apk_id: str
    shard_path: Path
    apk_path: Path
    label: int


@dataclass(frozen=True)
class ShardManifest:
    path: Path
    header_dim: int
    bow_dim: int
    entries: list[ShardEntry]
    multidex_mode: str | None = None


def load_shard_manifest(path: Path | str) -> ShardManifest:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    entries = [
        ShardEntry(
            apk_id=str(entry["apk_id"]),
            shard_path=Path(str(entry["shard_path"])),
            apk_path=Path(str(entry["apk_path"])),
            label=int(entry["label"]),
        )
        for entry in payload["entries"]
    ]

    return ShardManifest(
        path=manifest_path,
        header_dim=int(payload["header_dim"]),
        bow_dim=int(payload["bow_dim"]),
        entries=entries,
        multidex_mode=payload.get("multidex_mode"),
    )
