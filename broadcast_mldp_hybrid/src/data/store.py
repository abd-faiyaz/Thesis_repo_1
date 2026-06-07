"""Load P2 feature shards from artifacts/processed/features_{split}.pt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import PipelineConfig


@dataclass(frozen=True)
class FeatureShard:
    """In-memory representation of a preprocessed manifest feature shard."""

    x: torch.Tensor
    y: torch.Tensor
    paths: list[str]
    sha256: list[str]
    feature_dim: int
    split: str
    source_path: Path


def feature_shard_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / f"features_{split}.pt"


def load_feature_shard(path: Path | str, *, split: str = "") -> FeatureShard:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Feature shard not found: {source}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected shard payload type: {type(payload)!r}")

    x = payload["x"].float()
    y = payload["y"].long()
    paths = [str(p) for p in payload["paths"]]
    sha256 = [str(h) for h in payload.get("sha256", payload.get("sha256s", []))]
    feature_dim = int(payload.get("feature_dim", x.shape[1]))

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x/y length mismatch in {source}: {x.shape[0]} vs {y.shape[0]}"
        )
    if len(paths) != x.shape[0]:
        raise ValueError(
            f"x/paths length mismatch in {source}: {x.shape[0]} vs {len(paths)}"
        )

    return FeatureShard(
        x=x,
        y=y,
        paths=paths,
        sha256=sha256,
        feature_dim=feature_dim,
        split=split or source.stem.removeprefix("features_"),
        source_path=source,
    )


def load_split_shards(cfg: PipelineConfig) -> dict[str, FeatureShard]:
    """Load train/val/test shards written by preprocess_apks.py."""
    shards: dict[str, FeatureShard] = {}
    for split in ("train", "val", "test"):
        path = feature_shard_path(cfg.paths.processed, split)
        if path.is_file():
            shards[split] = load_feature_shard(path, split=split)
    if "train" not in shards:
        raise FileNotFoundError(
            f"Missing train shard: {feature_shard_path(cfg.paths.processed, 'train')}"
        )
    return shards


def load_preprocessing_meta(processed_dir: Path) -> dict:
    meta_path = processed_dir / "preprocessing_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
