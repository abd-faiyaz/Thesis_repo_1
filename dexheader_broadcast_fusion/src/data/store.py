"""Load P2 fusion feature shards (separate H and R tensors)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import PipelineConfig


@dataclass(frozen=True)
class FusionFeatureShard:
    H: torch.Tensor
    R: torch.Tensor
    y: torch.Tensor
    paths: list[str]
    sha256: list[str]
    dex_dim: int
    receiver_dim: int
    split: str
    source_path: Path


def feature_shard_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / f"features_{split}.pt"


def load_feature_shard(path: Path | str, *, split: str = "") -> FusionFeatureShard:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Feature shard not found: {source}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected shard payload type: {type(payload)!r}")

    H = payload["H"].float()
    R = payload["R"].float()
    y = payload["y"].long()
    paths = [str(p) for p in payload["paths"]]
    sha256 = [str(h) for h in payload.get("sha256", payload.get("sha256s", []))]
    dex_dim = int(payload.get("dex_dim", H.shape[1]))
    receiver_dim = int(payload.get("receiver_dim", R.shape[1]))

    n = H.shape[0]
    if R.shape[0] != n or y.shape[0] != n:
        raise ValueError(f"Shard length mismatch in {source}")
    if len(paths) != n:
        raise ValueError(f"paths length mismatch in {source}")

    return FusionFeatureShard(
        H=H,
        R=R,
        y=y,
        paths=paths,
        sha256=sha256,
        dex_dim=dex_dim,
        receiver_dim=receiver_dim,
        split=split or source.stem.removeprefix("features_"),
        source_path=source,
    )


def load_split_shards(cfg: PipelineConfig) -> dict[str, FusionFeatureShard]:
    shards: dict[str, FusionFeatureShard] = {}
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
