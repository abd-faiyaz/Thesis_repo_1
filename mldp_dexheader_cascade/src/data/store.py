"""Load P2 cascade feature shards from artifacts/processed/features_{split}.pt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import PipelineConfig


@dataclass(frozen=True)
class CascadeFeatureShard:
    """In-memory representation of a preprocessed cascade feature shard."""

    x_s: torch.Tensor
    h: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    paths: list[str]
    sha256: list[str]
    dims: dict[str, int]
    split: str
    source_path: Path


def feature_shard_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / f"features_{split}.pt"


def _dims_from_payload(payload: dict, x_s: torch.Tensor, h: torch.Tensor, x: torch.Tensor) -> dict[str, int]:
    raw = payload.get("feature_dims")
    if isinstance(raw, dict):
        return {
            "S": int(raw.get("S", x_s.shape[1])),
            "H": int(raw.get("H", h.shape[1])),
            "d": int(raw.get("d", x.shape[1])),
        }
    return {"S": int(x_s.shape[1]), "H": int(h.shape[1]), "d": int(x.shape[1])}


def load_feature_shard(path: Path | str, *, split: str = "") -> CascadeFeatureShard:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Feature shard not found: {source}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected shard payload type: {type(payload)!r}")

    required = ("x_S", "H", "x", "y")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Shard {source} missing keys: {missing}")

    x_s = payload["x_S"].float()
    h = payload["H"].float()
    x = payload["x"].float()
    y = payload["y"].long()
    paths = [str(p) for p in payload["paths"]]
    sha256 = [str(s) for s in payload.get("sha256", payload.get("sha256s", []))]
    dims = _dims_from_payload(payload, x_s, h, x)

    n = x.shape[0]
    if x_s.shape[0] != n or h.shape[0] != n or y.shape[0] != n:
        raise ValueError(f"Length mismatch in {source}")
    if len(paths) != n:
        raise ValueError(f"x/paths length mismatch in {source}: {n} vs {len(paths)}")

    return CascadeFeatureShard(
        x_s=x_s,
        h=h,
        x=x,
        y=y,
        paths=paths,
        sha256=sha256,
        dims=dims,
        split=split or source.stem.removeprefix("features_"),
        source_path=source,
    )


def load_split_shards(cfg: PipelineConfig) -> dict[str, CascadeFeatureShard]:
    shards: dict[str, CascadeFeatureShard] = {}
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
