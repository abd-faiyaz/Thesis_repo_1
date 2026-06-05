"""Load preprocessed feature bundles from .pt or .npy artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ProcessedBundle:
    """In-memory representation of a preprocessed Dex header feature file."""

    features: torch.Tensor
    labels: torch.Tensor
    paths: list[str]
    feature_dim: int
    source_path: Path


def _load_npy_bundle(path: Path) -> ProcessedBundle:
    """Load a Phase 2 npy triplet (features, labels, paths) plus optional meta.json."""
    stem = path.with_suffix("")
    features_path = stem.with_suffix(".features.npy")
    labels_path = stem.with_suffix(".labels.npy")
    paths_path = stem.with_suffix(".paths.npy")
    meta_path = stem.with_suffix(".meta.json")

    if not features_path.is_file():
        raise FileNotFoundError(f"Processed features not found: {features_path}")

    features = torch.from_numpy(np.load(features_path)).float()
    labels = torch.from_numpy(np.load(labels_path)).float()
    paths_raw = np.load(paths_path, allow_pickle=True)
    paths = [str(p) for p in paths_raw.tolist()]

    feature_dim = int(features.shape[1])
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        feature_dim = int(meta.get("feature_dim", feature_dim))

    return ProcessedBundle(
        features=features,
        labels=labels,
        paths=paths,
        feature_dim=feature_dim,
        source_path=path,
    )


def load_processed_bundle(path: Path | str) -> ProcessedBundle:
    """Load a processed bundle from a .pt file or npy artifact stem."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed bundle not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pt":
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            features = payload["features"].float()
            labels = payload["labels"].float()
            paths = [str(p) for p in payload["paths"]]
            feature_dim = int(payload.get("feature_dim", features.shape[1]))
        else:
            raise ValueError(f"Unexpected .pt payload type: {type(payload)!r}")

        return ProcessedBundle(
            features=features,
            labels=labels,
            paths=paths,
            feature_dim=feature_dim,
            source_path=source,
        )

    if suffix in {".npy", ".json"}:
        return _load_npy_bundle(source)

    raise ValueError(f"Unsupported processed bundle format: {source}")
