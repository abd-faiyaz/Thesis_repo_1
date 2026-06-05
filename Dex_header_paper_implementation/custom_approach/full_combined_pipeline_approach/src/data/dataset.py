"""PyTorch Dataset over per-APK .npz feature shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.store import ShardEntry, ShardManifest, load_shard_manifest


class CombinedPipelineDataset(Dataset):
    def __init__(
        self,
        entries: list[ShardEntry],
        *,
        header_dim: int,
        bow_dim: int,
        indices: list[int] | None = None,
    ) -> None:
        if indices is not None:
            entries = [entries[i] for i in indices]
        self.entries = entries
        self.header_dim = header_dim
        self.bow_dim = bow_dim

    @classmethod
    def from_manifest(cls, manifest_path: Path | str) -> CombinedPipelineDataset:
        manifest = load_shard_manifest(manifest_path)
        return cls(
            manifest.entries,
            header_dim=manifest.header_dim,
            bow_dim=manifest.bow_dim,
        )

    @classmethod
    def from_shard_manifest(cls, manifest: ShardManifest) -> CombinedPipelineDataset:
        return cls(
            manifest.entries,
            header_dim=manifest.header_dim,
            bow_dim=manifest.bow_dim,
        )

    @property
    def combined_dim(self) -> int:
        return self.header_dim + self.bow_dim

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.entries[index]
        with np.load(entry.shard_path) as data:
            header = torch.from_numpy(np.asarray(data["header"], dtype=np.float32))
            bow = torch.from_numpy(np.asarray(data["bow"], dtype=np.float32))
            label = torch.tensor(int(data["label"]), dtype=torch.long)
        return header, bow, label
