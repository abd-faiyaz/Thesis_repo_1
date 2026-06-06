"""PyTorch Dataset over permission shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.store import ShardRecord, load_manifest, load_shard_vector


class PermissionDataset(Dataset):
    def __init__(self, processed_dir: Path, split: str) -> None:
        self.processed_dir = processed_dir
        self.split = split
        self.records: list[ShardRecord] = load_manifest(processed_dir, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        vec, label = load_shard_vector(record.shard_path)
        return torch.from_numpy(vec), torch.tensor(label, dtype=torch.float32)


def stack_split_arrays(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    records = load_manifest(processed_dir, split)
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        vec, label = load_shard_vector(record.shard_path)
        vectors.append(vec)
        labels.append(label)
    if not vectors:
        raise ValueError(f"No shards for split={split!r}")
    return np.stack(vectors, axis=0), np.asarray(labels, dtype=np.float64)
