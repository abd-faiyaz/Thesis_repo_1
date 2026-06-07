"""PyTorch Dataset for preprocessed hybrid manifest feature vectors."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.store import FeatureShard, load_feature_shard


class HybridManifestDataset(Dataset):
    """Dataset over P2 fused vectors x = [x_S || x_R] with binary labels."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        indices: torch.Tensor | list[int] | None = None,
    ) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y length mismatch: {x.shape[0]} vs {y.shape[0]}"
            )

        self.x = x.float()
        self.y = y.long()

        if indices is None:
            self.indices = torch.arange(x.shape[0], dtype=torch.long)
        else:
            self.indices = torch.as_tensor(indices, dtype=torch.long)

    @classmethod
    def from_shard(
        cls,
        shard: FeatureShard,
        *,
        indices: torch.Tensor | list[int] | None = None,
    ) -> HybridManifestDataset:
        return cls(shard.x, shard.y, indices=indices)

    @classmethod
    def from_processed_file(cls, path: Path | str) -> HybridManifestDataset:
        shard = load_feature_shard(path)
        return cls.from_shard(shard)

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[index])
        return self.x[sample_idx], self.y[sample_idx]
