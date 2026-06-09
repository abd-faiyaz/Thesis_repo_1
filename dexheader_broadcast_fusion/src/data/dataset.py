"""FusionDataset — separate dex header H and receiver R inputs."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.data.store import FusionFeatureShard


class FusionDataset(Dataset):
    def __init__(self, H: torch.Tensor, R: torch.Tensor, y: torch.Tensor) -> None:
        if H.shape[0] != R.shape[0] or H.shape[0] != y.shape[0]:
            raise ValueError("H/R/y length mismatch")
        self.H = H.float()
        self.R = R.float()
        self.y = y.long()

    @property
    def dex_dim(self) -> int:
        return int(self.H.shape[1])

    @property
    def receiver_dim(self) -> int:
        return int(self.R.shape[1])

    def __len__(self) -> int:
        return int(self.H.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.H[index], self.R[index], self.y[index]

    @classmethod
    def from_shard(cls, shard: FusionFeatureShard) -> FusionDataset:
        return cls(shard.H, shard.R, shard.y)
