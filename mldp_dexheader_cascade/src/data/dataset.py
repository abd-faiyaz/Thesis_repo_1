"""PyTorch Dataset for P2 cascade tensors (x_S, H, x, y)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.store import CascadeFeatureShard, load_feature_shard


class CascadeDataset(Dataset):
    """Dataset over MLDP bits x_S, dex header H, fused x, and binary labels."""

    def __init__(
        self,
        x_s: torch.Tensor,
        h: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        indices: torch.Tensor | list[int] | None = None,
    ) -> None:
        n = x.shape[0]
        for name, tensor in (("x_S", x_s), ("H", h), ("y", y)):
            if tensor.shape[0] != n:
                raise ValueError(f"{name} and x length mismatch: {tensor.shape[0]} vs {n}")

        self.x_s = x_s.float()
        self.h = h.float()
        self.x = x.float()
        self.y = y.long()

        if indices is None:
            self.indices = torch.arange(n, dtype=torch.long)
        else:
            self.indices = torch.as_tensor(indices, dtype=torch.long)

    @classmethod
    def from_shard(
        cls,
        shard: CascadeFeatureShard,
        *,
        indices: torch.Tensor | list[int] | None = None,
    ) -> CascadeDataset:
        return cls(shard.x_s, shard.h, shard.x, shard.y, indices=indices)

    @classmethod
    def from_processed_file(cls, path: Path | str) -> CascadeDataset:
        shard = load_feature_shard(path)
        return cls.from_shard(shard)

    @property
    def s_dim(self) -> int:
        return int(self.x_s.shape[1])

    @property
    def h_dim(self) -> int:
        return int(self.h.shape[1])

    @property
    def fused_dim(self) -> int:
        return int(self.x.shape[1])

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[index])
        return (
            self.x_s[sample_idx],
            self.h[sample_idx],
            self.x[sample_idx],
            self.y[sample_idx],
        )
