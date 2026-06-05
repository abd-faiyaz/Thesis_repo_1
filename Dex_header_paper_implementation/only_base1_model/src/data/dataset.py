"""PyTorch Dataset for preprocessed Dex header feature vectors."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.store import ProcessedBundle, load_processed_bundle


class DexDataset(Dataset):
    """Dataset over preprocessed `[N, feature_dim]` Dex header tensors."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        indices: torch.Tensor | list[int] | None = None,
    ) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features and labels length mismatch: {features.shape[0]} vs {labels.shape[0]}"
            )

        self.features = features
        self.labels = labels.float()

        if indices is None:
            self.indices = torch.arange(features.shape[0], dtype=torch.long)
        else:
            self.indices = torch.as_tensor(indices, dtype=torch.long)

    @classmethod
    def from_bundle(
        cls,
        bundle: ProcessedBundle,
        *,
        indices: torch.Tensor | list[int] | None = None,
    ) -> DexDataset:
        return cls(bundle.features, bundle.labels, indices=indices)

    @classmethod
    def from_processed_file(cls, path: Path | str) -> DexDataset:
        bundle = load_processed_bundle(path)
        return cls.from_bundle(bundle)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[index])
        x = self.features[sample_idx]
        y = self.labels[sample_idx]
        return x, y
