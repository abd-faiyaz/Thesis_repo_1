"""Feature-slice datasets for MLDP-only and receiver-only ablations."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import HybridManifestDataset
from src.data.dataloaders import build_eval_loader, build_train_loader
from src.data.store import FeatureShard, feature_shard_path, load_feature_shard

ABLATION_MODES = (
    "full_fusion",
    "mldp_perms_only",
    "receiver_actions_only",
)


class FeatureSliceDataset(Dataset):
    """View a contiguous slice of fused feature vectors."""

    def __init__(
        self,
        base: HybridManifestDataset,
        *,
        start: int,
        end: int,
    ) -> None:
        self.base = base
        self.start = start
        self.end = end

    @property
    def feature_dim(self) -> int:
        return self.end - self.start

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[index]
        return x[self.start : self.end], y


def build_sliced_dataset(
    shard: FeatureShard,
    *,
    mode: str,
    s_size: int,
) -> Dataset:
    base = HybridManifestDataset.from_shard(shard)
    if mode == "full_fusion":
        return base
    if mode == "mldp_perms_only":
        return FeatureSliceDataset(base, start=0, end=s_size)
    if mode == "receiver_actions_only":
        return FeatureSliceDataset(base, start=s_size, end=base.feature_dim)
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def sliced_input_dim(full_dim: int, *, mode: str, s_size: int) -> int:
    if mode == "full_fusion":
        return full_dim
    if mode == "mldp_perms_only":
        return s_size
    if mode == "receiver_actions_only":
        return full_dim - s_size
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def build_ablation_loaders(
    train_shard: FeatureShard,
    val_shard: FeatureShard,
    *,
    mode: str,
    s_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, int]:
    train_ds = build_sliced_dataset(train_shard, mode=mode, s_size=s_size)
    val_ds = build_sliced_dataset(val_shard, mode=mode, s_size=s_size)
    input_dim = sliced_input_dim(
        HybridManifestDataset.from_shard(train_shard).feature_dim,
        mode=mode,
        s_size=s_size,
    )

    train_loader = build_train_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_eval_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, input_dim


def build_eval_loader_for_shard(
    shard: FeatureShard,
    *,
    mode: str,
    s_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = build_sliced_dataset(shard, mode=mode, s_size=s_size)
    return build_eval_loader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_test_val_shards(cfg) -> tuple[FeatureShard, FeatureShard]:
    """Load P2 feature shards only (no APK I/O)."""
    processed = cfg.paths.processed
    test_path = feature_shard_path(processed, "test")
    val_path = feature_shard_path(processed, "val")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test features not found: {test_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Val features not found: {val_path}")
    return (
        load_feature_shard(test_path, split="test"),
        load_feature_shard(val_path, split="val"),
    )

