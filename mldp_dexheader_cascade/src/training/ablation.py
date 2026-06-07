"""Separate-head ablation datasets and loaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import CascadeDataset
from src.data.dataloaders import build_eval_loader, build_train_loader
from src.data.store import CascadeFeatureShard

ABLATION_MODES = (
    "mode_a_fusion",
    "mldp_perms_only",
)


class XYDataset(Dataset):
    """Map CascadeDataset rows to (features, label)."""

    def __init__(self, base: CascadeDataset, *, feature: str) -> None:
        self.base = base
        self.feature = feature

    @property
    def input_dim(self) -> int:
        if self.feature == "x_S":
            return self.base.s_dim
        if self.feature == "x":
            return self.base.fused_dim
        raise ValueError(self.feature)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_s, _, x, y = self.base[index]
        if self.feature == "x_S":
            return x_s, y
        if self.feature == "x":
            return x, y
        raise ValueError(self.feature)


def build_ablation_dataset(shard: CascadeFeatureShard, *, mode: str) -> XYDataset:
    base = CascadeDataset.from_shard(shard)
    if mode == "mode_a_fusion":
        return XYDataset(base, feature="x")
    if mode == "mldp_perms_only":
        return XYDataset(base, feature="x_S")
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def build_ablation_loaders(
    train_shard: CascadeFeatureShard,
    val_shard: CascadeFeatureShard,
    *,
    mode: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, int]:
    train_ds = build_ablation_dataset(train_shard, mode=mode)
    val_ds = build_ablation_dataset(val_shard, mode=mode)
    input_dim = train_ds.input_dim

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
    shard: CascadeFeatureShard,
    *,
    mode: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = build_ablation_dataset(shard, mode=mode)
    return build_eval_loader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_val_test_shards(cfg) -> tuple[CascadeFeatureShard, CascadeFeatureShard]:
    from src.data.store import feature_shard_path, load_feature_shard

    val_path = feature_shard_path(cfg.paths.processed, "val")
    test_path = feature_shard_path(cfg.paths.processed, "test")
    if not val_path.is_file():
        raise FileNotFoundError(f"Val features not found: {val_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test features not found: {test_path}")
    return (
        load_feature_shard(val_path, split="val"),
        load_feature_shard(test_path, split="test"),
    )
