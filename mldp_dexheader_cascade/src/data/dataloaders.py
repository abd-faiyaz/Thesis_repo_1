"""DataLoader builders for CascadeDataset."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.dataset import CascadeDataset
from src.data.store import CascadeFeatureShard, load_split_shards


def split_class_balance(y: torch.Tensor) -> dict[str, int | float]:
    labels = y.long()
    n = int(labels.numel())
    malware = int((labels == 1).sum().item())
    benign = n - malware
    return {
        "total": n,
        "benign": benign,
        "malware": malware,
        "malware_fraction": (malware / n) if n else 0.0,
    }


def compute_pos_weight(y: torch.Tensor) -> float:
    """BCE pos_weight = N_neg / N_pos on the train split."""
    stats = split_class_balance(y)
    pos = int(stats["malware"])
    neg = int(stats["benign"])
    if pos == 0:
        raise ValueError("Cannot compute pos_weight: no positive (malware) samples")
    return float(neg / pos)


def print_split_balance(shards: Mapping[str, CascadeFeatureShard]) -> dict[str, dict[str, int | float]]:
    stats: dict[str, dict[str, int | float]] = {}
    print("Split class balance:")
    for split in ("train", "val", "test"):
        shard = shards.get(split)
        if shard is None:
            continue
        split_stats = split_class_balance(shard.y)
        stats[split] = split_stats
        print(
            f"  {split:5s}: n={split_stats['total']:5d}  "
            f"benign={split_stats['benign']:5d}  malware={split_stats['malware']:5d}  "
            f"malware_frac={split_stats['malware_fraction']:.3f}"
        )
    if "train" in shards:
        pos_weight = compute_pos_weight(shards["train"].y)
        print(f"  pos_weight (N_neg/N_pos on train): {pos_weight:.4f}")
        stats["pos_weight"] = {"value": pos_weight}
    return stats


def _loader_settings(cfg: PipelineConfig) -> tuple[int, int, bool]:
    training = cfg.training
    data = cfg.raw.get("data", {})
    batch_size = int(data.get("batch_size", training.get("batch_size", 256)))
    num_workers = int(data.get("num_workers", 4))
    pin_memory = bool(data.get("pin_memory", torch.cuda.is_available()))
    return batch_size, num_workers, pin_memory


def build_train_loader(
    dataset: CascadeDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_eval_loader(
    dataset: CascadeDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataloaders(
    cfg: PipelineConfig,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    dict[str, int],
    dict[str, dict[str, int | float]],
]:
    """
    Build train, val, and test loaders from P2 feature shards.

    Returns (train_loader, val_loader, test_loader, feature_dims, balance_stats).
    """
    shards = load_split_shards(cfg)
    balance_stats = print_split_balance(shards)

    batch_size, num_workers, pin_memory = _loader_settings(cfg)

    train_ds = CascadeDataset.from_shard(shards["train"])
    feature_dims = dict(shards["train"].dims)

    val_shard = shards.get("val")
    if val_shard is None:
        raise FileNotFoundError(
            f"Missing val shard under {cfg.paths.processed}; re-run P2 preprocess"
        )
    val_ds = CascadeDataset.from_shard(val_shard)

    test_shard = shards.get("test")
    if test_shard is None:
        raise FileNotFoundError(
            f"Missing test shard under {cfg.paths.processed}; re-run P2 preprocess"
        )
    test_ds = CascadeDataset.from_shard(test_shard)

    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if ds.s_dim != train_ds.s_dim or ds.h_dim != train_ds.h_dim or ds.fused_dim != train_ds.fused_dim:
            raise ValueError(
                f"{name} dims S={ds.s_dim} H={ds.h_dim} d={ds.fused_dim} "
                f"!= train S={train_ds.s_dim} H={train_ds.h_dim} d={train_ds.fused_dim}"
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
    test_loader = build_eval_loader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, feature_dims, balance_stats


def build_dataloaders_from_config(
    cfg: PipelineConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_loader, val_loader, test_loader, feature_dims, _ = build_dataloaders(cfg)
    return train_loader, val_loader, test_loader, feature_dims
