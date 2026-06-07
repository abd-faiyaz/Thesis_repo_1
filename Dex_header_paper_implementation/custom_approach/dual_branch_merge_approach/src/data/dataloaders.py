"""DataLoader builders for DualBranchDataset."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.dataset import DualBranchDataset


def build_train_loader(
    dataset: DualBranchDataset,
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
    dataset: DualBranchDataset,
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


def build_dataloaders_from_manifests(
    train_manifest: Path | str,
    val_manifest: Path | str,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, int, int]:
    train_ds = DualBranchDataset.from_manifest(train_manifest)
    val_ds = DualBranchDataset.from_manifest(val_manifest)
    return (
        build_train_loader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        build_eval_loader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        train_ds.header_dim,
        train_ds.bow_dim,
    )


def build_dataloaders_from_config(
    cfg: PipelineConfig,
) -> tuple[DataLoader, DataLoader, int, int]:
    data = cfg.data
    return build_dataloaders_from_manifests(
        cfg.paths.manifest_train,
        cfg.paths.manifest_val,
        batch_size=int(data.get("batch_size", 16)),
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=bool(data.get("pin_memory", True)),
    )


def build_test_loader_from_config(
    cfg: PipelineConfig,
) -> tuple[DataLoader, int, int]:
    """Build a sequential test loader for the temporal holdout split."""
    data = cfg.data
    if not cfg.paths.manifest_test.is_file():
        raise FileNotFoundError(
            f"Test manifest not found: {cfg.paths.manifest_test}. "
            "Run preprocessing with temporal_year split and extract --split all."
        )
    test_ds = DualBranchDataset.from_manifest(cfg.paths.manifest_test)
    loader = build_eval_loader(
        test_ds,
        batch_size=int(data.get("batch_size", 16)),
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=bool(data.get("pin_memory", True)),
    )
    return loader, test_ds.header_dim, test_ds.bow_dim
