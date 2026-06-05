"""DataLoader builders for CombinedPipelineDataset."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.dataset import CombinedPipelineDataset


def build_train_loader(
    dataset: CombinedPipelineDataset,
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
    dataset: CombinedPipelineDataset,
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
    train_ds = CombinedPipelineDataset.from_manifest(train_manifest)
    val_ds = CombinedPipelineDataset.from_manifest(val_manifest)
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
