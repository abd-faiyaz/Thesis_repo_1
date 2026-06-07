"""Train/validation DataLoaders for preprocessed Dex header features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.dataset import DexDataset
from src.data.splits import temporal_three_way_split_indices, write_split_path_files
from src.data.store import ProcessedBundle, load_processed_bundle


def resolve_processed_path(cfg: PipelineConfig) -> Path:
    """Return the configured aggregate processed feature file path."""
    pre = cfg.preprocessing
    filename = str(pre.get("aggregate_filename", "dex_header_features.pt"))
    return cfg.paths.processed_dir / filename


def split_train_val_indices(
    num_samples: int,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random permutation split into train and validation index tensors."""
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator)

    val_size = max(1, int(round(num_samples * val_fraction)))
    if val_size >= num_samples:
        val_size = num_samples - 1

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if train_idx.numel() == 0:
        train_idx = perm[:-1]
        val_idx = perm[-1:]
    return train_idx, val_idx


def build_train_loader(
    dataset: DexDataset,
    *,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_eval_loader(
    dataset: DexDataset,
    *,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def resolve_split_settings(cfg: PipelineConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """Merge split settings from config (preprocessing overrides data for year lists)."""
    pre: dict[str, Any] = {}
    data: dict[str, Any] = {}
    splits_dir: Path | None = None
    if cfg is not None:
        pre = cfg.preprocessing
        data = cfg.data
        raw_splits = cfg.raw.get("paths", {}).get("splits_dir")
        if raw_splits:
            splits_dir = Path(raw_splits)
            if not splits_dir.is_absolute():
                splits_dir = (cfg.root / splits_dir).resolve()

    split_mode = overrides.get(
        "split_mode",
        pre.get("split_mode", data.get("split_mode", "stratified_random")),
    )
    splits_dir_raw = overrides.get("splits_dir", splits_dir)
    splits_dir_str: str | None
    if splits_dir_raw is None:
        splits_dir_str = None
    else:
        splits_dir_str = str(splits_dir_raw)

    test_years = overrides.get("test_years", pre.get("test_years", pre.get("val_years", [2022, 2023])))
    val_fraction = overrides.get(
        "val_fraction",
        pre.get("val_fraction", data.get("val_fraction", 0.1)),
    )

    return {
        "split_mode": str(split_mode),
        "train_years": overrides.get("train_years", pre.get("train_years", [2020, 2021])),
        "test_years": test_years,
        "val_fraction": float(val_fraction),
        "seed": int(overrides.get("seed", data.get("random_seed", 42))),
        "splits_dir": splits_dir_str,
    }


def resolve_split_indices(
    bundle: ProcessedBundle,
    *,
    split_mode: str = "stratified_random",
    train_years: list[int | str] | None = None,
    test_years: list[int | str] | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if split_mode == "temporal_year":
        train_idx, val_idx, test_idx = temporal_three_way_split_indices(
            bundle.paths,
            bundle.labels,
            train_years=train_years or [2020, 2021],
            test_years=test_years or [2022, 2023],
            val_fraction=val_fraction,
            seed=seed,
        )
        return train_idx, val_idx, test_idx
    if split_mode == "stratified_random":
        train_idx, val_idx = split_train_val_indices(
            bundle.features.shape[0],
            val_fraction=val_fraction,
            seed=seed,
        )
        return train_idx, val_idx, None
    raise ValueError(
        f"Unknown split_mode={split_mode!r}; use 'temporal_year' or 'stratified_random'"
    )


def resolve_train_val_indices(
    bundle: ProcessedBundle,
    *,
    split_mode: str = "stratified_random",
    train_years: list[int | str] | None = None,
    test_years: list[int | str] | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_idx, val_idx, _ = resolve_split_indices(
        bundle,
        split_mode=split_mode,
        train_years=train_years,
        test_years=test_years,
        val_fraction=val_fraction,
        seed=seed,
    )
    return train_idx, val_idx


def build_dataloaders_from_bundle(
    bundle: ProcessedBundle,
    *,
    split_mode: str = "stratified_random",
    train_years: list[int | str] | None = None,
    test_years: list[int | str] | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    splits_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """Build shuffled train and sequential val loaders from an in-memory bundle."""
    train_idx, val_idx, test_idx = resolve_split_indices(
        bundle,
        split_mode=split_mode,
        train_years=train_years,
        test_years=test_years,
        val_fraction=val_fraction,
        seed=seed,
    )
    if splits_dir is not None:
        write_split_path_files(splits_dir, bundle.paths, train_idx, val_idx, test_idx)

    train_ds = DexDataset.from_bundle(bundle, indices=train_idx)
    val_ds = DexDataset.from_bundle(bundle, indices=val_idx)

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
    return train_loader, val_loader, bundle.feature_dim


def build_dataloaders_from_config(
    cfg: PipelineConfig,
) -> tuple[DataLoader, DataLoader, int]:
    """Load processed artifacts from config and build train/val DataLoaders."""
    processed_path = resolve_processed_path(cfg)
    bundle = load_processed_bundle(processed_path)

    split = resolve_split_settings(cfg)
    splits_dir: Path | None = None
    if split["splits_dir"]:
        splits_dir = Path(split["splits_dir"])
    data_cfg = cfg.data
    if split["split_mode"] == "temporal_year":
        print(
            f"Temporal year split: train_years={split['train_years']} "
            f"test_years={split['test_years']} val_fraction={split['val_fraction']}"
        )
    else:
        print(
            f"Random split: val_fraction={split['val_fraction']} seed={split['seed']}"
        )

    return build_dataloaders_from_bundle(
        bundle,
        split_mode=split["split_mode"],
        train_years=split["train_years"],
        test_years=split["test_years"],
        val_fraction=split["val_fraction"],
        seed=split["seed"],
        batch_size=int(data_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        splits_dir=splits_dir,
    )


def build_test_loader_from_config(
    cfg: PipelineConfig,
) -> tuple[DataLoader, int]:
    """Build a sequential test loader for the temporal holdout years."""
    processed_path = resolve_processed_path(cfg)
    bundle = load_processed_bundle(processed_path)
    split = resolve_split_settings(cfg)

    if split["split_mode"] != "temporal_year":
        raise ValueError("Test split is only defined for split_mode='temporal_year'")

    _, _, test_idx = resolve_split_indices(
        bundle,
        split_mode=split["split_mode"],
        train_years=split["train_years"],
        test_years=split["test_years"],
        val_fraction=split["val_fraction"],
        seed=split["seed"],
    )
    if test_idx is None:
        raise ValueError("No test indices available for this split configuration")

    test_ds = DexDataset.from_bundle(bundle, indices=test_idx)
    data_cfg = cfg.data
    loader = build_eval_loader(
        test_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    return loader, bundle.feature_dim
