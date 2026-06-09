"""Train/validation/test split helpers (temporal holdout or random index split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from shared_splits import temporal_holdout_split_indices, year_from_apk_path
from sklearn.model_selection import train_test_split

__all__ = [
    "temporal_holdout_split_indices",
    "temporal_three_way_split_indices",
    "write_split_path_files",
    "year_from_apk_path",
]


def temporal_three_way_split_indices(
    paths: list[str],
    labels: torch.Tensor | np.ndarray,
    *,
    train_years: list[int | str],
    test_years: list[int | str] | None = None,
    holdout_years: list[int | str] | None = None,
    val_fraction: float = 0.5,
    val_fraction_of_holdout: float | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Temporal holdout split (legacy name retained for callers).

    train_years → all training APKs (e.g. 2020, 2021)
    holdout_years / test_years → stratified val + test (disjoint, e.g. 2022, 2023)
    """
    holdout = holdout_years if holdout_years is not None else test_years
    val_frac = (
        val_fraction_of_holdout
        if val_fraction_of_holdout is not None
        else val_fraction
    )
    return temporal_holdout_split_indices(
        paths,
        labels,
        train_years=train_years,
        holdout_years=holdout,
        val_fraction_of_holdout=val_frac,
        seed=seed,
    )


def write_split_path_files(
    splits_dir: Path,
    paths: list[str],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor | None = None,
) -> None:
    """Write train.txt / val.txt / test.txt (one APK path per line) for reproducibility."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_paths = [paths[int(i)] for i in train_idx.tolist()]
    val_paths = [paths[int(i)] for i in val_idx.tolist()]
    (splits_dir / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_paths) + "\n", encoding="utf-8")
    if test_idx is not None:
        test_paths = [paths[int(i)] for i in test_idx.tolist()]
        (splits_dir / "test.txt").write_text("\n".join(test_paths) + "\n", encoding="utf-8")
