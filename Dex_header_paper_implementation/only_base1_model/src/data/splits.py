"""Train/validation/test split helpers (temporal year holdout or random index split)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def year_from_apk_path(apk_path: Path | str) -> str | None:
    """Extract a 4-digit year folder (e.g. 2020) from an APK path."""
    match = re.search(r"/(20\d{2})/", str(apk_path).replace("\\", "/"))
    return match.group(1) if match else None


def temporal_three_way_split_indices(
    paths: list[str],
    labels: torch.Tensor | np.ndarray,
    *,
    train_years: list[int | str],
    test_years: list[int | str],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Temporal split with a stratified validation holdout from train years.

    train_years → pool split into train + val (e.g. 2020, 2021)
    test_years  → held-out test only (e.g. 2022, 2023); never used during training
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    train_set = {str(y) for y in train_years}
    test_set = {str(y) for y in test_years}
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"train_years and test_years overlap: {sorted(overlap)}")

    dev_idx: list[int] = []
    test_idx: list[int] = []
    unassigned: list[str] = []

    for i, path in enumerate(paths):
        year = year_from_apk_path(path)
        if year in train_set:
            dev_idx.append(i)
        elif year in test_set:
            test_idx.append(i)
        else:
            unassigned.append(path)

    if not dev_idx:
        raise ValueError(f"No samples matched train_years={sorted(train_set)}")
    if not test_idx:
        raise ValueError(f"No samples matched test_years={sorted(test_set)}")
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} sample(s) not in train_years or test_years "
            f"(example: {unassigned[0]})"
        )

    label_arr = np.asarray(labels).astype(int).ravel()
    dev_labels = label_arr[dev_idx]
    dev_indices = np.arange(len(dev_idx))
    train_local, val_local = train_test_split(
        dev_indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=dev_labels,
    )
    train_idx = [dev_idx[int(i)] for i in train_local]
    val_idx = [dev_idx[int(i)] for i in val_local]

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
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
