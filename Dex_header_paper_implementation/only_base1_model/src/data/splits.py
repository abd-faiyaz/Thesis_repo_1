"""Train/validation split helpers (temporal year holdout or random index split)."""

from __future__ import annotations

import re
from pathlib import Path

import torch


def year_from_apk_path(apk_path: Path | str) -> str | None:
    """Extract a 4-digit year folder (e.g. 2020) from an APK path."""
    match = re.search(r"/(20\d{2})/", str(apk_path).replace("\\", "/"))
    return match.group(1) if match else None


def temporal_split_indices(
    paths: list[str],
    *,
    train_years: list[int | str],
    val_years: list[int | str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hold out samples by top-level year folder in each APK path.

    train_years → train indices (e.g. 2020, 2021)
    val_years   → validation/test indices (e.g. 2022, 2023)
    """
    train_set = {str(y) for y in train_years}
    val_set = {str(y) for y in val_years}
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"train_years and val_years overlap: {sorted(overlap)}")

    train_idx: list[int] = []
    val_idx: list[int] = []
    unassigned: list[str] = []

    for i, path in enumerate(paths):
        year = year_from_apk_path(path)
        if year in train_set:
            train_idx.append(i)
        elif year in val_set:
            val_idx.append(i)
        else:
            unassigned.append(path)

    if not train_idx:
        raise ValueError(f"No samples matched train_years={sorted(train_set)}")
    if not val_idx:
        raise ValueError(f"No samples matched val_years={sorted(val_set)}")
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} sample(s) not in train_years or val_years "
            f"(example: {unassigned[0]})"
        )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
    )


def write_split_path_files(
    splits_dir: Path,
    paths: list[str],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> None:
    """Write train.txt / val.txt (one APK path per line) for reproducibility."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_paths = [paths[int(i)] for i in train_idx.tolist()]
    val_paths = [paths[int(i)] for i in val_idx.tolist()]
    (splits_dir / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_paths) + "\n", encoding="utf-8")
