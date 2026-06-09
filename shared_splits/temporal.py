"""Temporal split: train on train_years; val+test from holdout_years (disjoint)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
import torch
from sklearn.model_selection import train_test_split

DEFAULT_TRAIN_YEARS = (2020, 2021)
DEFAULT_HOLDOUT_YEARS = (2022, 2023)
DEFAULT_VAL_FRACTION_OF_HOLDOUT = 0.5
DEFAULT_RANDOM_SEED = 42

T = TypeVar("T")


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_years: tuple[int, ...]
    holdout_years: tuple[int, ...]
    val_fraction_of_holdout: float
    random_seed: int
    split_mode: str = "temporal_holdout"

    @property
    def train_year_set(self) -> set[str]:
        return {str(y) for y in self.train_years}

    @property
    def holdout_year_set(self) -> set[str]:
        return {str(y) for y in self.holdout_years}


def year_from_apk_path(apk_path: Path | str) -> str | None:
    """Extract a 4-digit year folder (e.g. 2020) from an APK path."""
    match = re.search(r"/(20\d{2})/", str(apk_path).replace("\\", "/"))
    return match.group(1) if match else None


def _year_key(year: int | str | None) -> str | None:
    if year is None:
        return None
    return str(year)


def _normalize_year_list(values: Iterable[int | str] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return default
    return tuple(int(y) for y in values)


def resolve_split_config(raw: dict[str, Any] | None) -> TemporalSplitConfig:
    """Resolve split settings from a `splits` or `preprocessing` dict."""
    cfg = raw or {}
    train_years = _normalize_year_list(cfg.get("train_years"), DEFAULT_TRAIN_YEARS)
    holdout_raw = (
        cfg.get("holdout_years")
        or cfg.get("test_years")
        or cfg.get("temporal_holdout_years")
    )
    holdout_years = _normalize_year_list(holdout_raw, DEFAULT_HOLDOUT_YEARS)

    val_fraction = cfg.get("val_fraction_of_holdout")
    if val_fraction is None:
        val_fraction = cfg.get("val_fraction", DEFAULT_VAL_FRACTION_OF_HOLDOUT)
    val_fraction = float(val_fraction)

    split_mode = str(cfg.get("split_mode", "temporal_holdout"))
    if split_mode == "temporal_year":
        split_mode = "temporal_holdout"
    if split_mode == "stratified_development":
        split_mode = "temporal_holdout"

    seed = int(cfg.get("random_seed", cfg.get("seed", DEFAULT_RANDOM_SEED)))
    return TemporalSplitConfig(
        train_years=train_years,
        holdout_years=holdout_years,
        val_fraction_of_holdout=val_fraction,
        random_seed=seed,
        split_mode=split_mode,
    )


def temporal_holdout_partition(
    items: Sequence[T],
    labels: Sequence[int],
    *,
    get_year: Callable[[T], int | str | None],
    train_years: Sequence[int | str] | None = None,
    holdout_years: Sequence[int | str] | None = None,
    val_fraction_of_holdout: float = DEFAULT_VAL_FRACTION_OF_HOLDOUT,
    seed: int = DEFAULT_RANDOM_SEED,
    reject_unassigned: bool = True,
) -> tuple[list[T], list[T], list[T], list[T]]:
    """
    Partition items into train / val / test / other.

    train  → all items from train_years
    val, test → stratified split of holdout_years only (disjoint)
    """
    if not 0.0 < val_fraction_of_holdout < 1.0:
        raise ValueError(
            f"val_fraction_of_holdout must be in (0, 1), got {val_fraction_of_holdout}"
        )

    train_set = {str(y) for y in (train_years or DEFAULT_TRAIN_YEARS)}
    holdout_set = {str(y) for y in (holdout_years or DEFAULT_HOLDOUT_YEARS)}
    overlap = train_set & holdout_set
    if overlap:
        raise ValueError(f"train_years and holdout_years overlap: {sorted(overlap)}")

    if len(items) != len(labels):
        raise ValueError(f"items/labels length mismatch: {len(items)} / {len(labels)}")

    train_items: list[T] = []
    holdout_items: list[T] = []
    holdout_labels: list[int] = []
    other_items: list[T] = []
    unassigned: list[T] = []

    label_arr = np.asarray(labels).astype(int).ravel()
    for i, item in enumerate(items):
        year = _year_key(get_year(item))
        if year in train_set:
            train_items.append(item)
        elif year in holdout_set:
            holdout_items.append(item)
            holdout_labels.append(int(label_arr[i]))
        elif year is None:
            unassigned.append(item)
        else:
            other_items.append(item)

    if not train_items:
        raise ValueError(f"No items matched train_years={sorted(train_set)}")
    if not holdout_items:
        raise ValueError(f"No items matched holdout_years={sorted(holdout_set)}")
    if reject_unassigned and unassigned:
        sample = unassigned[0]
        raise ValueError(
            f"{len(unassigned)} item(s) missing a year folder "
            f"(example: {sample})"
        )
    if reject_unassigned and other_items:
        sample = other_items[0]
        raise ValueError(
            f"{len(other_items)} item(s) not in train_years or holdout_years "
            f"(example: {sample})"
        )

    holdout_labels_arr = np.asarray(holdout_labels, dtype=int)
    class_counts = np.bincount(holdout_labels_arr) if holdout_labels_arr.size else np.array([])
    stratify = holdout_labels_arr if class_counts.size and np.min(class_counts[class_counts > 0]) >= 2 else None
    val_items, test_items = train_test_split(
        holdout_items,
        test_size=1.0 - val_fraction_of_holdout,
        stratify=stratify,
        random_state=seed,
    )

    val_keys = {id(x) for x in val_items}
    test_keys = {id(x) for x in test_items}
    if val_keys & test_keys:
        raise RuntimeError("val/test overlap detected after stratified split")

    return train_items, list(val_items), list(test_items), other_items + unassigned


def temporal_holdout_split_indices(
    paths: list[str],
    labels: torch.Tensor | np.ndarray,
    *,
    train_years: Sequence[int | str] | None = None,
    holdout_years: Sequence[int | str] | None = None,
    val_fraction_of_holdout: float = DEFAULT_VAL_FRACTION_OF_HOLDOUT,
    seed: int = DEFAULT_RANDOM_SEED,
    reject_unassigned: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Index-based train/val/test split for in-memory feature bundles."""
    label_list = np.asarray(labels).astype(int).ravel().tolist()
    indices = list(range(len(paths)))

    train_idx_items, val_idx_items, test_idx_items, _ = temporal_holdout_partition(
        indices,
        label_list,
        get_year=lambda i: year_from_apk_path(paths[i]),
        train_years=train_years,
        holdout_years=holdout_years,
        val_fraction_of_holdout=val_fraction_of_holdout,
        seed=seed,
        reject_unassigned=reject_unassigned,
    )

    return (
        torch.tensor(train_idx_items, dtype=torch.long),
        torch.tensor(val_idx_items, dtype=torch.long),
        torch.tensor(test_idx_items, dtype=torch.long),
    )


def crosscheck_temporal_holdout(
    rows: Sequence[Any],
    *,
    get_split: Callable[[Any], str],
    get_year: Callable[[Any], int | str | None],
    train_years: Sequence[int | str] | None = None,
    holdout_years: Sequence[int | str] | None = None,
    get_path: Callable[[Any], str] | None = None,
) -> list[str]:
    """Return error messages when split assignments violate the temporal policy."""
    train_set = {str(y) for y in (train_years or DEFAULT_TRAIN_YEARS)}
    holdout_set = {str(y) for y in (holdout_years or DEFAULT_HOLDOUT_YEARS)}
    errors: list[str] = []
    for row in rows:
        split = get_split(row)
        year = _year_key(get_year(row))
        path = get_path(row) if get_path else str(row)
        if split == "train" and year in holdout_set:
            errors.append(f"train split contains holdout year {year}: {path}")
        if split in {"val", "test"} and year in train_set:
            errors.append(f"{split} split contains train year {year}: {path}")
        if split in {"val", "test"} and year not in holdout_set:
            errors.append(f"{split} split outside holdout years (year={year}): {path}")
    return errors
