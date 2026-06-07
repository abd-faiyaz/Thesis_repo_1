"""Shared helpers for preprocessing scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PipelineConfig, load_config
from src.preprocessing.labels import LabelError, infer_label_from_parent


@dataclass(frozen=True)
class DatasetRow:
    apk_path: Path
    label: int
    apk_id: str


def normalize_name_set(names: list[str]) -> set[str]:
    return {n.strip().lower() for n in names}


def apk_id_for_path(apk_path: Path) -> str:
    digest = hashlib.sha256()
    with apk_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_apks(apk_root: Path) -> list[Path]:
    if not apk_root.is_dir():
        raise FileNotFoundError(f"APK root not found: {apk_root}")
    return sorted(apk_root.rglob("*.apk"))


def label_settings(cfg: PipelineConfig) -> tuple[set[str], set[str]]:
    pre = cfg.preprocessing
    benign = normalize_name_set(pre.get("benign_names", ["benign", "goodware", "clean", "0"]))
    malicious = normalize_name_set(
        pre.get("malicious_names", ["malware", "malicious", "virus", "1"])
    )
    return benign, malicious


def scan_apk_rows(cfg: PipelineConfig, apk_root: Path | None = None) -> list[DatasetRow]:
    root = apk_root or cfg.paths.apk_root
    benign, malicious = label_settings(cfg)
    rows: list[DatasetRow] = []
    for apk_path in discover_apks(root):
        label = infer_label_from_parent(
            apk_path, benign_names=benign, malicious_names=malicious
        )
        rows.append(
            DatasetRow(
                apk_path=apk_path.resolve(),
                label=label,
                apk_id=apk_id_for_path(apk_path),
            )
        )
    if not rows:
        raise FileNotFoundError(f"No .apk files under {root}")
    return rows


def write_dataset_index(path: Path, rows: list[DatasetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["apk_id", "path", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"apk_id": row.apk_id, "path": str(row.apk_path), "label": row.label}
            )


def read_dataset_index(path: Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                DatasetRow(
                    apk_path=Path(row["path"]).resolve(),
                    label=int(row["label"]),
                    apk_id=row["apk_id"],
                )
            )
    return rows


def year_from_apk_path(apk_path: Path) -> str | None:
    """Extract a 4-digit year folder (e.g. 2020) from an APK path."""
    match = re.search(r"/(20\d{2})/", str(apk_path).replace("\\", "/"))
    return match.group(1) if match else None


def temporal_three_way_split(
    rows: list[DatasetRow],
    *,
    train_years: list[int | str],
    test_years: list[int | str],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[DatasetRow], list[DatasetRow], list[DatasetRow]]:
    """
    Hold out APKs by top-level year folder under apk_root.

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

    dev_rows: list[DatasetRow] = []
    test_rows: list[DatasetRow] = []
    unassigned: list[DatasetRow] = []

    for row in rows:
        year = year_from_apk_path(row.apk_path)
        if year in train_set:
            dev_rows.append(row)
        elif year in test_set:
            test_rows.append(row)
        else:
            unassigned.append(row)

    if not dev_rows:
        raise ValueError(f"No APKs matched train_years={sorted(train_set)}")
    if not test_rows:
        raise ValueError(f"No APKs matched test_years={sorted(test_set)}")
    if unassigned:
        sample = unassigned[0].apk_path
        raise ValueError(
            f"{len(unassigned)} APK(s) not in train_years or test_years "
            f"(example: {sample})"
        )

    labels = np.array([r.label for r in dev_rows])
    indices = np.arange(len(dev_rows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_rows = [dev_rows[int(i)] for i in train_idx]
    val_rows = [dev_rows[int(i)] for i in val_idx]
    return train_rows, val_rows, test_rows


def stratified_split(
    rows: list[DatasetRow],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[DatasetRow], list[DatasetRow]]:
    labels = np.array([r.label for r in rows])
    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    return train_rows, val_rows


def write_split_file(path: Path, rows: list[DatasetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row.apk_id}\n")


def read_split_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def rows_for_split(
    all_rows: list[DatasetRow], split_ids: set[str]
) -> list[DatasetRow]:
    by_id = {r.apk_id: r for r in all_rows}
    missing = split_ids - set(by_id)
    if missing:
        raise KeyError(f"Split references unknown apk_id(s): {len(missing)} missing")
    return [by_id[apk_id] for apk_id in sorted(split_ids)]


def log_failure(log_path: Path, apk_path: Path, reason: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{apk_path}\t{reason}\n")


def load_processed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_processed_id(path: Path, apk_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{apk_id}\n")


def write_shard_manifest(
    path: Path,
    entries: list[dict[str, object]],
    *,
    header_dim: int,
    bow_dim: int,
    multidex_mode: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "header_dim": header_dim,
        "bow_dim": bow_dim,
        "num_samples": len(entries),
        "entries": entries,
    }
    if multidex_mode is not None:
        payload["multidex_mode"] = multidex_mode
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_config_or_exit(config_path: Path | None = None) -> PipelineConfig:
    return load_config(config_path)
