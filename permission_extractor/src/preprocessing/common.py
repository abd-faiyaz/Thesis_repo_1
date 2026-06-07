"""Shared preprocessing helpers."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PipelineConfig
from src.preprocessing.labels import infer_label_from_parent


@dataclass(frozen=True)
class DatasetRow:
    apk_path: Path
    label: int
    apk_id: str
    year: str | None
    split: str


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


def year_from_apk_path(apk_path: Path) -> str | None:
    match = re.search(r"/(20\d{2})/", str(apk_path).replace("\\", "/"))
    return match.group(1) if match else None


def label_settings(cfg: PipelineConfig) -> tuple[set[str], set[str]]:
    pre = cfg.preprocessing
    benign = normalize_name_set(pre.get("benign_names", ["benign", "goodware", "clean", "0"]))
    malicious = normalize_name_set(
        pre.get("malicious_names", ["malware", "malicious", "virus", "1"])
    )
    return benign, malicious


def scan_apk_rows(
    cfg: PipelineConfig,
    apk_root: Path | None = None,
    *,
    limit: int | None = None,
) -> list[DatasetRow]:
    root = apk_root or cfg.paths.apk_root
    benign, malicious = label_settings(cfg)
    seen_ids: set[str] = set()
    rows: list[DatasetRow] = []

    for apk_path in discover_apks(root):
        apk_id = apk_id_for_path(apk_path)
        if apk_id in seen_ids:
            continue
        seen_ids.add(apk_id)
        label = infer_label_from_parent(
            apk_path, benign_names=benign, malicious_names=malicious
        )
        rows.append(
            DatasetRow(
                apk_path=apk_path.resolve(),
                label=label,
                apk_id=apk_id,
                year=year_from_apk_path(apk_path),
                split="unassigned",
            )
        )
        if limit is not None and len(rows) >= limit:
            break

    if not rows:
        raise FileNotFoundError(f"No .apk files under {root}")
    return rows


def assign_splits(cfg: PipelineConfig, rows: list[DatasetRow]) -> list[DatasetRow]:
    pre = cfg.preprocessing
    dev_years = {str(y) for y in pre.get("development_years", [2020, 2021])}
    holdout_years = {str(y) for y in pre.get("temporal_holdout_years", [2022, 2023])}
    seed = int(pre.get("random_seed", 42))

    dev_rows = [r for r in rows if r.year in dev_years]
    holdout_rows = [r for r in rows if r.year in holdout_years]
    other = [r for r in rows if r.year not in dev_years and r.year not in holdout_years]

    if not dev_rows:
        raise ValueError(f"No APKs found for development years {sorted(dev_years)}")

    labels = np.array([r.label for r in dev_rows])
    train_ratio = float(pre.get("train_ratio", 0.70))
    val_ratio = float(pre.get("val_ratio", 0.15))
    dev_test_ratio = float(pre.get("dev_test_ratio", 0.15))

    train_rows, temp_rows = train_test_split(
        dev_rows, train_size=train_ratio, stratify=labels, random_state=seed
    )
    temp_labels = np.array([r.label for r in temp_rows])
    val_size = val_ratio / (val_ratio + dev_test_ratio)
    val_rows, dev_test_rows = train_test_split(
        temp_rows, train_size=val_size, stratify=temp_labels, random_state=seed
    )

    def tag(split_name: str, items: list[DatasetRow]) -> list[DatasetRow]:
        return [
            DatasetRow(
                apk_path=r.apk_path,
                label=r.label,
                apk_id=r.apk_id,
                year=r.year,
                split=split_name,
            )
            for r in items
        ]

    result: list[DatasetRow] = []
    result.extend(tag("train", train_rows))
    result.extend(tag("val", val_rows))
    result.extend(tag("dev_test", dev_test_rows))
    result.extend(tag("temporal_holdout", holdout_rows))
    result.extend(tag("other", other))
    return result


def write_dataset_index(path: Path, rows: list[DatasetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["apk_id", "path", "label", "year", "split"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "apk_id": row.apk_id,
                    "path": str(row.apk_path),
                    "label": row.label,
                    "year": row.year or "",
                    "split": row.split,
                }
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
                    year=row.get("year") or None,
                    split=row.get("split", "unassigned"),
                )
            )
    return rows


def rows_for_split(rows: list[DatasetRow], split: str) -> list[DatasetRow]:
    return [r for r in rows if r.split == split]


def write_split_file(path: Path, rows: list[DatasetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{r.apk_id}\t{r.apk_path}\t{r.label}\n" for r in rows]
    path.write_text("".join(lines), encoding="utf-8")


def split_counts(rows: list[DatasetRow]) -> Counter[str]:
    return Counter(r.split for r in rows)
