#!/usr/bin/env python3
"""Stratified train/val/test splits from apk_index.csv."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_config(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_split(
    rows: list[dict[str, str]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, str]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    assignment: dict[str, str] = {}
    rng = random.Random(seed)

    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        for i, row in enumerate(group):
            key = row["apk_path"]
            if i < n_train:
                assignment[key] = "train"
            elif i < n_train + n_val:
                assignment[key] = "val"
            else:
                assignment[key] = "test"
        print(f"{label}: train={n_train} val={n_val} test={n - n_train - n_val}")

    return assignment


def write_split(path: Path, apk_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(apk_paths) + ("\n" if apk_paths else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/val/test split files.")
    parser.add_argument("--config", type=Path, default=Path("Shared_pipeline_Files/data/dataset_paths.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = args.manifest or Path(cfg["manifest_csv"])
    if not manifest.is_file():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        return 1

    rows = read_manifest(manifest)
    if not rows:
        print("Manifest is empty.", file=sys.stderr)
        return 1

    assignment = stratified_split(rows, args.train_frac, args.val_frac, args.seed)

    splits_dir = Path(cfg.get("splits_dir", "Shared_pipeline_Files/data/splits"))
    train_paths, val_paths, test_paths = [], [], []
    for row in rows:
        split = assignment[row["apk_path"]]
        row["split"] = split
        if split == "train":
            train_paths.append(row["apk_path"])
        elif split == "val":
            val_paths.append(row["apk_path"])
        else:
            test_paths.append(row["apk_path"])

    write_split(splits_dir / "train.txt", sorted(train_paths))
    write_split(splits_dir / "val.txt", sorted(val_paths))
    write_split(splits_dir / "test.txt", sorted(test_paths))

    # Update manifest with split column
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["apk_path", "sha256", "label", "year", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Splits written under {splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
