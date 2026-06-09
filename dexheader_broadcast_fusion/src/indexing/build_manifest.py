"""P1 — Build APK index with temporal train + stratified holdout val/test splits."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import PipelineConfig

YEAR_RE = re.compile(r"/(20\d{2})/")


@dataclass(frozen=True)
class IndexRow:
    apk_path: str
    sha256: str
    label: int
    year: int | None
    split: str
    apk_size_bytes: int | None = None
    num_dex_files: int | None = None


def normalize_name_set(names: list[str]) -> set[str]:
    return {str(n).strip().lower() for n in names}


def infer_label_from_parent(
    apk_path: Path,
    apk_root: Path,
    *,
    benign_names: set[str],
    malicious_names: set[str],
) -> int | None:
    try:
        rel = apk_path.relative_to(apk_root)
    except ValueError:
        rel = apk_path
    for part in rel.parts[:-1]:
        key = part.lower()
        if key in benign_names:
            return 0
        if key in malicious_names:
            return 1
    return None


def year_from_path(apk_path: Path | str) -> int | None:
    match = YEAR_RE.search(str(apk_path).replace("\\", "/"))
    return int(match.group(1)) if match else None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def apk_metadata(path: Path) -> tuple[int | None, int | None]:
    """Return (size_bytes, num_dex_files) or (None, None) if unreadable."""
    try:
        size = path.stat().st_size
        with zipfile.ZipFile(path, "r") as zf:
            dex_count = sum(1 for name in zf.namelist() if name.endswith(".dex"))
        return size, dex_count
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return None, None


def discover_apks(apk_root: Path) -> list[Path]:
    if not apk_root.is_dir():
        raise FileNotFoundError(f"apk_root not found: {apk_root}")
    return sorted(p for p in apk_root.rglob("*.apk") if p.is_file())


def _index_progress(items, *, desc: str):
    return tqdm(items, desc=desc, unit="apk", dynamic_ncols=True)


def _label_to_int(label_raw: str) -> int:
    key = label_raw.strip().lower()
    if key in {"benign", "goodware", "clean", "good", "0"}:
        return 0
    if key in {"malware", "malicious", "virus", "bad", "1"}:
        return 1
    return int(label_raw)


def load_shared_manifest_rows(
    manifest_csv: Path,
    apk_root: Path,
) -> list[IndexRow] | None:
    if not manifest_csv.is_file():
        return None

    rows: list[IndexRow] = []
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rel = raw["apk_path"]
            apk_path = Path(rel)
            if not apk_path.is_absolute():
                apk_path = (apk_root / rel).resolve()
            label = _label_to_int(raw["label"])
            year_raw = raw.get("year", "")
            year = int(year_raw) if year_raw else year_from_path(apk_path)
            sha = raw.get("sha256", "").strip()
            rows.append(
                IndexRow(
                    apk_path=str(apk_path),
                    sha256=sha,
                    label=label,
                    year=year,
                    split="unassigned",
                )
            )
    return rows


def scan_apk_rows(
    cfg: PipelineConfig,
    *,
    apk_root: Path | None = None,
    limit: int | None = None,
    use_shared: bool = True,
) -> tuple[list[IndexRow], list[str], list[str]]:
    """
    Scan corpus or reuse shared manifest.

    Returns (rows, failed_paths, duplicate_log_lines).
    """
    root = apk_root or cfg.paths.apk_root
    idx_cfg = cfg.indexing
    benign = normalize_name_set(idx_cfg.get("benign_names", ["benign"]))
    malicious = normalize_name_set(idx_cfg.get("malicious_names", ["malware"]))
    collect_meta = bool(idx_cfg.get("collect_apk_metadata", True))

    failed: list[str] = []
    dup_lines: list[str] = []

    shared_rows: list[IndexRow] | None = None
    if use_shared and cfg.paths.shared_manifest_csv:
        shared_rows = load_shared_manifest_rows(cfg.paths.shared_manifest_csv, root)

    if shared_rows:
        print(
            f"P1 index: {len(shared_rows)} rows from shared manifest "
            f"({cfg.paths.shared_manifest_csv})",
            flush=True,
        )
        candidate_rows = shared_rows
    else:
        print(f"P1 index: discovering APKs under {root} ...", flush=True)
        candidate_rows = []
        apks = discover_apks(root)
        if limit:
            apks = apks[:limit]
        print(f"P1 index: hashing {len(apks)} APKs ...", flush=True)
        for apk_path in _index_progress(apks, desc="P1 hash"):
            label = infer_label_from_parent(
                apk_path, root, benign_names=benign, malicious_names=malicious
            )
            if label is None:
                failed.append(f"no_label\t{apk_path}")
                continue
            try:
                digest = sha256_file(apk_path)
            except OSError as exc:
                failed.append(f"hash_error\t{apk_path}\t{exc}")
                continue
            size_bytes: int | None = None
            num_dex: int | None = None
            if collect_meta:
                size_bytes, num_dex = apk_metadata(apk_path)
                if size_bytes is None:
                    failed.append(f"unreadable_zip\t{apk_path}")
                    continue
            candidate_rows.append(
                IndexRow(
                    apk_path=str(apk_path.resolve()),
                    sha256=digest,
                    label=label,
                    year=year_from_path(apk_path),
                    split="unassigned",
                    apk_size_bytes=size_bytes,
                    num_dex_files=num_dex,
                )
            )

    seen: dict[str, IndexRow] = {}
    rows: list[IndexRow] = []
    dedupe_desc = "P1 validate" if shared_rows else "P1 finalize"
    for row in _index_progress(candidate_rows, desc=dedupe_desc):
        apk = Path(row.apk_path)
        if not apk.is_file():
            failed.append(f"missing_file\t{row.apk_path}")
            continue
        digest = row.sha256 or sha256_file(apk)
        if digest in seen:
            dup_lines.append(f"duplicate_sha256\t{digest}\t{row.apk_path}\tkept={seen[digest].apk_path}")
            continue
        if not row.sha256:
            size_bytes = row.apk_size_bytes
            num_dex = row.num_dex_files
            if collect_meta and size_bytes is None:
                size_bytes, num_dex = apk_metadata(apk)
                if size_bytes is None:
                    failed.append(f"unreadable_zip\t{row.apk_path}")
                    continue
            row = replace(row, sha256=digest, apk_size_bytes=size_bytes, num_dex_files=num_dex)
        seen[digest] = row
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break

    if not rows:
        raise FileNotFoundError(f"No indexable APKs under {root}")
    return rows, failed, dup_lines


def assign_splits(cfg: PipelineConfig, rows: list[IndexRow]) -> list[IndexRow]:
    from shared_splits import resolve_split_config, temporal_holdout_partition

    split_cfg = resolve_split_config(cfg.splits)
    train_rows, val_rows, test_rows, other_rows = temporal_holdout_partition(
        rows,
        [r.label for r in rows],
        get_year=lambda row: row.year,
        train_years=split_cfg.train_years,
        holdout_years=split_cfg.holdout_years,
        val_fraction_of_holdout=split_cfg.val_fraction_of_holdout,
        seed=split_cfg.random_seed,
    )

    def tag(split_name: str, items: list[IndexRow]) -> list[IndexRow]:
        return [replace(r, split=split_name) for r in items]

    result: list[IndexRow] = []
    result.extend(tag("train", train_rows))
    result.extend(tag("val", val_rows))
    result.extend(tag("test", test_rows))
    result.extend(tag("other", other_rows))
    return result


def split_summary(rows: list[IndexRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = summary.setdefault(row.split, {"benign": 0, "malware": 0, "total": 0})
        bucket["total"] += 1
        if row.label == 0:
            bucket["benign"] += 1
        else:
            bucket["malware"] += 1
    return summary


def year_split_crosscheck(rows: list[IndexRow], cfg: PipelineConfig) -> list[str]:
    from shared_splits import crosscheck_temporal_holdout, resolve_split_config

    split_cfg = resolve_split_config(cfg.splits)
    return crosscheck_temporal_holdout(
        rows,
        get_split=lambda row: row.split,
        get_year=lambda row: row.year,
        get_path=lambda row: row.apk_path,
        train_years=split_cfg.train_years,
        holdout_years=split_cfg.holdout_years,
    )


def write_index_csv(path: Path, rows: list[IndexRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "apk_path",
        "sha256",
        "label",
        "year",
        "split",
        "apk_size_bytes",
        "num_dex_files",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "apk_path": row.apk_path,
                    "sha256": row.sha256,
                    "label": row.label,
                    "year": row.year if row.year is not None else "",
                    "split": row.split,
                    "apk_size_bytes": row.apk_size_bytes if row.apk_size_bytes is not None else "",
                    "num_dex_files": row.num_dex_files if row.num_dex_files is not None else "",
                }
            )


def write_index_json(path: Path, rows: list[IndexRow], cfg: PipelineConfig) -> None:
    payload = {
        "model_id": cfg.model_id,
        "count": len(rows),
        "splits": split_summary(rows),
        "split_policy": cfg.splits,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_split_lists(splits_dir: Path, rows: list[IndexRow]) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        paths = [r.apk_path for r in rows if r.split == split_name]
        (splits_dir / f"{split_name}.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""),
            encoding="utf-8",
        )


def append_log(path: Path, lines: list[str]) -> None:
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
