"""Infer malware/benign labels from APK paths or CSV manifests."""

from __future__ import annotations

import csv
from pathlib import Path


class LabelError(ValueError):
    """Could not resolve a binary label for an APK."""


def _normalize_name(value: str) -> str:
    return value.strip().lower()


def infer_label_from_parent(
    apk_path: Path,
    *,
    benign_names: set[str],
    malicious_names: set[str],
) -> int:
    """
    Walk parents until a folder name matches benign (0) or malicious (1) sets.
    """
    for parent in apk_path.parents:
        name = _normalize_name(parent.name)
        if name in benign_names:
            return 0
        if name in malicious_names:
            return 1
    raise LabelError(f"No label folder found for {apk_path}")


def load_labels_csv(csv_path: Path) -> dict[str, int]:
    """
    CSV columns: path (or apk_path), label (0/1 or benign/malicious).
    Paths may be absolute or relative to the CSV parent directory.
    """
    mapping: dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise LabelError(f"Empty CSV: {csv_path}")

        path_key = next(
            (k for k in reader.fieldnames if k.lower() in ("path", "apk_path", "apk", "file")),
            None,
        )
        label_key = next(
            (k for k in reader.fieldnames if k.lower() in ("label", "class", "y")),
            None,
        )
        if not path_key or not label_key:
            raise LabelError(f"CSV must have path and label columns: {csv_path}")

        for row in reader:
            raw_path = row[path_key].strip()
            p = Path(raw_path)
            if not p.is_absolute():
                p = (csv_path.parent / p).resolve()
            label = _parse_label_value(row[label_key])
            mapping[str(p)] = label
    return mapping


def _parse_label_value(value: str) -> int:
    v = value.strip().lower()
    if v in ("0", "benign", "good", "goodware", "clean"):
        return 0
    if v in ("1", "malware", "malicious", "bad", "virus"):
        return 1
    raise LabelError(f"Unknown label value: {value!r}")


def resolve_label(
    apk_path: Path,
    *,
    label_mode: str,
    labels_csv: Path | None,
    benign_names: set[str],
    malicious_names: set[str],
    csv_cache: dict[str, int] | None = None,
) -> int:
    apk_resolved = apk_path.resolve()
    if label_mode == "csv":
        if not labels_csv:
            raise LabelError("label_mode=csv requires labels_csv in config")
        cache = csv_cache if csv_cache is not None else load_labels_csv(labels_csv)
        key = str(apk_resolved)
        if key not in cache:
            raise LabelError(f"APK not listed in labels CSV: {apk_path}")
        return cache[key]
    if label_mode == "parent_folder":
        return infer_label_from_parent(
            apk_resolved,
            benign_names=benign_names,
            malicious_names=malicious_names,
        )
    raise LabelError(f"Unsupported label_mode: {label_mode}")
