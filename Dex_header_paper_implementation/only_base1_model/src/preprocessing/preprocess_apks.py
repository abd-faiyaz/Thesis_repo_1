"""Batch-extract Dex header features from APKs and save a training-ready tensor file."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import ensure_artifact_dirs, load_config
from src.features.dex_header import (
    FEATURE_DIM,
    DexHeaderError,
    extract_header_features,
    parse_dex_header_fields,
)
from src.features.normalization import (
    fit_minmax,
    save_normalization_stats,
    transform_minmax,
)
from src.preprocessing.apk_extract import ApkExtractError, read_classes_dex
from src.preprocessing.labels import LabelError, load_labels_csv, resolve_label

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _discover_apks(apk_root: Path) -> list[Path]:
    if not apk_root.is_dir():
        raise FileNotFoundError(f"APK root not found: {apk_root}")
    return sorted(apk_root.rglob("*.apk"))


def _log_failure(log_path: Path, apk_path: Path, reason: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{apk_path}\t{reason}\n")


def _save_aggregate(
    out_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    output_format: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if output_format == "npy":
        np.save(out_path.with_suffix(".features.npy"), features)
        np.save(out_path.with_suffix(".labels.npy"), labels)
        np.save(out_path.with_suffix(".paths.npy"), np.array(paths, dtype=object))
        return

    bundle = {
        "features": torch.from_numpy(features).float(),
        "labels": torch.from_numpy(labels).float(),
        "paths": paths,
        "feature_dim": meta["feature_dim"],
        "num_samples": meta["num_samples"],
        "normalization_mins": torch.from_numpy(mins).float(),
        "normalization_maxs": torch.from_numpy(maxs).float(),
        "created_at": meta["created_at"],
    }
    torch.save(bundle, out_path)


def preprocess(
    *,
    apk_root: Path,
    processed_dir: Path,
    failed_log: Path,
    normalization_stats_path: Path,
    dex_entry_name: str,
    output_format: str,
    aggregate_filename: str,
    label_mode: str,
    labels_csv: Path | None,
    benign_names: set[str],
    malicious_names: set[str],
    limit: int | None = None,
) -> dict[str, int]:
    """Extract features from all APKs under apk_root; return summary counts."""
    apks = _discover_apks(apk_root)
    if limit is not None:
        apks = apks[:limit]

    if not apks:
        raise FileNotFoundError(f"No .apk files under {apk_root}")

    csv_cache = load_labels_csv(labels_csv) if label_mode == "csv" and labels_csv else None

    raw_features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    failed = 0

    for apk_path in tqdm(apks, desc="Extracting Dex headers", unit="apk"):
        try:
            label = resolve_label(
                apk_path,
                label_mode=label_mode,
                labels_csv=labels_csv,
                benign_names=benign_names,
                malicious_names=malicious_names,
                csv_cache=csv_cache,
            )
            dex_bytes = read_classes_dex(apk_path, entry_name=dex_entry_name)
            parse_dex_header_fields(dex_bytes)
            vector = extract_header_features(dex_bytes)
        except (ApkExtractError, DexHeaderError, LabelError) as exc:
            failed += 1
            _log_failure(failed_log, apk_path, str(exc))
            continue
        except Exception as exc:
            failed += 1
            _log_failure(failed_log, apk_path, f"unexpected: {exc}")
            continue

        raw_features.append(vector)
        labels.append(label)
        paths.append(str(apk_path.resolve()))

    if not raw_features:
        raise RuntimeError("No APKs processed successfully; see failed_apks.log")

    feature_matrix = np.stack(raw_features, axis=0)
    label_array = np.array(labels, dtype=np.float64)

    mins, maxs = fit_minmax(feature_matrix)
    save_normalization_stats(
        normalization_stats_path,
        mins,
        maxs,
        extra={"num_samples": len(paths), "apk_root": str(apk_root)},
    )
    normalized = transform_minmax(feature_matrix, mins, maxs)

    out_path = processed_dir / aggregate_filename
    _save_aggregate(
        out_path,
        normalized,
        label_array,
        paths,
        mins,
        maxs,
        output_format,
    )

    return {
        "total_apks": len(apks),
        "successful": len(paths),
        "failed": failed,
        "feature_dim": FEATURE_DIM,
        "output": str(out_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess APKs: extract classes.dex header features."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--apk-root",
        type=Path,
        default=None,
        help="Override paths.apk_root from config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N APKs (smoke tests)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    pre = cfg.preprocessing
    apk_root = args.apk_root or cfg.paths.apk_root
    label_mode = pre.get("label_mode", "parent_folder")
    labels_csv_raw = pre.get("labels_csv")
    labels_csv = Path(labels_csv_raw) if labels_csv_raw else None
    if labels_csv and not labels_csv.is_absolute():
        labels_csv = (cfg.root / labels_csv).resolve()

    benign_names = {_normalize_set_name(n) for n in pre.get("benign_names", ["benign", "goodware", "clean", "0"])}
    malicious_names = {
        _normalize_set_name(n) for n in pre.get("malicious_names", ["malware", "malicious", "virus", "1"])
    }

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    print(f"APK root: {apk_root}")
    print(f"Output dir: {cfg.paths.processed_dir}")
    print(f"Label mode: {label_mode}")

    summary = preprocess(
        apk_root=apk_root,
        processed_dir=cfg.paths.processed_dir,
        failed_log=cfg.paths.failed_apks_log,
        normalization_stats_path=cfg.paths.normalization_stats,
        dex_entry_name=pre.get("dex_entry_name", "classes.dex"),
        output_format=pre.get("output_format", "pt"),
        aggregate_filename=pre.get("aggregate_filename", "dex_header_features.pt"),
        label_mode=label_mode,
        labels_csv=labels_csv,
        benign_names=benign_names,
        malicious_names=malicious_names,
        limit=args.limit,
    )

    print("\nPreprocessing complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if summary["failed"]:
        print(f"  failures logged to: {cfg.paths.failed_apks_log}")
    return 0


def _normalize_set_name(name: str) -> str:
    return name.strip().lower()


if __name__ == "__main__":
    raise SystemExit(main())
