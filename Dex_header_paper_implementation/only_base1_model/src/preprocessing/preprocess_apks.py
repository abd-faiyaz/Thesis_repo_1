"""Batch-extract Dex header features from APKs and save a training-ready tensor file."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.config import ensure_artifact_dirs, load_config
from src.features.dex_header import FEATURE_DIM, DexHeaderError
from src.features.multidex import multidex_settings
from src.features.normalization import (
    build_normalization_metadata,
    fit_minmax,
    save_normalization_stats,
    transform_minmax,
)
from src.preprocessing.apk_extract import ApkExtractError, extract_apk_header_extraction
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


def _resolve_multidex_config(pre: dict[str, Any]) -> dict[str, Any]:
    """Resolve multidex settings; map legacy dex_entry_name to primary_only."""
    if pre.get("dex_entry_name") and not pre.get("multidex"):
        warnings.warn(
            "preprocessing.dex_entry_name is deprecated; using multidex.mode=primary_only",
            DeprecationWarning,
            stacklevel=2,
        )
        md = multidex_settings(pre)
        return {
            "mode": "primary_only",
            "dex_pattern": md["dex_pattern"],
            "max_dex": md["max_dex"],
        }
    return multidex_settings(pre)


def _feature_dim_for_mode(mode: str, max_dex: int) -> int:
    if mode == "concat":
        return FEATURE_DIM * max_dex
    return FEATURE_DIM


def _save_aggregate(
    out_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    output_format: str,
    *,
    bundle_metadata: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **bundle_metadata,
    }

    if output_format == "npy":
        np.save(out_path.with_suffix(".features.npy"), features)
        np.save(out_path.with_suffix(".labels.npy"), labels)
        np.save(out_path.with_suffix(".paths.npy"), np.array(paths, dtype=object))
        meta_path = out_path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
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
        "multidex_mode": meta["multidex_mode"],
        "dex_pattern": meta["dex_pattern"],
        "cache_version": meta["cache_version"],
        "max_dex": meta["max_dex"],
        "dex_file_counts": meta.get("dex_file_counts"),
    }
    torch.save(bundle, out_path)


def preprocess(
    *,
    apk_root: Path,
    processed_dir: Path,
    failed_log: Path,
    normalization_stats_path: Path,
    multidex: dict[str, Any],
    cache_version: int,
    output_format: str,
    aggregate_filename: str,
    label_mode: str,
    labels_csv: Path | None,
    benign_names: set[str],
    malicious_names: set[str],
    limit: int | None = None,
) -> dict[str, int | str]:
    """Extract features from all APKs under apk_root; return summary counts."""
    apks = _discover_apks(apk_root)
    if limit is not None:
        apks = apks[:limit]

    if not apks:
        raise FileNotFoundError(f"No .apk files under {apk_root}")

    csv_cache = load_labels_csv(labels_csv) if label_mode == "csv" and labels_csv else None
    expected_feature_dim = _feature_dim_for_mode(
        str(multidex["mode"]),
        int(multidex["max_dex"]),
    )

    raw_features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    dex_count_hist: dict[int, int] = {}
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
            extraction = extract_apk_header_extraction(
                apk_path,
                mode=str(multidex["mode"]),
                pattern=str(multidex["dex_pattern"]),
                max_dex=int(multidex["max_dex"]),
            )
            vector = extraction.vector
            if vector.shape != (expected_feature_dim,):
                raise ApkExtractError(
                    f"Feature shape {vector.shape} != expected ({expected_feature_dim},)"
                )
            dex_count_hist[extraction.num_dex_files] = (
                dex_count_hist.get(extraction.num_dex_files, 0) + 1
            )
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

    dex_file_counts = {str(k): v for k, v in sorted(dex_count_hist.items())}
    norm_meta = build_normalization_metadata(
        multidex_mode=str(multidex["mode"]),
        dex_pattern=str(multidex["dex_pattern"]),
        cache_version=cache_version,
        num_samples=len(paths),
        apk_root=str(apk_root),
        max_dex=int(multidex["max_dex"]),
        dex_file_counts=dex_file_counts,
    )

    mins, maxs = fit_minmax(feature_matrix, feature_dim=expected_feature_dim)
    save_normalization_stats(
        normalization_stats_path,
        mins,
        maxs,
        feature_dim=expected_feature_dim,
        extra=norm_meta,
    )
    normalized = transform_minmax(feature_matrix, mins, maxs)

    bundle_metadata = {
        "multidex_mode": norm_meta["multidex_mode"],
        "dex_pattern": norm_meta["dex_pattern"],
        "cache_version": norm_meta["cache_version"],
        "max_dex": norm_meta["max_dex"],
        "dex_file_counts": dex_file_counts,
    }
    out_path = processed_dir / aggregate_filename
    _save_aggregate(
        out_path,
        normalized,
        label_array,
        paths,
        mins,
        maxs,
        output_format,
        bundle_metadata=bundle_metadata,
    )

    return {
        "total_apks": len(apks),
        "successful": len(paths),
        "failed": failed,
        "feature_dim": expected_feature_dim,
        "multidex_mode": str(multidex["mode"]),
        "cache_version": cache_version,
        "dex_file_counts": dex_file_counts,
        "output": str(out_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess APKs: extract and aggregate classes*.dex header features."
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
    multidex = _resolve_multidex_config(pre)
    cache_version = int(pre.get("cache_version", 2))

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    print(f"APK root: {apk_root}")
    print(f"Output dir: {cfg.paths.processed_dir}")
    print(f"Label mode: {label_mode}")
    print(f"Multidex mode: {multidex['mode']}")
    print(f"Dex pattern: {multidex['dex_pattern']}")
    print(f"Cache version: {cache_version}")

    summary = preprocess(
        apk_root=apk_root,
        processed_dir=cfg.paths.processed_dir,
        failed_log=cfg.paths.failed_apks_log,
        normalization_stats_path=cfg.paths.normalization_stats,
        multidex=multidex,
        cache_version=cache_version,
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
