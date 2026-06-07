"""P2 batch job — MLDP freeze S, dex headers, vectorize all splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.constants import DEX_FEATURE_DIM
from src.data.index import ApkIndexRow, load_apk_index, rows_for_split
from src.features.manifest_decode import ManifestDecodeError, decode_manifest
from src.features.mldp.select import run_mldp_selection, save_mldp_artifacts
from src.features.normalization import (
    copy_deployed_normalization,
    load_normalization_header,
    transform_vector,
)
from src.features.vectorize import vectorize_cascade
from src.features.vocab import save_feature_layout
from src.preprocessing.apk_extract import ApkExtractError, extract_apk_raw_header

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _limit_split_rows(rows: list[ApkIndexRow], limit: int, seed: int) -> list[ApkIndexRow]:
    if len(rows) <= limit:
        return rows
    labels = np.array([r.label for r in rows])
    indices = np.arange(len(rows))
    chosen, _ = train_test_split(
        indices, train_size=limit, stratify=labels, random_state=seed
    )
    return [rows[int(i)] for i in sorted(chosen)]


def _apply_index_limit(rows: list[ApkIndexRow], limit: int | None, seed: int) -> list[ApkIndexRow]:
    if limit is None:
        return rows
    by_split: dict[str, list[ApkIndexRow]] = {}
    for row in rows:
        by_split.setdefault(row.split, []).append(row)
    limited: list[ApkIndexRow] = []
    for split in ("train", "val", "test"):
        split_rows = by_split.get(split, [])
        if split_rows:
            limited.extend(_limit_split_rows(split_rows, limit, seed))
    return limited


def _git_revision(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _log_failure(log_path: Path, apk_path: Path, reason: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{apk_path}\t{reason}\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dex_settings(cfg: PipelineConfig) -> tuple[str, str]:
    dex_cfg = cfg.dex
    mode = str(dex_cfg.get("multidex_mode", "sum"))
    pattern = str(dex_cfg.get("dex_pattern", r"^classes(\d*)\.dex$"))
    return mode, pattern


def _resolve_normalization(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    out_path = cfg.paths.processed / "normalization_header.json"
    use_deployed = bool(cfg.model.get("mode_a_use_deployed_dex_normalization", True))

    if use_deployed:
        src = cfg.paths.deployed_mlp_header_bundle / "features" / "normalization_header.json"
        if not src.is_file():
            raise FileNotFoundError(f"Deployed normalization not found: {src}")
        copy_deployed_normalization(src, out_path)
        mins, maxs, meta = load_normalization_header(out_path)
        meta = dict(meta)
        meta["source"] = "deployed_mlp_header"
        meta["source_path"] = str(src)
        meta["source_sha256"] = _sha256_file(src)
        out_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return mins, maxs, meta

    raise NotImplementedError(
        "Corpus-fitted dex normalization is disabled; set model.mode_a_use_deployed_dex_normalization: true"
    )


def _parse_train_for_mldp(
    train_rows: list[ApkIndexRow],
    *,
    include_sdk_23: bool,
    failed_log: Path,
) -> tuple[list[set[str]], list[int], dict[str, tuple[str, ...]], int]:
    transactions: list[set[str]] = []
    labels: list[int] = []
    perm_cache: dict[str, tuple[str, ...]] = {}
    failures = 0

    for row in tqdm(train_rows, desc="parse:train(mldp)"):
        try:
            parsed = decode_manifest(row.apk_path, include_sdk_23=include_sdk_23)
        except ManifestDecodeError as exc:
            _log_failure(failed_log, row.apk_path, str(exc))
            failures += 1
            continue
        transactions.append(set(parsed.permissions))
        labels.append(row.label)
        perm_cache[row.sha256] = parsed.permissions

    return transactions, labels, perm_cache, failures


def _extract_apk_features(
    row: ApkIndexRow,
    *,
    mldp_vocab: list[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    multidex_mode: str,
    dex_pattern: str,
    include_sdk_23: bool,
    perm_cache: dict[str, tuple[str, ...]] | None,
    failed_log: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    cached = (perm_cache or {}).get(row.sha256)
    if cached is not None:
        permissions = cached
    else:
        try:
            parsed = decode_manifest(row.apk_path, include_sdk_23=include_sdk_23)
        except ManifestDecodeError as exc:
            _log_failure(failed_log, row.apk_path, f"manifest:{exc}")
            return None
        permissions = parsed.permissions

    try:
        h_raw = extract_apk_raw_header(
            row.apk_path,
            mode=multidex_mode,
            pattern=dex_pattern,
        )
    except ApkExtractError as exc:
        _log_failure(failed_log, row.apk_path, f"dex:{exc}")
        return None

    h = transform_vector(h_raw, mins, maxs).astype(np.float32)
    return vectorize_cascade(permissions, h, mldp_vocab=mldp_vocab)


def _vectorize_split(
    rows: list[ApkIndexRow],
    *,
    split: str,
    mldp_vocab: list[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    multidex_mode: str,
    dex_pattern: str,
    include_sdk_23: bool,
    failed_log: Path,
    perm_cache: dict[str, tuple[str, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str], list[str], int]:
    x_s_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[str] = []
    sha256s: list[str] = []
    failures = 0

    for row in tqdm(rows, desc=f"vectorize:{split}"):
        out = _extract_apk_features(
            row,
            mldp_vocab=mldp_vocab,
            mins=mins,
            maxs=maxs,
            multidex_mode=multidex_mode,
            dex_pattern=dex_pattern,
            include_sdk_23=include_sdk_23,
            perm_cache=perm_cache,
            failed_log=failed_log,
        )
        if out is None:
            failures += 1
            continue
        x_s, h, x = out
        x_s_list.append(x_s)
        h_list.append(h)
        x_list.append(x)
        ys.append(row.label)
        paths.append(str(row.apk_path))
        sha256s.append(row.sha256)

    if not x_s_list:
        raise RuntimeError(f"No samples vectorized for split={split!r}")

    return (
        np.stack(x_s_list, axis=0).astype(np.float32),
        np.stack(h_list, axis=0).astype(np.float32),
        np.stack(x_list, axis=0).astype(np.float32),
        ys,
        paths,
        sha256s,
        failures,
    )


def _save_feature_shard(
    out_path: Path,
    *,
    x_s: np.ndarray,
    h: np.ndarray,
    x: np.ndarray,
    y: list[int],
    paths: list[str],
    sha256s: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "x_S": torch.from_numpy(x_s),
        "H": torch.from_numpy(h),
        "x": torch.from_numpy(x),
        "y": torch.tensor(y, dtype=torch.int64),
        "paths": paths,
        "sha256": sha256s,
        "feature_dims": {
            "S": int(x_s.shape[1]),
            "H": int(h.shape[1]),
            "d": int(x.shape[1]),
        },
        "num_samples": int(x.shape[0]),
    }
    torch.save(bundle, out_path)


def preprocess(cfg: PipelineConfig, *, limit: int | None = None) -> dict:
    ensure_artifact_dirs(cfg)
    cfg.paths.failed_apks_log.write_text("", encoding="utf-8")

    index_rows = load_apk_index(cfg=cfg)
    seed = int(cfg.splits.get("random_seed", 42))
    env_limit = os.environ.get("PREPROCESS_LIMIT")
    if limit is None and env_limit:
        limit = int(env_limit)
    index_rows = _apply_index_limit(index_rows, limit, seed)

    train_rows = rows_for_split(index_rows, "train")
    if not train_rows:
        raise RuntimeError("No train rows in apk_index.csv")

    include_sdk_23 = bool(cfg.features.get("include_uses_permission_sdk_23", True))
    multidex_mode, dex_pattern = _dex_settings(cfg)

    train_tx, train_labels, train_perm_cache, train_failures = _parse_train_for_mldp(
        train_rows,
        include_sdk_23=include_sdk_23,
        failed_log=cfg.paths.failed_apks_log,
    )
    if not train_tx:
        raise RuntimeError("All train APKs failed manifest parse")

    selected, trace = run_mldp_selection(
        cfg, train_transactions=train_tx, train_labels=train_labels
    )
    save_mldp_artifacts(cfg.paths.processed, selected, trace)
    save_feature_layout(
        cfg.paths.processed / "feature_layout.json",
        s_size=len(selected),
        h_size=DEX_FEATURE_DIM,
    )

    mins, maxs, norm_meta = _resolve_normalization(cfg)

    split_failures: Counter[str] = Counter()
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        split_rows = rows_for_split(index_rows, split)
        if not split_rows:
            continue
        x_s, h, x, ys, paths, sha256s, failures = _vectorize_split(
            split_rows,
            split=split,
            mldp_vocab=selected,
            mins=mins,
            maxs=maxs,
            multidex_mode=multidex_mode,
            dex_pattern=dex_pattern,
            include_sdk_23=include_sdk_23,
            failed_log=cfg.paths.failed_apks_log,
            perm_cache=train_perm_cache if split == "train" else None,
        )
        _save_feature_shard(
            cfg.paths.processed / f"features_{split}.pt",
            x_s=x_s,
            h=h,
            x=x,
            y=ys,
            paths=paths,
            sha256s=sha256s,
        )
        split_failures[split] = failures
        split_counts[split] = int(x.shape[0])

    meta = {
        "preprocessing_version": _git_revision(cfg.root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": cfg.model_id,
        "S": len(selected),
        "H": DEX_FEATURE_DIM,
        "d": len(selected) + DEX_FEATURE_DIM,
        "multidex_mode": multidex_mode,
        "dex_pattern": dex_pattern,
        "dex_normalization": "per_byte_div255 -> multidex_sum -> corpus_minmax",
        "normalization_source": norm_meta.get("source", "unknown"),
        "train_mldp_parse_failures": train_failures,
        "split_counts": split_counts,
        "split_failures": dict(split_failures),
        "mldp_stages": trace.get("stages", {}),
        "fallback_published_list_used": trace.get("fallback_published_list_used", False),
        "preprocess_limit": limit,
    }
    meta_path = cfg.paths.processed / "preprocessing_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 manifest + dex header feature extraction.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Per-split APK cap (smoke test)")
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    meta = preprocess(cfg, limit=args.limit)

    print(f"|S|={meta['S']}  H={meta['H']}  d={meta['d']}")
    print(f"Split counts: {meta['split_counts']}")
    print(f"Split failures: {meta['split_failures']}")
    print(f"MLDP stages: {meta['mldp_stages']}")
    print(f"Dex normalization: {meta['normalization_source']}")
    if meta.get("fallback_published_list_used"):
        print("NOTE: published Table I fallback list used for S")
    print(f"Artifacts → {cfg.paths.processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
