"""P2 — dex header (BM1 norm) + static receiver actions; separate H and R shards."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import PipelineConfig, dex_settings, ensure_artifact_dirs, load_config
from src.data.index import ApkIndexRow, load_apk_index, rows_for_split
from src.features.dex_header import FEATURE_DIM
from src.features.manifest_decode import ManifestDecodeError, decode_manifest
from src.features.normalization import load_normalization_stats, transform_minmax
from src.features.receivers import filter_receiver_system_actions, load_system_actions
from src.features.vectorize import vectorize_receiver_actions
from src.features.vocab import build_receiver_vocab, save_feature_layout, save_receiver_vocab
from src.preprocessing.apk_extract import ApkExtractError, extract_apk_header_extraction

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


def _resolve_normalization_header(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, Path]:
    """Reuse shipped BM1 normalization (decision 5B)."""
    bundle_norm = (
        cfg.paths.deployed_mlp_header_bundle / "features" / "normalization_header.json"
    )
    if not bundle_norm.is_file():
        raise FileNotFoundError(f"BM1 normalization_header.json not found: {bundle_norm}")
    mins, maxs = load_normalization_stats(bundle_norm)

    dest = cfg.paths.processed / "normalization_header.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_norm, dest)
    return mins, maxs, bundle_norm


def _extract_header_normalized(
    apk_path: Path,
    *,
    dex_cfg: dict,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    extraction = extract_apk_header_extraction(
        apk_path,
        mode=str(dex_cfg["mode"]),
        pattern=str(dex_cfg["dex_pattern"]),
        max_dex=int(dex_cfg["max_dex"]),
    )
    if extraction.vector.shape != (FEATURE_DIM,):
        raise ApkExtractError(f"Unexpected header dim: {extraction.vector.shape}")
    return transform_minmax(extraction.vector.reshape(1, -1), mins, maxs).reshape(-1)


def _parse_train_receivers(
    train_rows: list[ApkIndexRow],
    system_actions: frozenset[str],
    failed_log: Path,
) -> tuple[list[list[str]], dict[str, list[str]], int]:
    receiver_lists: list[list[str]] = []
    cache: dict[str, list[str]] = {}
    failures = 0

    for row in tqdm(train_rows, desc="parse:train:receivers"):
        try:
            parsed = decode_manifest(row.apk_path)
        except ManifestDecodeError as exc:
            _log_failure(failed_log, row.apk_path, str(exc))
            failures += 1
            continue
        filtered = filter_receiver_system_actions(parsed.receiver_actions, system_actions)
        receiver_lists.append(filtered)
        cache[row.sha256] = filtered

    return receiver_lists, cache, failures


def _vectorize_split(
    rows: list[ApkIndexRow],
    *,
    split: str,
    receiver_vocab: list[str],
    system_actions: frozenset[str],
    dex_cfg: dict,
    mins: np.ndarray,
    maxs: np.ndarray,
    failed_log: Path,
    receiver_cache: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], int]:
    H_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[str] = []
    sha256s: list[str] = []
    failures = 0

    for row in tqdm(rows, desc=f"vectorize:{split}"):
        try:
            H = _extract_header_normalized(
                row.apk_path, dex_cfg=dex_cfg, mins=mins, maxs=maxs
            )
        except (ApkExtractError, OSError) as exc:
            _log_failure(failed_log, row.apk_path, f"dex:{exc}")
            failures += 1
            continue

        cached = (receiver_cache or {}).get(row.sha256)
        if cached is not None:
            filtered = cached
        else:
            try:
                parsed = decode_manifest(row.apk_path)
            except ManifestDecodeError as exc:
                _log_failure(failed_log, row.apk_path, str(exc))
                failures += 1
                continue
            filtered = filter_receiver_system_actions(parsed.receiver_actions, system_actions)

        R = vectorize_receiver_actions(filtered, receiver_vocab=receiver_vocab)
        H_list.append(H.astype(np.float32))
        R_list.append(R)
        ys.append(row.label)
        paths.append(str(row.apk_path))
        sha256s.append(row.sha256)

    if not H_list:
        raise RuntimeError(f"No samples vectorized for split={split}")

    H_arr = np.stack(H_list, axis=0)
    R_arr = np.stack(R_list, axis=0)
    y_arr = np.array(ys, dtype=np.int64)
    return H_arr, R_arr, y_arr, paths, sha256s, failures


def _save_shard(
    out_path: Path,
    H: np.ndarray,
    R: np.ndarray,
    y: np.ndarray,
    paths: list[str],
    sha256s: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "H": torch.from_numpy(H),
        "R": torch.from_numpy(R),
        "y": torch.from_numpy(y),
        "paths": paths,
        "sha256": sha256s,
        "dex_dim": int(H.shape[1]),
        "receiver_dim": int(R.shape[1]),
        "num_samples": int(H.shape[0]),
    }
    torch.save(bundle, out_path)


def preprocess(cfg: PipelineConfig, *, limit: int | None = None) -> dict:
    ensure_artifact_dirs(cfg)
    cfg.paths.failed_apks_log.write_text("", encoding="utf-8")

    index_rows = load_apk_index(cfg=cfg)
    seed = int(cfg.splits.get("random_seed", 42))
    index_rows = _apply_index_limit(index_rows, limit, seed)

    train_rows = rows_for_split(index_rows, "train")
    if not train_rows:
        raise RuntimeError("No train rows in apk_index.csv")

    system_actions = load_system_actions(cfg.paths.system_actions_file)
    dex_cfg = dex_settings(cfg)
    mins, maxs, norm_source = _resolve_normalization_header(cfg)

    min_doc = int(cfg.features.get("receiver_action_min_doc_freq", 1))
    train_receiver_lists, train_cache, train_parse_failures = _parse_train_receivers(
        train_rows, system_actions, cfg.paths.failed_apks_log
    )
    if not train_receiver_lists:
        raise RuntimeError("All train APKs failed manifest parse")

    receiver_vocab = build_receiver_vocab(train_receiver_lists, min_doc_freq=min_doc)
    save_receiver_vocab(cfg.paths.processed / "receiver_action_vocab.json", receiver_vocab)

    receiver_embed_dim = int(cfg.model.get("receiver_embed_dim", 32))
    save_feature_layout(
        cfg.paths.processed / "feature_layout.json",
        dex_dim=FEATURE_DIM,
        r_size=len(receiver_vocab),
        receiver_embed_dim=receiver_embed_dim,
    )

    system_dest = cfg.paths.processed / "system_actions.json"
    shutil.copy2(cfg.paths.system_actions_file, system_dest)

    split_failures: Counter[str] = Counter()
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        split_rows = rows_for_split(index_rows, split)
        if not split_rows:
            continue
        H, R, y, paths, sha256s, failures = _vectorize_split(
            split_rows,
            split=split,
            receiver_vocab=receiver_vocab,
            system_actions=system_actions,
            dex_cfg=dex_cfg,
            mins=mins,
            maxs=maxs,
            failed_log=cfg.paths.failed_apks_log,
            receiver_cache=train_cache if split == "train" else None,
        )
        _save_shard(cfg.paths.processed / f"features_{split}.pt", H, R, y, paths, sha256s)
        split_failures[split] = failures
        split_counts[split] = int(H.shape[0])

    meta = {
        "preprocessing_version": _git_revision(cfg.root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": cfg.model_id,
        "R": len(receiver_vocab),
        "dex_dim": FEATURE_DIM,
        "multidex_mode": dex_cfg["mode"],
        "normalization_source": str(norm_source),
        "normalization_policy": "reuse_deployed_bm1",
        "train_parse_failures": train_parse_failures,
        "split_counts": split_counts,
        "split_failures": dict(split_failures),
        "system_actions_file": str(cfg.paths.system_actions_file),
        "system_actions_count": len(system_actions),
    }
    meta_path = cfg.paths.processed / "preprocessing_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 dex header + receiver fusion features.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Per-split APK cap (smoke)")
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    meta = preprocess(cfg, limit=args.limit)
    print(f"R={meta['R']}  dex_dim={meta['dex_dim']}  multidex={meta['multidex_mode']}")
    print(f"Split counts: {meta['split_counts']}")
    print(f"Normalization: {meta['normalization_policy']} ({meta['normalization_source']})")
    print(f"Artifacts → {cfg.paths.processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
