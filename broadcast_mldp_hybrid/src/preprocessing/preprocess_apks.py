"""P2 batch job — parse manifests, freeze S/A from train, vectorize all splits."""

from __future__ import annotations

import argparse
import json
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
from src.data.index import ApkIndexRow, load_apk_index, rows_for_split
from src.features.manifest_decode import ManifestDecodeError, decode_manifest
from src.features.mldp.select import run_mldp_selection, save_mldp_artifacts
from src.features.receivers import filter_receiver_system_actions, load_system_actions
from src.features.vectorize import vectorize_hybrid
from src.features.vocab import (
    build_receiver_vocab,
    save_feature_layout,
    save_receiver_vocab,
)

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


def _parse_train_corpus(
    train_rows: list[ApkIndexRow],
    system_actions: frozenset[str],
    failed_log: Path,
) -> tuple[list[set[str]], list[int], list[list[str]], dict[str, tuple[tuple[str, ...], list[str]]], int]:
    transactions: list[set[str]] = []
    labels: list[int] = []
    receiver_action_lists: list[list[str]] = []
    cache: dict[str, tuple[tuple[str, ...], list[str]]] = {}
    failures = 0

    for row in tqdm(train_rows, desc="parse:train"):
        try:
            parsed = decode_manifest(row.apk_path)
        except ManifestDecodeError as exc:
            _log_failure(failed_log, row.apk_path, str(exc))
            failures += 1
            continue
        filtered = filter_receiver_system_actions(parsed.receiver_actions, system_actions)
        transactions.append(set(parsed.permissions))
        labels.append(row.label)
        receiver_action_lists.append(filtered)
        cache[row.sha256] = (parsed.permissions, filtered)

    return transactions, labels, receiver_action_lists, cache, failures


def _vectorize_split(
    rows: list[ApkIndexRow],
    *,
    split: str,
    mldp_vocab: list[str],
    receiver_vocab: list[str],
    system_actions: frozenset[str],
    failed_log: Path,
    parse_cache: dict[str, tuple[tuple[str, ...], list[str]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], int]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[str] = []
    sha256s: list[str] = []
    failures = 0

    for row in tqdm(rows, desc=f"vectorize:{split}"):
        cached = (parse_cache or {}).get(row.sha256)
        if cached is not None:
            perms, filtered_actions = cached
        else:
            try:
                parsed = decode_manifest(row.apk_path)
            except ManifestDecodeError as exc:
                _log_failure(failed_log, row.apk_path, str(exc))
                failures += 1
                continue
            filtered_actions = filter_receiver_system_actions(
                parsed.receiver_actions, system_actions
            )
            perms = parsed.permissions
        vec = vectorize_hybrid(
            perms,
            filtered_actions,
            mldp_vocab=mldp_vocab,
            receiver_vocab=receiver_vocab,
        )
        xs.append(vec)
        ys.append(row.label)
        paths.append(str(row.apk_path))
        sha256s.append(row.sha256)

    if not xs:
        raise RuntimeError("No samples vectorized for split")

    x_arr = np.stack(xs, axis=0).astype(np.float32)
    y_arr = np.array(ys, dtype=np.int64)
    return x_arr, y_arr, paths, sha256s, failures


def _save_feature_shard(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    paths: list[str],
    sha256s: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "paths": paths,
        "sha256": sha256s,
        "feature_dim": int(x.shape[1]),
        "num_samples": int(x.shape[0]),
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

    train_tx, train_labels, train_receiver_lists, train_cache, train_failures = _parse_train_corpus(
        train_rows, system_actions, cfg.paths.failed_apks_log
    )
    if not train_tx:
        raise RuntimeError("All train APKs failed manifest parse")

    selected, trace = run_mldp_selection(
        cfg, train_transactions=train_tx, train_labels=train_labels
    )
    save_mldp_artifacts(cfg.paths.processed, selected, trace)

    receiver_vocab = build_receiver_vocab(train_receiver_lists)
    save_receiver_vocab(cfg.paths.processed / "receiver_action_vocab.json", receiver_vocab)
    save_feature_layout(
        cfg.paths.processed / "feature_layout.json",
        s_size=len(selected),
        r_size=len(receiver_vocab),
    )

    split_failures: Counter[str] = Counter()
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        split_rows = rows_for_split(index_rows, split)
        if not split_rows:
            continue
        x, y, paths, sha256s, failures = _vectorize_split(
            split_rows,
            split=split,
            mldp_vocab=selected,
            receiver_vocab=receiver_vocab,
            system_actions=system_actions,
            failed_log=cfg.paths.failed_apks_log,
            parse_cache=train_cache if split == "train" else None,
        )
        out_path = cfg.paths.processed / f"features_{split}.pt"
        _save_feature_shard(out_path, x, y, paths, sha256s)
        split_failures[split] = failures
        split_counts[split] = int(x.shape[0])

    meta = {
        "preprocessing_version": _git_revision(cfg.root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": cfg.model_id,
        "S": len(selected),
        "R": len(receiver_vocab),
        "d": len(selected) + len(receiver_vocab),
        "train_parse_failures": train_failures,
        "split_counts": split_counts,
        "split_failures": dict(split_failures),
        "mldp_stages": trace.get("stages", {}),
        "fallback_published_list_used": trace.get("fallback_published_list_used", False),
        "system_actions_file": str(cfg.paths.system_actions_file),
        "system_actions_count": len(system_actions),
    }
    meta_path = cfg.paths.processed / "preprocessing_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 manifest feature extraction.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Per-split APK cap (smoke test)")
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    meta = preprocess(cfg, limit=args.limit)

    print(f"|S|={meta['S']}  R={meta['R']}  d={meta['d']}")
    print(f"Split counts: {meta['split_counts']}")
    print(f"MLDP stages: {meta['mldp_stages']}")
    if meta.get("fallback_published_list_used"):
        print("NOTE: published Table I fallback list used for S")
    print(f"Artifacts → {cfg.paths.processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
