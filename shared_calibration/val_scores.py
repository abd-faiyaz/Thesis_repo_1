"""Per-split score dumps for cross-model tier calibration."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "Shared_pipeline_Files").is_dir():
            return candidate
    return start.parent


def default_canonical_val_path(start: Path) -> Path:
    return find_repo_root(start) / "Shared_pipeline_Files/data/splits/canonical_val.txt"


def load_canonical_val_ids(manifest_path: Path | None) -> set[str] | None:
    if manifest_path is None or not Path(manifest_path).is_file():
        return None
    ids: set[str] = set()
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.add(stripped.lower())
    return ids or None


def apk_ids_from_paths(paths: list[str]) -> list[str]:
    return [sha256_file(Path(path)) for path in paths]


def score_rows_from_arrays(
    apk_ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(apk_ids) != labels.shape[0] or labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"apk_ids/labels/scores length mismatch: "
            f"{len(apk_ids)} / {labels.shape[0]} / {scores.shape[0]}"
        )
    return [
        {
            "apk_id": str(apk_ids[i]).lower(),
            "label": int(labels[i]),
            "score": float(scores[i]),
        }
        for i in range(labels.shape[0])
    ]


def filter_rows_to_canonical(
    rows: list[dict[str, Any]],
    canonical_ids: set[str] | None,
) -> list[dict[str, Any]]:
    if not canonical_ids:
        return rows
    return [row for row in rows if str(row["apk_id"]).lower() in canonical_ids]


def metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | None]:
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.size == 0:
        return {"accuracy": None, "f1": None, "roc_auc": None}
    preds = (scores >= threshold).astype(int)
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        out["roc_auc"] = None
    return out


def build_split_scores_payload(
    *,
    model_id: str,
    split: str,
    apk_ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    canonical_manifest: Path | None = None,
) -> dict[str, Any]:
    rows = score_rows_from_arrays(apk_ids, labels, scores)
    canonical_ids = load_canonical_val_ids(canonical_manifest)
    filtered = filter_rows_to_canonical(rows, canonical_ids)
    n_aligned = len(filtered)
    if canonical_ids and not filtered and rows:
        # Val split may not overlap canonical manifest (e.g. temporal holdout).
        filtered = rows
    filtered_labels = np.asarray([row["label"] for row in filtered], dtype=int)
    filtered_scores = np.asarray([row["score"] for row in filtered], dtype=float)
    return {
        "model_id": model_id,
        "split": split,
        "alignment_key": "sha256",
        "canonical_manifest": str(canonical_manifest) if canonical_manifest else None,
        "n_samples": len(rows),
        "n_aligned": n_aligned,
        "threshold": float(threshold),
        "metrics": metrics_at_threshold(filtered_labels, filtered_scores, threshold=threshold),
        "rows": filtered,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def write_split_scores(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def split_scores_filename(split: str) -> str:
    if split == "val":
        return "val_scores.json"
    return f"{split}_scores.json"


def sync_val_scores_to_workspace(
    val_scores_path: Path,
    *,
    model_id: str,
    workspace_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    val_scores_path = Path(val_scores_path)
    if not val_scores_path.is_file():
        return None
    root = repo_root or find_repo_root(val_scores_path)
    out_dir = workspace_dir or (root / "Shared_pipeline_Files/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{model_id}_val_scores.json"
    dest.write_text(val_scores_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def write_split_scores_bundle(
    *,
    model_id: str,
    split: str,
    metrics_dir: Path,
    apk_ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    repo_root: Path | None = None,
    sync_val_to_workspace: bool = True,
) -> Path:
    root = repo_root or find_repo_root(metrics_dir)
    canonical = default_canonical_val_path(root) if split == "val" else None
    payload = build_split_scores_payload(
        model_id=model_id,
        split=split,
        apk_ids=apk_ids,
        labels=labels,
        scores=scores,
        threshold=threshold,
        canonical_manifest=canonical,
    )
    out_path = Path(metrics_dir) / split_scores_filename(split)
    write_split_scores(out_path, payload)
    if split == "val" and sync_val_to_workspace:
        sync_val_scores_to_workspace(out_path, model_id=model_id, repo_root=root)
    return out_path
