"""Paper-faithful RBF-SVM on early-concat [H ‖ R] (offline only)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC

from src.config import PipelineConfig
from src.data.store import load_split_shards
from src.training.metrics import compute_metrics, format_metrics


def run_paper_svm_baseline(cfg: PipelineConfig) -> dict:
    if not bool(cfg.baseline.get("paper_svm", True)):
        return {}

    shards = load_split_shards(cfg)
    train = shards["train"]
    val = shards.get("val")
    test = shards.get("test")

    X_train = torch_cat(train.H.numpy(), train.R.numpy())
    y_train = train.y.numpy()

    clf = SVC(
        kernel=str(cfg.baseline.get("svm_kernel", "rbf")),
        C=float(cfg.baseline.get("svm_C", 10.0)),
        gamma=float(cfg.baseline.get("svm_gamma", 0.1)),
        class_weight="balanced",
        probability=True,
    )
    clf.fit(X_train, y_train)

    out: dict = {"train_n": int(X_train.shape[0]), "feature_dim": int(X_train.shape[1])}
    for split_name, shard in (("val", val), ("test", test)):
        if shard is None:
            continue
        X = torch_cat(shard.H.numpy(), shard.R.numpy())
        y = shard.y.numpy()
        scores = clf.predict_proba(X)[:, 1]
        preds = (scores >= 0.5).astype(int)
        metrics = compute_metrics(y, preds, scores)
        out[split_name] = metrics
        print(f"SVM {split_name}: {format_metrics(metrics)}")

    ckpt = cfg.paths.checkpoints / "svm_rbf.joblib"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ckpt)

    metrics_path = cfg.paths.metrics / "svm_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def torch_cat(H: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.concatenate([H, R], axis=1).astype(np.float64)
