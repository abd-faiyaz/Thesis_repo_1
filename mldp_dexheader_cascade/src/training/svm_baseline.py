"""Paper-faithful RBF-SVM + Decision Tree on MLDP block x_S only (offline, M7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.data.store import load_split_shards

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _shard_arrays(shard) -> tuple[np.ndarray, np.ndarray]:
    x = shard.x_s.numpy().astype(np.float32)
    y = shard.y.numpy().astype(np.int64)
    return x, y


def _subset(
    x: np.ndarray,
    y: np.ndarray,
    limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=limit, replace=False)
    return x[idx], y[idx]


def compute_sklearn_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        scores = np.asarray(y_score, dtype=np.float64).ravel()
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def fit_rbf_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float = 10.0,
    gamma: float = 0.1,
) -> SVC:
    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model


def fit_decision_tree(x_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    return model


def evaluate_sklearn_classifier(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x)
    else:
        y_score = None
    return compute_sklearn_metrics(y, y_pred, y_score)


def run_paper_baselines(
    cfg: PipelineConfig,
    *,
    limit: int | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Fit SVM + DT on x_S (train); score val + test."""
    if save:
        ensure_artifact_dirs(cfg)

    baseline_cfg = cfg.baseline
    if not baseline_cfg.get("paper_svm", True) and not baseline_cfg.get("decision_tree", True):
        return {}

    shards = load_split_shards(cfg)
    seed = int(cfg.training.get("seed", 42))

    x_train, y_train = _shard_arrays(shards["train"])
    x_train, y_train = _subset(x_train, y_train, limit, seed)
    x_val, y_val = _shard_arrays(shards["val"])
    x_test, y_test = _shard_arrays(shards["test"])

    result: dict[str, Any] = {}

    if baseline_cfg.get("paper_svm", True):
        svm = fit_rbf_svm(
            x_train,
            y_train,
            C=float(baseline_cfg.get("svm_C", 10.0)),
            gamma=float(baseline_cfg.get("svm_gamma", 0.1)),
        )
        svm_metrics = {
            "model": "svm_rbf",
            "feature_block": "mldp_perms_x_S",
            "kernel": str(baseline_cfg.get("svm_kernel", "rbf")),
            "C": float(baseline_cfg.get("svm_C", 10.0)),
            "gamma": float(baseline_cfg.get("svm_gamma", 0.1)),
            "class_weight": "balanced",
            "n_train": int(len(y_train)),
            "train_limit": limit,
            "val": evaluate_sklearn_classifier(svm, x_val, y_val),
            "test": evaluate_sklearn_classifier(svm, x_test, y_test),
        }
        result["svm_rbf"] = svm_metrics
        if save:
            joblib.dump(svm, cfg.paths.checkpoints / "svm_rbf.joblib")
            (cfg.paths.checkpoints / "svm_metrics.json").write_text(
                json.dumps(svm_metrics, indent=2) + "\n",
                encoding="utf-8",
            )

    if baseline_cfg.get("decision_tree", True):
        dt = fit_decision_tree(x_train, y_train)
        dt_metrics = {
            "model": "decision_tree",
            "feature_block": "mldp_perms_x_S",
            "class_weight": "balanced",
            "n_train": int(len(y_train)),
            "train_limit": limit,
            "val": evaluate_sklearn_classifier(dt, x_val, y_val),
            "test": evaluate_sklearn_classifier(dt, x_test, y_test),
        }
        result["decision_tree"] = dt_metrics
        if save:
            joblib.dump(dt, cfg.paths.checkpoints / "decision_tree.joblib")
            (cfg.paths.checkpoints / "dt_metrics.json").write_text(
                json.dumps(dt_metrics, indent=2) + "\n",
                encoding="utf-8",
            )

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit paper RBF-SVM + DT on x_S.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    result = run_paper_baselines(cfg, limit=args.limit, save=not args.no_save)
    for name, payload in result.items():
        val = payload["val"]
        test = payload["test"]
        print(
            f"{name}: val F1={val['f1']:.4f} | test F1={test['f1']:.4f}"
        )
    if not args.no_save:
        print(f"Saved → {cfg.paths.checkpoints}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
