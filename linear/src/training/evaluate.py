"""Evaluate LinRegDroid on val / dev_test / temporal_holdout splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import ensure_artifact_dirs, load_config
from src.constants import MODEL_ID
from src.data.dataset import stack_split_arrays
from src.models.mlr import linregdroid1_predict, raw_linear_scores

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def evaluate_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
) -> dict:
    raw = raw_linear_scores(model, X)
    malware_prob = np.clip(raw, 0.0, 1.0)
    y_pred = linregdroid1_predict(malware_prob, threshold=threshold)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, malware_prob))
    else:
        metrics["roc_auc"] = None
    return {
        "n_samples": int(len(y)),
        "metrics": metrics,
        "confusion_matrix": _confusion_matrix(y.astype(np.int64), y_pred),
        "threshold": threshold,
        "benign": int(np.sum(y == 0)),
        "malware": int(np.sum(y == 1)),
    }


def run_evaluation(cfg, splits: list[str]) -> dict:
    model_path = cfg.paths.checkpoints / "linregdroid.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing trained model: {model_path}; run train.py first")
    model = joblib.load(model_path)
    threshold = float(cfg.evaluation.get("threshold", 0.5))
    pre = cfg.preprocessing

    report: dict = {
        "model_id": MODEL_ID,
        "variant": cfg.model.get("variant", "linregdroid1"),
        "development_years": pre.get("development_years", [2020, 2021]),
        "temporal_holdout_years": pre.get("temporal_holdout_years", [2022, 2023]),
        "threshold": threshold,
        "splits": {},
    }

    for split in splits:
        X, y = stack_split_arrays(cfg.paths.processed, split)
        report["splits"][split] = evaluate_split(model, X, y, threshold=threshold)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LinRegDroid on configured splits.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "dev_test"],
        help="Default: val + dev_test from 2020/2021. Add temporal_holdout for 2022/2023.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    report = run_evaluation(cfg, args.splits)

    metrics_dir = cfg.paths.artifacts / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or metrics_dir / "evaluation_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for split, result in report["splits"].items():
        m = result["metrics"]
        print(
            f"{split}: n={result['n_samples']} "
            f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
            f"auc={m['roc_auc']}"
        )
    print(f"Wrote → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
