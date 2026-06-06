"""Evaluate selected MLDP model on configured splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import ensure_artifact_dirs, load_config
from src.constants import MODEL_ID
from src.data.dataset import stack_split_arrays
from src.models.linear_svm import malware_probabilities
from src.models.tiny_mlp import LinearSigmoidModule, TinyMlpModule

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def _load_probs(cfg, X: np.ndarray) -> tuple[np.ndarray, str]:
    ckpt = torch.load(cfg.paths.latest_checkpoint, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "linear_svc")
    feature_dim = int(ckpt["feature_dim"])

    if model_type == "tiny_mlp":
        hidden_dim = int(ckpt.get("hidden_dim", 32))
        model = TinyMlpModule(feature_dim, hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            probs = model(torch.from_numpy(X.astype(np.float32))).numpy().reshape(-1)
        return probs, model_type

    joblib_path = cfg.paths.checkpoints / "model.joblib"
    if joblib_path.is_file():
        svc = joblib.load(joblib_path)
        return malware_probabilities(svc, X), "linear_svc"

    module = LinearSigmoidModule(feature_dim)
    module.load_state_dict(ckpt["model_state_dict"])
    module.eval()
    with torch.no_grad():
        probs = module(torch.from_numpy(X.astype(np.float32))).numpy().reshape(-1)
    return probs, model_type


def evaluate_split(cfg, split: str, threshold: float) -> dict:
    X, y = stack_split_arrays(cfg.paths.processed, split)
    probs, model_type = _load_probs(cfg, X)
    y_pred = (probs >= threshold).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    else:
        metrics["roc_auc"] = None
    return {
        "n_samples": int(len(y)),
        "metrics": metrics,
        "confusion_matrix": _confusion_matrix(y.astype(np.int64), y_pred),
        "model_type": model_type,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MLDP model.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "dev_test"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    threshold = float(cfg.evaluation.get("threshold", 0.5))

    ckpt = torch.load(cfg.paths.latest_checkpoint, map_location="cpu", weights_only=False)
    report = {
        "model_id": MODEL_ID,
        "selected_model": ckpt.get("model_type"),
        "val_f1_at_selection": ckpt.get("val_f1"),
        "threshold": threshold,
        "splits": {},
    }

    for split in args.splits:
        report["splits"][split] = evaluate_split(cfg, split, threshold)

    metrics_dir = cfg.paths.artifacts / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or metrics_dir / "evaluation_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for split, result in report["splits"].items():
        m = result["metrics"]
        print(f"{split}: n={result['n_samples']} acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    print(f"Wrote → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
