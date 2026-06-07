"""Evaluate selected MLDP model on configured splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import ensure_artifact_dirs, load_config
from src.constants import DOMAIN_ID, MODEL_ID
from src.data.dataset import stack_split_arrays
from src.models.linear_svm import malware_probabilities
from src.models.tiny_mlp import LinearSigmoidModule, TinyMlpModule
from src.pipeline_integration import (
    build_test_results_payload,
    export_offline_evaluation,
    write_local_metrics_json,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

PRIMARY_TEST_SPLIT = "temporal_holdout"
DEFAULT_SPLITS = ["val", PRIMARY_TEST_SPLIT]


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
        "benign": int(np.sum(y == 0)),
        "malware": int(np.sum(y == 1)),
    }


def _write_test_results(cfg, report: dict, threshold: float) -> Path | None:
    holdout = report["splits"].get(PRIMARY_TEST_SPLIT)
    if holdout is None:
        print(
            f"  no {PRIMARY_TEST_SPLIT} results — test_results.json not written "
            "(re-run without PREPROCESS_LIMIT or include 2022+2023 APKs)"
        )
        return None

    checkpoint = cfg.paths.latest_checkpoint
    payload = build_test_results_payload(
        cfg,
        split_result=holdout,
        threshold=threshold,
        checkpoint_path=checkpoint,
        model_id=MODEL_ID,
        domain=DOMAIN_ID,
        extra={
            "selected_model": report.get("selected_model"),
            "val_f1_at_selection": report.get("val_f1_at_selection"),
        },
    )
    out_path = cfg.paths.artifacts / "metrics" / "test_results.json"
    write_local_metrics_json(out_path, payload)
    print(f"  primary test metrics → {out_path}")

    metrics = holdout["metrics"]
    export_offline_evaluation(
        cfg,
        split="test",
        metrics={k: v for k, v in metrics.items() if v is not None},
        n_samples=holdout["n_samples"],
        threshold=threshold,
        checkpoint_path=checkpoint,
        confusion_matrix=holdout["confusion_matrix"],
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MLDP model.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Default: val (model selection) + temporal_holdout (primary test / 2022+2023).",
    )
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
        "domain": DOMAIN_ID,
        "selected_model": ckpt.get("model_type"),
        "val_f1_at_selection": ckpt.get("val_f1"),
        "primary_test_split": PRIMARY_TEST_SPLIT,
        "threshold": threshold,
        "splits": {},
    }

    for split in args.splits:
        try:
            report["splits"][split] = evaluate_split(cfg, split, threshold)
        except (ValueError, FileNotFoundError) as exc:
            print(f"  skip {split}: {exc}")
            continue

    metrics_dir = cfg.paths.artifacts / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or metrics_dir / "evaluation_results.json"
    write_local_metrics_json(out_path, report)

    for split, result in report["splits"].items():
        m = result["metrics"]
        tag = " [primary test]" if split == PRIMARY_TEST_SPLIT else ""
        print(
            f"{split}{tag}: n={result['n_samples']} "
            f"acc={m['accuracy']:.4f} f1={m['f1']:.4f}"
        )
    print(f"Wrote → {out_path}")
    test_path = _write_test_results(cfg, report, threshold)
    try:
        from src.thesis_archive import after_eval

        paths = [out_path]
        if test_path is not None:
            paths.append(test_path)
        after_eval(*paths)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
