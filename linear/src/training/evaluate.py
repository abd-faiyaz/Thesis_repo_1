"""Evaluate LinRegDroid on val and test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from shared_calibration import (
    build_val_thresholds_payload,
    find_repo_root,
    format_cascade_band_summary,
    write_split_scores_bundle,
    write_thresholds,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import ensure_artifact_dirs, load_config
from src.constants import DOMAIN_ID, MODEL_ID
from src.data.dataset import stack_split_arrays
from src.data.store import load_manifest
from src.models.mlr import predict_variant
from src.pipeline_integration import (
    build_test_results_payload,
    export_offline_evaluation,
    resolve_split_settings,
    write_local_metrics_json,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

PRIMARY_TEST_SPLIT = "test"
DEFAULT_SPLITS = ["val", PRIMARY_TEST_SPLIT]
SUPPORTED_VARIANTS = ("linregdroid1", "linregdroid2")


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]


def _metrics_for_predictions(
    y: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_variant(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    variant: str,
    threshold: float,
) -> dict[str, Any]:
    y_pred, scores = predict_variant(model, X, variant=variant, threshold=threshold)
    metrics = _metrics_for_predictions(y, y_pred, scores)
    return {
        "variant": variant,
        "metrics": metrics,
        "confusion_matrix": _confusion_matrix(y.astype(np.int64), y_pred),
    }


def evaluate_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    primary_variant: str,
    threshold: float,
    report_both_variants: bool,
) -> dict[str, Any]:
    variants_to_run = list(SUPPORTED_VARIANTS) if report_both_variants else [primary_variant]
    variant_results: dict[str, dict[str, Any]] = {}
    for variant in variants_to_run:
        variant_results[variant] = evaluate_variant(
            model, X, y, variant=variant, threshold=threshold
        )

    primary = variant_results[primary_variant]
    return {
        "n_samples": int(len(y)),
        "primary_variant": primary_variant,
        "metrics": primary["metrics"],
        "confusion_matrix": primary["confusion_matrix"],
        "threshold": threshold,
        "benign": int(np.sum(y == 0)),
        "malware": int(np.sum(y == 1)),
        "variants": variant_results,
    }


def _metrics_thresholds_path(cfg) -> Path:
    return cfg.paths.artifacts / "metrics" / "thresholds.json"


def _metrics_dir(cfg) -> Path:
    return cfg.paths.artifacts / "metrics"


def export_split_scores(
    cfg,
    model,
    split: str,
    *,
    threshold: float,
) -> Path | None:
    try:
        records = load_manifest(cfg.paths.processed, split)
        x, y = stack_split_arrays(cfg.paths.processed, split)
        _y_pred, scores = predict_variant(
            model,
            x,
            variant="linregdroid1",
            threshold=threshold,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"  skip {split} score export: {exc}")
        return None

    out = write_split_scores_bundle(
        model_id=MODEL_ID,
        split=split,
        metrics_dir=_metrics_dir(cfg),
        apk_ids=[record.apk_id for record in records],
        labels=y,
        scores=scores,
        threshold=threshold,
        repo_root=find_repo_root(cfg.root),
    )
    print(f"  {split} scores → {out}")
    return out


def write_val_thresholds(
    cfg,
    model,
    *,
    tune_on_val: bool | None = None,
    calibrate_bands: bool = True,
    out_path: Path | None = None,
) -> dict:
    eval_cfg = cfg.evaluation
    default_threshold = float(eval_cfg.get("threshold", 0.5))
    primary_variant = str(cfg.model.get("variant", "linregdroid1"))
    do_tune = (
        bool(eval_cfg.get("tune_threshold_on_val", True))
        if tune_on_val is None
        else tune_on_val
    )

    records = load_manifest(cfg.paths.processed, "val")
    X_val, y_val = stack_split_arrays(cfg.paths.processed, "val")
    _, val_scores = predict_variant(
        model,
        X_val,
        variant="linregdroid1",
        threshold=default_threshold,
    )
    payload = build_val_thresholds_payload(
        model_id=MODEL_ID,
        y_true=y_val,
        scores=val_scores,
        default=default_threshold,
        tune=do_tune,
        calibrate_bands=calibrate_bands,
        cascade_targets=cfg.raw.get("cascade", {}),
        extra={
            "decision_variant": primary_variant,
            "threshold_variant": "linregdroid1",
            "note": (
                "tuned_val applies to LinRegDroid1 (clamped score); "
                "LinRegDroid2 ignores tuned_val (nearest-class rule)."
            ),
        },
    )
    metrics_dir = cfg.paths.artifacts / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = out_path or _metrics_thresholds_path(cfg)
    write_thresholds(thresholds_path, payload)
    band_summary = format_cascade_band_summary(payload)
    print(
        f"  val-tuned threshold={payload['tuned_val']:.4f} (linregdroid1)"
        + (f"  cascade {band_summary}" if band_summary else "")
        + f" → {thresholds_path}"
    )
    write_split_scores_bundle(
        model_id=MODEL_ID,
        split="val",
        metrics_dir=_metrics_dir(cfg),
        apk_ids=[record.apk_id for record in records],
        labels=y_val,
        scores=val_scores,
        threshold=float(payload["tuned_val"]),
        repo_root=find_repo_root(cfg.root),
    )
    return payload


def _resolve_eval_threshold(
    cfg,
    model,
    *,
    tune_on_val: bool | None = None,
) -> tuple[float, float, str]:
    payload = write_val_thresholds(cfg, model, tune_on_val=tune_on_val)
    primary_variant = str(cfg.model.get("variant", "linregdroid1"))
    return float(payload["default"]), float(payload["tuned_val"]), primary_variant


def run_evaluation(
    cfg,
    splits: list[str],
    *,
    tune_on_val: bool | None = None,
) -> dict:
    model_path = cfg.paths.checkpoints / "linregdroid.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing trained model: {model_path}; run train.py first")
    model = joblib.load(model_path)
    default_threshold, threshold, primary_variant = _resolve_eval_threshold(
        cfg, model, tune_on_val=tune_on_val
    )
    if primary_variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant {primary_variant!r}; use linregdroid1 or linregdroid2")
    report_both = bool(cfg.evaluation.get("report_both_variants", True))
    split_meta = resolve_split_settings(cfg)

    report: dict = {
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "variant": primary_variant,
        "report_both_variants": report_both,
        "train_years": split_meta["train_years"],
        "holdout_years": split_meta["holdout_years"],
        "primary_test_split": PRIMARY_TEST_SPLIT,
        "default_threshold": default_threshold,
        "threshold": threshold,
        "tuned_val": threshold,
        "splits": {},
    }

    for split in splits:
        try:
            X, y = stack_split_arrays(cfg.paths.processed, split)
        except (ValueError, FileNotFoundError) as exc:
            print(f"  skip {split}: {exc}")
            continue
        report["splits"][split] = evaluate_split(
            model,
            X,
            y,
            primary_variant=primary_variant,
            threshold=threshold,
            report_both_variants=report_both,
        )

    return report


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
        extra={"variant": report.get("variant")},
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
    parser = argparse.ArgumentParser(description="Evaluate LinRegDroid on configured splits.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Default: val (threshold tuning) + test (primary / 2022+2023 holdout).",
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default=None,
        help="Override config model.variant for primary metrics (both rules still reported if enabled).",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Skip val max-F1 threshold tuning; use evaluation.threshold from config.",
    )
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    if args.variant is not None:
        cfg.raw.setdefault("model", {})["variant"] = args.variant
    ensure_artifact_dirs(cfg)
    report = run_evaluation(cfg, args.splits, tune_on_val=not args.no_tune_threshold)
    threshold = float(report["threshold"])
    model = joblib.load(cfg.paths.checkpoints / "linregdroid.joblib")

    metrics_dir = cfg.paths.artifacts / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or metrics_dir / "evaluation_results.json"
    write_local_metrics_json(out_path, report)

    for split, result in report["splits"].items():
        m = result["metrics"]
        tag = " [primary test]" if split == PRIMARY_TEST_SPLIT else ""
        print(
            f"{split}{tag} ({result['primary_variant']}): n={result['n_samples']} "
            f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']}"
        )
        if report.get("report_both_variants") and "variants" in result:
            alt = "linregdroid2" if result["primary_variant"] == "linregdroid1" else "linregdroid1"
            alt_m = result["variants"][alt]["metrics"]
            print(
                f"  {alt}: acc={alt_m['accuracy']:.4f} f1={alt_m['f1']:.4f} auc={alt_m['roc_auc']}"
            )
        if split != "val":
            export_split_scores(cfg, model, split, threshold=threshold)
    print(f"Wrote → {out_path}")
    test_path = _write_test_results(cfg, report, threshold)
    try:
        from src.thesis_archive import after_eval

        paths = [out_path, _metrics_thresholds_path(cfg)]
        if test_path is not None:
            paths.append(test_path)
        after_eval(*paths)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
