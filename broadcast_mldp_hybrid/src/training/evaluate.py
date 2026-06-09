"""P6 — test-split evaluation from preprocessed features only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.data.dataloaders import _loader_settings
from src.models.factory import build_deployment_model_from_config
from src.training.ablation import (
    ABLATION_MODES,
    build_eval_loader_for_shard,
    load_test_val_shards,
    sliced_input_dim,
)
from src.training.checkpoint import load_best_checkpoint, load_frozen_vocabs, restore_model_weights
from shared_calibration import (
    build_val_thresholds_payload,
    find_repo_root,
    format_cascade_band_summary,
    write_split_scores_bundle,
    write_thresholds,
)

from src.training.metrics import (
    build_confusion_matrix,
    compute_metrics,
    format_metrics,
)
from src.training.setup import resolve_device

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

THESIS_TEMPORAL_F1_NOTE = (
    "Temporal holdout (2022+2023) F1 typically ~75-90%; do not compare to "
    "paper #12 ~97% duplicate-config headline (M9)."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ablation_checkpoint_path(cfg: PipelineConfig, mode: str) -> Path:
    return cfg.paths.checkpoints / f"ablation_{mode}.pt"


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        scores = torch.sigmoid(logits).view(-1).cpu().numpy()
        labels = batch_y.cpu().numpy().astype(int).ravel()
        preds = (scores >= threshold).astype(int)

        y_true_list.append(labels)
        y_pred_list.append(preds)
        y_score_list.append(scores)

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_score_list),
    )


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
) -> dict[str, Any]:
    y_true, y_pred, y_score = collect_predictions(
        model, loader, device, threshold=threshold
    )
    metrics = compute_metrics(y_true, y_pred, y_score)
    return {
        "metrics": metrics,
        "confusion_matrix": build_confusion_matrix(y_true, y_pred),
        "n_samples": int(y_true.shape[0]),
        "threshold": threshold,
    }


def load_ablation_model(
    cfg: PipelineConfig,
    *,
    mode: str,
    input_dim: int,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    path = checkpoint_path or ablation_checkpoint_path(cfg, mode)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model_state" in payload:
            input_dim = int(payload.get("input_dim", input_dim))
            model = build_deployment_model_from_config(cfg, input_dim)
            model.load_state_dict(payload["model_state"])
            return model

    if mode == "full_fusion":
        best_path = cfg.paths.checkpoints / "best.pt"
        payload = load_best_checkpoint(best_path)
        model = build_deployment_model_from_config(cfg, int(payload["d"]))
        restore_model_weights(model, payload)
        return model

    raise FileNotFoundError(
        f"Missing ablation checkpoint for {mode!r}: {path}. Re-run P5 training."
    )


def load_paper_baselines(cfg: PipelineConfig) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    mapping = {
        "svm_rbf": cfg.paths.checkpoints / "svm_metrics.json",
        "decision_tree": cfg.paths.checkpoints / "dt_metrics.json",
    }
    for name, path in mapping.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        test_metrics = payload.get("test", {})
        baselines[name] = {
            "f1": float(test_metrics.get("f1", float("nan"))),
            "accuracy": float(test_metrics.get("accuracy", float("nan"))),
            "roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
        }
    return baselines


def run_test_evaluation(
    cfg: PipelineConfig,
    *,
    checkpoint_path: Path | None = None,
    metrics_out: Path | None = None,
    tune_on_val: bool | None = None,
) -> dict[str, Any]:
    """
    Evaluate on features_test.pt only (no APK parsing).

    Threshold default 0.5; optionally tune on val split first.
    """
    ensure_artifact_dirs(cfg)
    test_shard, val_shard = load_test_val_shards(cfg)
    _, _, layout = load_frozen_vocabs(cfg.paths.processed)
    s_size = int(layout["S"])
    r_size = int(layout["R"])
    total_dim = int(layout["total"])

    batch_size, num_workers, pin_memory = _loader_settings(cfg)
    device = resolve_device(str(cfg.training.get("device", "cuda")))
    eval_cfg = cfg.evaluation
    default_threshold = float(eval_cfg.get("threshold", 0.5))
    do_tune = bool(eval_cfg.get("tune_threshold_on_val", True)) if tune_on_val is None else tune_on_val

    val_loader_full = build_eval_loader_for_shard(
        val_shard,
        mode="full_fusion",
        s_size=s_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader_full = build_eval_loader_for_shard(
        test_shard,
        mode="full_fusion",
        s_size=s_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    full_model = load_ablation_model(cfg, mode="full_fusion", input_dim=total_dim)
    full_model.to(device)

    y_true_val, _, val_scores = collect_predictions(
        full_model, val_loader_full, device, threshold=default_threshold
    )
    thresholds_payload = build_val_thresholds_payload(
        model_id=cfg.model_id,
        y_true=y_true_val,
        scores=val_scores,
        default=default_threshold,
        tune=do_tune,
        calibrate_bands=True,
        cascade_targets=cfg.raw.get("cascade", {}),
        extra={
            "model_type": str(cfg.classifier.get("deployment", "tiny_mlp")),
            "description": "Predict malware when malware_prob >= tuned_val (val-tuned threshold)",
        },
    )
    tuned_threshold = float(thresholds_payload["tuned_val"])
    thresholds_path = cfg.paths.metrics / "thresholds.json"
    write_thresholds(thresholds_path, thresholds_payload)
    band_summary = format_cascade_band_summary(thresholds_payload)
    if band_summary:
        print(f"  cascade bands: {band_summary}")

    repo_root = find_repo_root(cfg.root)
    val_score_path = write_split_scores_bundle(
        model_id=cfg.model_id,
        split="val",
        metrics_dir=cfg.paths.metrics,
        apk_ids=val_shard.sha256,
        labels=y_true_val,
        scores=val_scores,
        threshold=tuned_threshold,
        repo_root=repo_root,
    )
    print(f"  val scores → {val_score_path}")

    threshold = tuned_threshold if do_tune else default_threshold
    y_test, _, test_scores = collect_predictions(
        full_model, test_loader_full, device, threshold=threshold
    )
    test_score_path = write_split_scores_bundle(
        model_id=cfg.model_id,
        split="test",
        metrics_dir=cfg.paths.metrics,
        apk_ids=test_shard.sha256,
        labels=y_test,
        scores=test_scores,
        threshold=threshold,
        repo_root=repo_root,
        sync_val_to_workspace=False,
    )
    print(f"  test scores → {test_score_path}")

    primary = evaluate_model_on_loader(
        full_model, test_loader_full, device, threshold=threshold
    )

    ablation_payload: dict[str, dict[str, float]] = {}
    for mode in ABLATION_MODES:
        input_dim = sliced_input_dim(total_dim, mode=mode, s_size=s_size)
        try:
            model = load_ablation_model(cfg, mode=mode, input_dim=input_dim)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            continue
        model.to(device)
        test_loader = build_eval_loader_for_shard(
            test_shard,
            mode=mode,
            s_size=s_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        result = evaluate_model_on_loader(
            model, test_loader, device, threshold=threshold
        )
        ablation_payload[mode] = result["metrics"]

    split_cfg = cfg.splits
    payload: dict[str, Any] = {
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "split": "test",
        "train_years": split_cfg.get("train_years", [2020, 2021]),
        "test_years": split_cfg.get("holdout_years", [2022, 2023]),
        "n_samples": primary["n_samples"],
        "feature_dims": {"S": s_size, "R": r_size, "total": total_dim},
        "metrics": primary["metrics"],
        "confusion_matrix": primary["confusion_matrix"],
        "threshold": threshold,
        "thresholds": thresholds_payload,
        "ablations": ablation_payload,
        "paper_baselines": load_paper_baselines(cfg),
        "thesis_guidance_m9": THESIS_TEMPORAL_F1_NOTE,
        "evaluated_at": _utc_now(),
        "data_source": str(test_shard.source_path),
    }

    out_path = metrics_out or (cfg.paths.metrics / "test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    try:
        from src.thesis_archive import after_eval

        after_eval(out_path, thresholds_path)
    except ImportError:
        pass

    print(f"Test evaluation ({primary['n_samples']} samples from features_test.pt)")
    print(f"  threshold={threshold:.4f}  {format_metrics(primary['metrics'])}")
    print(f"  confusion_matrix={primary['confusion_matrix']}")
    if ablation_payload:
        print("  ablation test F1:")
        for mode, metrics in ablation_payload.items():
            print(f"    {mode}: F1={metrics['f1']:.4f}")
    print(f"  metrics → {out_path}")
    print(f"  thresholds → {thresholds_path}")
    print(f"  note: {THESIS_TEMPORAL_F1_NOTE}")

    return {**payload, "metrics_path": str(out_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate broadcast+MLDP hybrid on temporal test split (P6)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Unused; ablation_*.pt used")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Default: artifacts/metrics/test_results.json",
    )
    parser.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Use fixed evaluation.threshold (default 0.5)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_test_evaluation(
        cfg,
        metrics_out=args.metrics_out,
        tune_on_val=not args.no_tune_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
