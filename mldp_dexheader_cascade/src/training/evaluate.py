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
from src.models import (
    DeployedMlpHeaderRef,
    build_fused_mlp,
    build_fused_mlp_from_config,
    build_mldp_logistic,
)
from src.models.mldp_logistic import MldpStage1TinyMlp
from src.training.ablation import ABLATION_MODES, build_eval_loader_for_shard, load_val_test_shards
from shared_calibration import find_repo_root, write_split_scores_bundle

from src.training.calibrate_thresholds import (
    build_thresholds_payload,
    calibrate_cascade_thresholds,
    false_alarm_rate,
    false_omission_rate,
    write_thresholds,
)
from src.training.checkpoint import load_checkpoint, load_frozen_artifacts, restore_model_weights
from src.training.dex_header_eval import eval_deployed_dex_header_from_config
from src.training.metrics import (
    build_confusion_matrix,
    compute_metrics,
    format_metrics,
    tune_threshold,
)
from src.training.setup import resolve_device

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

THESIS_TEMPORAL_F1_NOTE = (
    "Temporal holdout (2022+2023) F1 reflects train-on-2020-2021 protocol; "
    "Mode B exit rate is measured, not assumed."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@torch.no_grad()
def collect_logits_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        scores = torch.sigmoid(logits).view(-1).cpu().numpy()
        labels = batch_y.cpu().numpy().astype(int).ravel()
        y_true_list.append(labels)
        y_score_list.append(scores)
    return np.concatenate(y_true_list), np.concatenate(y_score_list)


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_score)
    return {
        "metrics": metrics,
        "confusion_matrix": build_confusion_matrix(y_true, y_pred),
        "n_samples": int(y_true.shape[0]),
        "threshold": threshold,
    }


def load_mode_a_model(cfg: PipelineConfig, device: torch.device) -> nn.Module:
    path = cfg.paths.checkpoints / "mode_a_best.pt"
    payload = load_checkpoint(path)
    model = build_fused_mlp_from_config(cfg, int(payload["d"]))
    restore_model_weights(model, payload)
    return model.to(device)


def load_stage1_model(cfg: PipelineConfig, device: torch.device) -> nn.Module:
    path = cfg.paths.checkpoints / "stage1_best.pt"
    payload = load_checkpoint(path)
    head = str(payload.get("head", "logistic"))
    s_dim = int(payload["S_dim"])
    if head == "tiny_mlp":
        model = MldpStage1TinyMlp(
            s_dim,
            hidden_dim=int(cfg.model.get("mode_b_stage1_mlp_hidden", 32)),
        )
    else:
        model = build_mldp_logistic(s_dim)
    restore_model_weights(model, payload)
    return model.to(device)


def load_ablation_model(
    cfg: PipelineConfig,
    *,
    mode: str,
    input_dim: int,
) -> nn.Module:
    path = cfg.paths.checkpoints / f"ablation_{mode}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ablation checkpoint: {path}")
    payload = load_checkpoint(path)
    dim = int(payload.get("input_dim", input_dim))
    if mode == "mldp_perms_only":
        hidden = int(cfg.model.get("mode_b_stage1_mlp_hidden", 32))
        model = build_fused_mlp(dim, hidden_dim=hidden)
    else:
        model = build_fused_mlp_from_config(cfg, dim)
    restore_model_weights(model, payload)
    return model


def cascade_end_to_end(
    y_true: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    *,
    t_low: float,
    t_high: float,
    stage2_threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).ravel()
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    s2 = np.asarray(s2, dtype=np.float64).ravel()

    preds = np.zeros_like(y_true)
    early_exit = np.zeros(y_true.shape[0], dtype=bool)
    used_stage2 = np.zeros(y_true.shape[0], dtype=bool)
    final_scores = np.zeros_like(s1)

    for i in range(y_true.shape[0]):
        if s1[i] <= t_low:
            preds[i] = 0
            early_exit[i] = True
            final_scores[i] = s1[i]
        elif s1[i] >= t_high:
            preds[i] = 1
            early_exit[i] = True
            final_scores[i] = s1[i]
        else:
            used_stage2[i] = True
            final_scores[i] = s2[i]
            preds[i] = 1 if s2[i] >= stage2_threshold else 0

    metrics = compute_metrics(y_true, preds, final_scores)
    return {
        "metrics": metrics,
        "confusion_matrix": build_confusion_matrix(y_true, preds),
        "step1_exit_rate": float(early_exit.mean()),
        "stage2_invocation_rate": float(used_stage2.mean()),
        "false_omission_rate_at_t_low": false_omission_rate(y_true, s1, t_low),
        "false_alarm_rate_at_t_high": false_alarm_rate(y_true, s1, t_high),
        "end_to_end_f1": float(metrics["f1"]),
        "end_to_end_acc": float(metrics["accuracy"]),
    }


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
    metrics_out: Path | None = None,
    tune_on_val: bool | None = None,
) -> dict[str, Any]:
    ensure_artifact_dirs(cfg)
    val_shard, test_shard = load_val_test_shards(cfg)
    _, layout = load_frozen_artifacts(cfg.paths.processed)
    s_size = int(layout["S"])
    h_size = int(layout["H"])
    d_size = int(layout["d"])

    batch_size, num_workers, pin_memory = _loader_settings(cfg)
    device = resolve_device(str(cfg.training.get("device", "cuda")))
    eval_cfg = cfg.evaluation
    default_threshold = float(eval_cfg.get("threshold", 0.5))
    do_tune = (
        bool(eval_cfg.get("tune_threshold_on_val", True))
        if tune_on_val is None
        else tune_on_val
    )

    mode_a = load_mode_a_model(cfg, device)
    stage1 = load_stage1_model(cfg, device)
    stage2_ref = DeployedMlpHeaderRef.from_config(cfg)

    val_loader_x = build_eval_loader_for_shard(
        val_shard, mode="mode_a_fusion", batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader_s = build_eval_loader_for_shard(
        val_shard, mode="mldp_perms_only", batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader_x = build_eval_loader_for_shard(
        test_shard, mode="mode_a_fusion", batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader_s = build_eval_loader_for_shard(
        test_shard, mode="mldp_perms_only", batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    y_val, mode_a_val_scores = collect_logits_scores(mode_a, val_loader_x, device)
    y_val_s, stage1_val_scores = collect_logits_scores(stage1, val_loader_s, device)
    if not np.array_equal(y_val, y_val_s):
        raise RuntimeError("Val label mismatch between fusion and perm loaders")

    mode_a_threshold = default_threshold
    if do_tune:
        mode_a_threshold = tune_threshold(y_val, mode_a_val_scores)

    cascade_cal = calibrate_cascade_thresholds(
        y_val_s,
        stage1_val_scores,
        target_false_omission_rate=float(cfg.cascade.get("target_false_omission_rate", 0.02)),
        target_false_alarm_at_thigh=float(cfg.cascade.get("target_false_alarm_at_thigh", 0.02)),
    )

    thresholds_payload = build_thresholds_payload(
        cfg,
        mode_a_default=default_threshold,
        mode_a_tuned=mode_a_threshold,
        cascade=cascade_cal,
    )
    thresholds_path = cfg.paths.metrics / "thresholds.json"
    write_thresholds(thresholds_path, thresholds_payload)

    repo_root = find_repo_root(cfg.root)
    val_score_path = write_split_scores_bundle(
        model_id=cfg.model_id,
        split="val",
        metrics_dir=cfg.paths.metrics,
        apk_ids=val_shard.sha256,
        labels=y_val_s,
        scores=stage1_val_scores,
        threshold=mode_a_threshold,
        repo_root=repo_root,
    )
    print(f"  val scores (stage1) → {val_score_path}")

    y_test, mode_a_test_scores = collect_logits_scores(mode_a, test_loader_x, device)
    y_test_s, stage1_test_scores = collect_logits_scores(stage1, test_loader_s, device)
    if not np.array_equal(y_test, y_test_s):
        raise RuntimeError("Test label mismatch between fusion and perm loaders")

    test_score_path = write_split_scores_bundle(
        model_id=cfg.model_id,
        split="test",
        metrics_dir=cfg.paths.metrics,
        apk_ids=test_shard.sha256,
        labels=y_test_s,
        scores=stage1_test_scores,
        threshold=mode_a_threshold,
        repo_root=repo_root,
        sync_val_to_workspace=False,
    )
    print(f"  test scores (stage1) → {test_score_path}")

    mode_a_result = evaluate_scores(y_test, mode_a_test_scores, threshold=mode_a_threshold)

    s2_test = stage2_ref.score(test_shard.h.numpy().astype(np.float32))
    mode_b_result = cascade_end_to_end(
        y_test_s,
        stage1_test_scores,
        s2_test,
        t_low=cascade_cal["stage1_t_low"],
        t_high=cascade_cal["stage1_t_high"],
        stage2_threshold=mode_a_threshold,
    )

    ablation_payload: dict[str, dict[str, float]] = {}
    for mode in ABLATION_MODES:
        try:
            model = load_ablation_model(cfg, mode=mode, input_dim=d_size if mode == "mode_a_fusion" else s_size)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            continue
        model.to(device)
        loader = build_eval_loader_for_shard(
            test_shard,
            mode=mode,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        y_ab, scores_ab = collect_logits_scores(model, loader, device)
        result = evaluate_scores(y_ab, scores_ab, threshold=mode_a_threshold)
        ablation_payload[mode] = result["metrics"]

    dex_test_metrics = eval_deployed_dex_header_from_config(
        cfg, test_shard, threshold=mode_a_threshold
    )
    ablation_payload["dex_header_only"] = dex_test_metrics
    if "mode_a_fusion" in ablation_payload:
        ablation_payload["mode_a_fusion"] = mode_a_result["metrics"]

    split_cfg = cfg.splits
    payload: dict[str, Any] = {
        "model_id": cfg.model_id,
        "split": "test",
        "train_years": split_cfg.get("train_years", [2020, 2021]),
        "test_years": split_cfg.get("test_years", [2022, 2023]),
        "n_samples": mode_a_result["n_samples"],
        "feature_dims": {"S": s_size, "H": h_size, "d": d_size},
        "mode_a": {
            **mode_a_result["metrics"],
            "confusion_matrix": mode_a_result["confusion_matrix"],
            "threshold": mode_a_threshold,
        },
        "mode_b": {
            "stage1_t_low": cascade_cal["stage1_t_low"],
            "stage1_t_high": cascade_cal["stage1_t_high"],
            "step1_exit_rate": mode_b_result["step1_exit_rate"],
            "stage2_invocation_rate": mode_b_result["stage2_invocation_rate"],
            "false_omission_rate_at_t_low": mode_b_result["false_omission_rate_at_t_low"],
            "false_alarm_rate_at_t_high": mode_b_result["false_alarm_rate_at_t_high"],
            "end_to_end_f1": mode_b_result["end_to_end_f1"],
            "end_to_end_acc": mode_b_result["end_to_end_acc"],
            "confusion_matrix": mode_b_result["confusion_matrix"],
            "val_calibration": {
                "val_step1_exit_rate": cascade_cal["val_step1_exit_rate"],
                "val_false_omission_rate_at_t_low": cascade_cal["val_false_omission_rate_at_t_low"],
                "val_false_alarm_rate_at_t_high": cascade_cal["val_false_alarm_rate_at_t_high"],
            },
        },
        "ablations": {
            "mldp_perms_only": ablation_payload.get("mldp_perms_only", {}),
            "dex_header_only": ablation_payload.get("dex_header_only", {}),
            "mode_a_fusion": ablation_payload.get("mode_a_fusion", {}),
        },
        "paper_baselines": load_paper_baselines(cfg),
        "thresholds": thresholds_payload,
        "thesis_guidance": THESIS_TEMPORAL_F1_NOTE,
        "evaluated_at": _utc_now(),
        "data_source": str(test_shard.source_path),
    }

    out_path = metrics_out or (cfg.paths.metrics / "test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Test evaluation ({mode_a_result['n_samples']} samples from features_test.pt)")
    print(f"  Mode A: threshold={mode_a_threshold:.4f}  {format_metrics(mode_a_result['metrics'])}")
    print(
        f"  Mode B: t_low={cascade_cal['stage1_t_low']:.4f} t_high={cascade_cal['stage1_t_high']:.4f} "
        f"step1_exit_rate={mode_b_result['step1_exit_rate']:.3f} "
        f"end_to_end_F1={mode_b_result['end_to_end_f1']:.4f}"
    )
    print("  Ablation test F1:")
    for key in ("mldp_perms_only", "dex_header_only", "mode_a_fusion"):
        metrics = payload["ablations"].get(key, {})
        if metrics:
            print(f"    {key}: F1={metrics.get('f1', float('nan')):.4f}")
    print(f"  metrics → {out_path}")
    print(f"  thresholds → {thresholds_path}")

    return {**payload, "metrics_path": str(out_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mldp_dexheader_cascade on test split (P6).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--no-tune-threshold", action="store_true")
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
