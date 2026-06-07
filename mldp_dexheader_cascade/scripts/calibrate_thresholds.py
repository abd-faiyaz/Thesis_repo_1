#!/usr/bin/env python3
"""P6 — calibrate Mode A + Mode B cascade thresholds on val only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import ensure_artifact_dirs, load_config
from src.data.dataloaders import _loader_settings
from src.training.ablation import build_eval_loader_for_shard, load_val_test_shards
from src.training.calibrate_thresholds import (
    build_thresholds_payload,
    calibrate_cascade_thresholds,
    write_thresholds,
)
from src.training.evaluate import collect_logits_scores, load_mode_a_model, load_stage1_model
from src.training.metrics import tune_threshold
from src.training.setup import resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate thresholds on val holdout (M11)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: artifacts/metrics/thresholds.json",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    val_shard, _ = load_val_test_shards(cfg)

    batch_size, num_workers, pin_memory = _loader_settings(cfg)
    device = resolve_device(str(cfg.training.get("device", "cuda")))
    default_threshold = float(cfg.evaluation.get("threshold", 0.5))
    do_tune = bool(cfg.evaluation.get("tune_threshold_on_val", True))

    mode_a = load_mode_a_model(cfg, device)
    stage1 = load_stage1_model(cfg, device)

    val_loader_x = build_eval_loader_for_shard(
        val_shard, mode="mode_a_fusion", batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader_s = build_eval_loader_for_shard(
        val_shard, mode="mldp_perms_only", batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    y_val, mode_a_scores = collect_logits_scores(mode_a, val_loader_x, device)
    y_val_s, stage1_scores = collect_logits_scores(stage1, val_loader_s, device)
    if not np.array_equal(y_val, y_val_s):
        raise RuntimeError("Val label mismatch")

    mode_a_tuned = (
        tune_threshold(y_val, mode_a_scores) if do_tune else default_threshold
    )
    cascade = calibrate_cascade_thresholds(
        y_val_s,
        stage1_scores,
        target_false_omission_rate=float(cfg.cascade.get("target_false_omission_rate", 0.02)),
        target_false_alarm_at_thigh=float(cfg.cascade.get("target_false_alarm_at_thigh", 0.02)),
    )

    payload = build_thresholds_payload(
        cfg,
        mode_a_default=default_threshold,
        mode_a_tuned=mode_a_tuned,
        cascade=cascade,
    )
    out_path = args.out or (cfg.paths.metrics / "thresholds.json")
    write_thresholds(out_path, payload)

    print(f"Mode A threshold (val-tuned): {mode_a_tuned:.4f}")
    print(
        f"Mode B t_low={cascade['stage1_t_low']:.4f} t_high={cascade['stage1_t_high']:.4f} "
        f"val_step1_exit_rate={cascade['val_step1_exit_rate']:.3f}"
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
