#!/usr/bin/env python3
"""Recompute tuned_val + cascade bands on val without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import ensure_artifact_dirs, load_config
from src.data.dataloaders import _loader_settings
from src.training.ablation import build_eval_loader_for_shard, load_test_val_shards
from src.training.evaluate import collect_predictions, load_ablation_model
from src.training.setup import resolve_device
from shared_calibration import build_val_thresholds_payload, format_cascade_band_summary, write_thresholds
from src.training.checkpoint import load_frozen_vocabs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate tuned_val and cascade t_low/t_high on val only."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: artifacts/metrics/thresholds.json",
    )
    parser.add_argument("--no-tune-threshold", action="store_true")
    parser.add_argument("--no-cascade-bands", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    _, val_shard = load_test_val_shards(cfg)
    _, _, layout = load_frozen_vocabs(cfg.paths.processed)
    s_size = int(layout["S"])
    total_dim = int(layout["total"])
    batch_size, num_workers, pin_memory = _loader_settings(cfg)
    device = resolve_device(str(cfg.training.get("device", "cuda")))
    default_threshold = float(cfg.evaluation.get("threshold", 0.5))
    do_tune = bool(cfg.evaluation.get("tune_threshold_on_val", True)) and not args.no_tune_threshold

    val_loader = build_eval_loader_for_shard(
        val_shard,
        mode="full_fusion",
        s_size=s_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model = load_ablation_model(cfg, mode="full_fusion", input_dim=total_dim)
    model.to(device)
    y_true, _, scores = collect_predictions(
        model, val_loader, device, threshold=default_threshold
    )
    payload = build_val_thresholds_payload(
        model_id=cfg.model_id,
        y_true=y_true,
        scores=scores,
        default=default_threshold,
        tune=do_tune,
        calibrate_bands=not args.no_cascade_bands,
        cascade_targets=cfg.raw.get("cascade", {}),
        extra={
            "model_type": str(cfg.classifier.get("deployment", "tiny_mlp")),
            "description": "Predict malware when malware_prob >= tuned_val (val-tuned threshold)",
        },
    )
    out_path = args.out or (cfg.paths.metrics / "thresholds.json")
    write_thresholds(out_path, payload)
    print(f"tuned_val={payload['tuned_val']:.4f}")
    band_summary = format_cascade_band_summary(payload)
    if band_summary:
        print(band_summary)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
