#!/usr/bin/env python3
"""Recompute tuned_val + cascade bands on val without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import ensure_artifact_dirs, load_config
from src.training.evaluate import write_val_thresholds


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
    model_path = cfg.paths.checkpoints / "linregdroid.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    model = joblib.load(model_path)
    payload = write_val_thresholds(
        cfg,
        model,
        tune_on_val=not args.no_tune_threshold,
        calibrate_bands=not args.no_cascade_bands,
        out_path=args.out,
    )
    print(f"tuned_val={payload['tuned_val']:.4f}")
    if "cascade" in payload:
        c = payload["cascade"]
        print(
            f"t_low={c['t_low']:.4f} t_high={c['t_high']:.4f} "
            f"val_step1_exit_rate={c['val_step1_exit_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
