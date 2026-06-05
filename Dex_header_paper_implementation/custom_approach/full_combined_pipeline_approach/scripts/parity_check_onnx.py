#!/usr/bin/env python3
"""PyTorch vs ONNX parity on Pattern A export bundle (P8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.export.onnx_bundle import DEFAULT_TOLERANCE, run_parity_check, write_parity_outputs
from src.pipeline_integration import get_pipeline_settings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pattern A PyTorch vs ONNX parity check.")
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    settings = get_pipeline_settings(cfg)
    bundle = args.bundle or (cfg.root / "artifacts" / "export" / settings.model_id)
    checkpoint = args.checkpoint or cfg.paths.best_checkpoint

    report = run_parity_check(
        bundle,
        checkpoint=checkpoint.resolve(),
        tolerance=args.tolerance,
        config_path=args.config,
    )
    report_path = write_parity_outputs(
        report,
        bundle,
        local_parity_dir=cfg.root / "artifacts" / "parity",
    )

    status = "PASS" if report["passed"] else "FAIL"
    pt_onnx = report["pytorch_vs_onnx"]
    print(f"Parity {status}: max_abs_diff={pt_onnx['max_abs_diff']:.2e} (tolerance {args.tolerance})")
    print(f"  report → {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
