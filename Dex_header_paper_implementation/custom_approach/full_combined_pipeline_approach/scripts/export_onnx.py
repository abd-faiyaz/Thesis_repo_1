#!/usr/bin/env python3
"""Export Pattern A CombinedNet to ONNX bundle (P7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.export.onnx_bundle import export_bundle
from src.pipeline_integration import get_pipeline_settings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Pattern A CombinedNet ONNX bundle.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Default: artifacts/checkpoints/best.pt",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-parity-samples", type=int, default=8)
    parser.add_argument("--skip-verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    settings = get_pipeline_settings(cfg)
    checkpoint = args.checkpoint or cfg.paths.best_checkpoint
    out_dir = args.out_dir or (cfg.root / "artifacts" / "export" / settings.model_id)

    out_dir = export_bundle(
        checkpoint=checkpoint.resolve(),
        out_dir=out_dir,
        config_path=args.config,
        num_parity_samples=args.num_parity_samples,
        skip_verify=args.skip_verify,
    )
    print(f"Export bundle written to {out_dir}")
    try:
        from src.thesis_archive import after_export

        after_export()
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
