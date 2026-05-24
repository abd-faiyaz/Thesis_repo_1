#!/usr/bin/env python3
"""Export Pattern A CombinedNet to ONNX (Phase 4 — run after training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Pattern A model to ONNX bundle.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Default: artifacts/checkpoints/best.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: artifacts/export/pattern_a_combined/",
    )
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    from src.config import load_config
    from src.pipeline_integration import find_repo_root, get_pipeline_settings

    cfg = load_config(args.config)
    settings = get_pipeline_settings(cfg)
    out_dir = args.out_dir or (cfg.root / "artifacts" / "export" / settings.model_id)
    ckpt = args.checkpoint or cfg.paths.best_checkpoint

    print("Pattern A ONNX export (skeleton)")
    print(f"  model_id: {settings.model_id}")
    print(f"  checkpoint: {ckpt}")
    print(f"  out_dir: {out_dir}")
    print("  Implement torch.onnx.export in Phase 4 after full-dataset training.")
    repo = find_repo_root(cfg.root)
    if repo:
        assets = repo / "vigidroid" / "app" / "src" / "main" / "assets" / "models"
        print(f"  Then copy bundle to: {assets / settings.model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
