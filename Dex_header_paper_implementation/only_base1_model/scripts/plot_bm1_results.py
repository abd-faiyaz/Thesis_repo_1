#!/usr/bin/env python3
"""Backward-compatible wrapper — delegates to shared plot_thesis_results.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent.parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_thesis_results import generate_figures  # noqa: E402
from thesis_run_archive import resolve_run_id  # noqa: E402
from thesis_run_logging import ARCHIVE_PROFILES  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate BM1 thesis figures (wrapper around plot_thesis_results.py)."
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="output_archives/<run_id>/ (default: LATEST_RUN.txt)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "checkpoints" / "latest_checkpoint.pth",
        help="Unused; kept for CLI compatibility",
    )
    parser.add_argument("--config", type=Path, default=None, help="Unused; kept for compatibility")
    parser.add_argument("--skip-inference", action="store_true", help="Unused; kept for compatibility")
    parser.add_argument("--num-workers", type=int, default=0, help="Unused; kept for compatibility")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = ARCHIVE_PROFILES["mlp_header"]

    if args.archive_dir is not None:
        run_id = args.archive_dir.name
    else:
        run_id = resolve_run_id(ROOT, profile, None)

    paths = generate_figures(ROOT, profile, run_id)
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
