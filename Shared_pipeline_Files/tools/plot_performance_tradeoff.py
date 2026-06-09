#!/usr/bin/env python3
"""Pareto scatter: cost_score vs F1 (%)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from plot_thesis_common import add_plot_args  # noqa: E402
from thesis_plot_lib import load_table, plot_performance_tradeoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser)
    args = parser.parse_args()
    path = plot_performance_tradeoff(load_table(args.table), args.out_dir)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
