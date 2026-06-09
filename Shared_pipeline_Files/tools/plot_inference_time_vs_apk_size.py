#!/usr/bin/env python3
"""Stacked inference-time breakdown for top-5 feasibility-ranked models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from plot_thesis_common import add_plot_args  # noqa: E402
from thesis_plot_lib import (  # noqa: E402
    DEFAULT_EXTENDED_ABSTRACT_OUT,
    load_table,
    plot_inference_time_vs_apk_size,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser)
    parser.set_defaults(out_dir=DEFAULT_EXTENDED_ABSTRACT_OUT)
    args = parser.parse_args()
    path = plot_inference_time_vs_apk_size(load_table(args.table), args.out_dir)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
