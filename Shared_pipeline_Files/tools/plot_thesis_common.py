#!/usr/bin/env python3
"""CLI argument helpers for individual thesis plot scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis_plot_lib import DEFAULT_OUT, DEFAULT_TABLE


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help="plot_metrics_table.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for figures",
    )
