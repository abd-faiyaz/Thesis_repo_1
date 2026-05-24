#!/usr/bin/env python3
"""PyTorch vs ONNX parity check — wired in Phase 4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare parity_samples/ expected vs ONNX output.")
    parser.add_argument("--bundle", type=Path, required=True, help="artifacts/export/<model_id>/ directory")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    manifest = args.bundle / "export_manifest.json"
    samples = args.bundle / "parity_samples"
    if not manifest.is_file():
        print(f"Missing {manifest}", file=sys.stderr)
        return 1
    if not samples.is_dir():
        print(f"Missing {samples}", file=sys.stderr)
        return 1

    meta = json.loads(manifest.read_text(encoding="utf-8"))
    print(f"Parity check stub for model_id={meta.get('model_id')} tolerance={args.tolerance}")
    print("Implement ONNX Runtime comparison in Phase 4 after export scripts land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
