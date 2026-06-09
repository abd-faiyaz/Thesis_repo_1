#!/usr/bin/env python3
"""Collect per-model offline test metrics into results/offline/latest/{model_id}.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from plot_registry_lib import (  # noqa: E402
    find_test_results_source,
    load_registry,
    normalize_offline_payload,
    registry_models,
    repo_root,
    validate_registry,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to model_plot_registry.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: Shared_pipeline_Files/results/offline/latest)",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Subset of models (default: all registry models)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate registry and source paths; do not write files",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    registry = load_registry(root)
    errors = validate_registry(root)
    if errors:
        print("Registry validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        if not args.validate_only:
            return 1

    if args.validate_only:
        if errors:
            return 1
        print("Registry OK — all CSV models have offline sources.")
        return 0

    out_dir = (args.out_dir or root / "Shared_pipeline_Files/results/offline/latest").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.model_id) if args.model_id else None
    written = 0
    missing: list[str] = []

    for entry in registry_models(registry):
        model_id = entry["model_id"]
        if wanted is not None and model_id not in wanted:
            continue

        src = find_test_results_source(root, entry)
        if src is None:
            missing.append(model_id)
            continue

        payload = json.loads(src.read_text(encoding="utf-8"))
        normalized = normalize_offline_payload(payload, entry, source_path=src)
        dest = out_dir / f"{model_id}.json"
        dest.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
        print(f"  {model_id} ← {src.relative_to(root)}")
        written += 1

    print(f"Collected {written} file(s) → {out_dir}")
    if missing:
        print("Missing sources:", ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
