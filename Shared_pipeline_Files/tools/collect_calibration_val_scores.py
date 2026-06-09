#!/usr/bin/env python3
"""Collect per-model val_scores.json files into Shared_pipeline_Files/calibration/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MODEL_IDS = [
    "mldp_pruned_permission",
    "broadcast_mldp_hybrid",
    "mldp_dexheader_cascade",
    "dexheader_broadcast_fusion",
    "linregdroid_permission",
    "mlp_header",
    "early_fusion_dex_manifest",
    "dual_branch_dex_manifest",
    "bytecnn",
    "manifest_xgb",
]

SEARCH_PATHS = {
    "mldp_pruned_permission": ["permission_extractor/artifacts/metrics/val_scores.json"],
    "broadcast_mldp_hybrid": ["broadcast_mldp_hybrid/artifacts/metrics/val_scores.json"],
    "mldp_dexheader_cascade": ["mldp_dexheader_cascade/artifacts/metrics/val_scores.json"],
    "linregdroid_permission": ["linear/artifacts/metrics/val_scores.json"],
    "mlp_header": [
        "Dex_header_paper_implementation/only_base1_model/artifacts/metrics/val_scores.json"
    ],
    "early_fusion_dex_manifest": [
        "Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/metrics/val_scores.json"
    ],
    "dual_branch_dex_manifest": [
        "Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/metrics/val_scores.json"
    ],
    "dexheader_broadcast_fusion": [
        "dexheader_broadcast_fusion/artifacts/metrics/val_scores.json"
    ],
    "bytecnn": ["legacy_models/artifacts/bytecnn/metrics/val_scores.json"],
    "manifest_xgb": ["legacy_models/artifacts/manifest_xgb/metrics/val_scores.json"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect val_scores.json into calibration workspace.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_repo_root() / "Shared_pipeline_Files/calibration",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Subset of models to collect (default: all known thesis models).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 when some models lack val_scores.json (warn only).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    out_dir = args.workspace.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_ids = args.model_id or MODEL_IDS
    copied = 0
    missing: list[str] = []

    for model_id in model_ids:
        rel_paths = SEARCH_PATHS.get(model_id, [])
        src = next((root / rel for rel in rel_paths if (root / rel).is_file()), None)
        if src is None:
            missing.append(model_id)
            continue
        dest = out_dir / f"{model_id}_val_scores.json"
        shutil.copy2(src, dest)
        print(f"  {model_id} ← {src.relative_to(root)}")
        copied += 1

    print(f"Collected {copied}/{len(model_ids)} files → {out_dir}")
    if missing:
        print("Missing:", ", ".join(missing))
        if args.allow_missing:
            print("(allowed — run model evaluate.py to generate val_scores.json)")
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
