#!/usr/bin/env python3
"""Stamp val_f1 / val_accuracy from val_scores.json into export_manifest.json (C5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared_calibration.manifest_metrics import stamp_export_manifest  # noqa: E402

MODEL_BUNDLES: dict[str, dict[str, str]] = {
    "mldp_pruned_permission": {
        "val_scores": "permission_extractor/artifacts/metrics/val_scores.json",
        "export_manifest": "permission_extractor/artifacts/export/mldp_pruned_permission/export_manifest.json",
        "android_manifest": "vigidroid/app/src/main/assets/models/mldp_pruned_permission/export_manifest.json",
    },
    "broadcast_mldp_hybrid": {
        "val_scores": "broadcast_mldp_hybrid/artifacts/metrics/val_scores.json",
        "export_manifest": "broadcast_mldp_hybrid/artifacts/export/broadcast_mldp_hybrid/export_manifest.json",
        "android_manifest": "vigidroid/app/src/main/assets/models/broadcast_mldp_hybrid/export_manifest.json",
    },
    "mldp_dexheader_cascade": {
        "val_scores": "mldp_dexheader_cascade/artifacts/metrics/val_scores.json",
        "export_manifest": "mldp_dexheader_cascade/artifacts/export/mldp_dexheader_cascade/mode_b/export_manifest.json",
        "android_manifest": "vigidroid/app/src/main/assets/models/mldp_dexheader_cascade/mode_b/export_manifest.json",
    },
    "mlp_header": {
        "val_scores": "Dex_header_paper_implementation/only_base1_model/artifacts/metrics/val_scores.json",
        "export_manifest": "Dex_header_paper_implementation/only_base1_model/artifacts/export/mlp_header/export_manifest.json",
        "android_manifest": "vigidroid/app/src/main/assets/models/mlp_header/export_manifest.json",
    },
    "early_fusion_dex_manifest": {
        "val_scores": "Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/metrics/val_scores.json",
        "export_manifest": "Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/export/early_fusion_dex_manifest/export_manifest.json",
        "android_manifest": "vigidroid/app/src/main/assets/models/early_fusion_dex_manifest/export_manifest.json",
    },
    "dual_branch_dex_manifest": {
        "val_scores": "Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/metrics/val_scores.json",
        "export_manifest": "Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/export/dual_branch_dex_manifest/export_manifest.json",
        "android_manifest": "vigidroid/app/src/main/assets/models/dual_branch_dex_manifest/export_manifest.json",
    },
    "linregdroid_permission": {
        "val_scores": "linear/artifacts/metrics/val_scores.json",
        "export_manifest": "linear/artifacts/export/linregdroid_permission/export_manifest.json",
        "android_manifest": "vigidroid/app/src/main/assets/models/linregdroid_permission/export_manifest.json",
    },
}


def _repo_root() -> Path:
    return _REPO_ROOT


def stamp_model(
    model_id: str,
    *,
    root: Path,
    dry_run: bool = False,
) -> list[dict]:
    spec = MODEL_BUNDLES.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown model_id: {model_id}")
    val_scores = root / spec["val_scores"]
    if not val_scores.is_file():
        raise FileNotFoundError(f"Missing val_scores for {model_id}: {val_scores}")

    results: list[dict] = []
    for key in ("export_manifest", "android_manifest"):
        manifest = root / spec[key]
        if not manifest.is_file():
            continue
        row = stamp_export_manifest(manifest, val_scores, dry_run=dry_run)
        row["model_id"] = model_id
        row["target"] = key
        results.append(row)
    if not results:
        raise FileNotFoundError(f"No export_manifest.json found for {model_id}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Subset of models (default: all thesis bundles)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = _repo_root()
    model_ids = args.model_id or sorted(MODEL_BUNDLES.keys())
    stamped = 0
    errors: list[str] = []

    for model_id in model_ids:
        try:
            rows = stamp_model(model_id, root=root, dry_run=args.dry_run)
            for row in rows:
                status = "would stamp" if args.dry_run else "stamped"
                print(
                    f"  {model_id} [{row['target']}]: {status} "
                    f"val_f1={row.get('val_f1'):.4f} val_accuracy={row.get('val_accuracy'):.4f}"
                )
                if row.get("changed", True):
                    stamped += 1
        except (FileNotFoundError, ValueError, KeyError) as exc:
            errors.append(f"{model_id}: {exc}")
            print(f"  SKIP {model_id}: {exc}", file=sys.stderr)

    print(f"\nStamped {stamped} manifest(s)")
    if errors:
        print(f"Skipped {len(errors)} model(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
