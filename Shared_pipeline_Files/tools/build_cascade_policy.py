#!/usr/bin/env python3
"""Build cascade_policy.json from aligned per-model val_scores dumps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared_calibration.cascade_policy import (  # noqa: E402
    DEFAULT_TIER_SPEC,
    build_cascade_policy,
    load_mode_b_bands_from_thresholds,
    load_val_scores_workspace,
    try_inner_join_val_scores,
    write_json,
)

DEFAULT_WORKSPACE = _REPO_ROOT / "Shared_pipeline_Files/calibration"
DEFAULT_TIER_SPEC_PATH = _REPO_ROOT / "Shared_pipeline_Files/data/cascade_tier_spec.json"
DEFAULT_MODE_B_THRESHOLDS = (
    _REPO_ROOT / "mldp_dexheader_cascade/artifacts/metrics/thresholds.json"
)
DEFAULT_PER_MODEL_THRESHOLDS: dict[str, Path] = {
    "mldp_pruned_permission": _REPO_ROOT / "permission_extractor/artifacts/metrics/thresholds.json",
    "broadcast_mldp_hybrid": _REPO_ROOT / "broadcast_mldp_hybrid/artifacts/metrics/thresholds.json",
    "mldp_dexheader_cascade_mode_b": DEFAULT_MODE_B_THRESHOLDS,
    "early_fusion_dex_manifest": _REPO_ROOT
    / "Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/metrics/thresholds.json",
    "bytecnn": Path(),
}


def _repo_root() -> Path:
    return _REPO_ROOT


def _collect_models_from_tier_spec(tier_specs: list[dict]) -> list[str]:
    models: list[str] = []
    for tier in tier_specs:
        for model_id in tier.get("models", []):
            if model_id not in models:
                models.append(model_id)
    return models


def _parse_external_scores(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected MODEL_ID=PATH, got: {item}")
        model_id, raw_path = item.split("=", 1)
        out[model_id.strip()] = Path(raw_path.strip())
    return out


def _load_tier_spec(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return DEFAULT_TIER_SPEC
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tiers" in payload:
        return list(payload["tiers"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported tier spec shape: {path}")


def _models_with_scores(
    tier_specs: list[dict],
    workspace: Path,
    external_scores: dict[str, Path],
) -> tuple[dict[str, dict], list[str]]:
    all_models = _collect_models_from_tier_spec(tier_specs)
    loaded = load_val_scores_workspace(
        workspace,
        model_ids=all_models,
        external_scores=external_scores,
    )
    missing = [model_id for model_id in all_models if model_id not in loaded]
    return loaded, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-model tier calibration → cascade_policy.json"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Directory with <model_id>_val_scores.json files",
    )
    parser.add_argument(
        "--tier-spec",
        type=Path,
        default=DEFAULT_TIER_SPEC_PATH,
        help="Tier definition JSON (list or {\"tiers\": [...]}); default built-in spec",
    )
    parser.add_argument(
        "--external-scores",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Extra val_scores for models without thesis pipelines (e.g. manifest_xgb)",
    )
    parser.add_argument(
        "--reuse-tier2-bands-from",
        type=Path,
        default=None,
        help="mode_b thresholds.json — reuse t_low/t_high instead of re-calibrating tier 2",
    )
    parser.add_argument(
        "--policy-name",
        default="cascade_v1",
    )
    parser.add_argument(
        "--tier3-pattern-model",
        default="early_fusion_dex_manifest",
    )
    parser.add_argument(
        "--target-for",
        type=float,
        default=0.02,
        dest="target_false_omission_rate",
    )
    parser.add_argument(
        "--target-fa",
        type=float,
        default=0.02,
        dest="target_false_alarm_at_thigh",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "Shared_pipeline_Files/calibration/cascade_policy.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Feasibility report JSON (default: <out>_report.json)",
    )
    parser.add_argument(
        "--min-aligned",
        type=int,
        default=1,
        help="Minimum inner-join APK count (default 1 for smoke tests)",
    )
    args = parser.parse_args(argv)

    tier_specs = _load_tier_spec(args.tier_spec)
    external_scores = _parse_external_scores(args.external_scores)
    loaded, missing = _models_with_scores(tier_specs, args.workspace, external_scores)

    # Inner join only across models we actually have scores for.
    required_for_join = [
        model_id
        for model_id in _collect_models_from_tier_spec(tier_specs)
        if model_id in loaded
    ]
    if not required_for_join:
        print("No val_scores found in workspace.", file=sys.stderr)
        print(f"  workspace: {args.workspace.resolve()}", file=sys.stderr)
        if missing:
            print(f"  missing: {', '.join(missing)}", file=sys.stderr)
        print(
            "\nRun pipeline evaluate (Phase D) or collect_calibration_val_scores.py first.",
            file=sys.stderr,
        )
        return 1

    if missing:
        print(f"Note: calibrating without val scores for: {', '.join(missing)}")

    reference_aligned = try_inner_join_val_scores(loaded, required_models=required_for_join)
    n_ref = len(reference_aligned.apk_ids) if reference_aligned else 0
    if n_ref < args.min_aligned and n_ref == 0:
        print(
            "Warning: zero global APK overlap across all scored models; "
            "using per-tier joins and per-model threshold fallbacks.",
            file=sys.stderr,
        )

    reuse_tier_bands: dict[int, dict[str, float]] = {}
    tier2_path = args.reuse_tier2_bands_from
    if tier2_path is None and DEFAULT_MODE_B_THRESHOLDS.is_file():
        tier2_path = DEFAULT_MODE_B_THRESHOLDS
    if tier2_path is not None and tier2_path.is_file():
        reuse_tier_bands[2] = load_mode_b_bands_from_thresholds(tier2_path)
        print(
            f"Tier 2 bands reused from {tier2_path}: "
            f"t_low={reuse_tier_bands[2]['t_low']:.4f} "
            f"t_high={reuse_tier_bands[2]['t_high']:.4f}"
        )

    per_model_thresholds = {
        model_id: path
        for model_id, path in DEFAULT_PER_MODEL_THRESHOLDS.items()
        if path.is_file()
    }

    policy, report = build_cascade_policy(
        tier_specs=tier_specs,
        payloads=loaded,
        target_false_omission_rate=args.target_false_omission_rate,
        target_false_alarm_at_thigh=args.target_false_alarm_at_thigh,
        policy_name=args.policy_name,
        tier3_pattern_model=args.tier3_pattern_model,
        enabled=False,
        reuse_tier_bands=reuse_tier_bands,
        per_model_thresholds=per_model_thresholds,
        aligned=reference_aligned,
    )

    write_json(args.out, policy)
    report_path = args.report or args.out.with_name(args.out.stem + "_report.json")
    write_json(report_path, report)

    if report.get("alignment_warnings"):
        print("\nAlignment warnings:")
        for warning in report["alignment_warnings"]:
            print(f"  - {warning}")

    print(f"Reference aligned APKs (dex cluster): {n_ref}")
    print(f"Wrote policy → {args.out.resolve()}")
    print(f"Wrote report → {report_path.resolve()}")
    print("\nTier bands:")
    for row in report["feasibility"]:
        print(
            f"  tier {row['tier']}: t_low={row['t_low']:.4f} t_high={row['t_high']:.4f} "
            f"tier_exit={row['tier_exit_rate']:.3f} cumulative_exit={row['cumulative_exit_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
