#!/usr/bin/env python3
"""Offline C5 validation: cascade vs full-pipeline replay on aligned val scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared_calibration.cascade_policy import load_val_scores_workspace  # noqa: E402
from shared_calibration.cascade_simulate import (  # noqa: E402
    load_malware_threshold,
    load_policy,
    simulate_cascade_batch,
    summarize_outcomes,
)

DEFAULT_WORKSPACE = _REPO_ROOT / "Shared_pipeline_Files/calibration"
DEFAULT_POLICY = DEFAULT_WORKSPACE / "cascade_policy.json"
DEFAULT_OUT = DEFAULT_WORKSPACE / "cascade_simulation_report.json"

PER_MODEL_THRESHOLDS: dict[str, Path] = {
    "mldp_pruned_permission": _REPO_ROOT / "permission_extractor/artifacts/metrics/thresholds.json",
    "broadcast_mldp_hybrid": _REPO_ROOT / "broadcast_mldp_hybrid/artifacts/metrics/thresholds.json",
}


def _load_malware_thresholds() -> dict[str, float]:
    out: dict[str, float] = {}
    for model_id, path in PER_MODEL_THRESHOLDS.items():
        if path.is_file():
            out[model_id] = load_malware_threshold(path)
    return out


def _collect_models_from_policy(policy: dict) -> list[str]:
    models: list[str] = []
    for tier in policy.get("tiers", []):
        for model_id in tier.get("models", []):
            if model_id not in models:
                models.append(model_id)
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.policy.is_file():
        print(f"Policy not found: {args.policy}", file=sys.stderr)
        return 1

    policy = load_policy(args.policy)
    model_ids = _collect_models_from_policy(policy)
    payloads = load_val_scores_workspace(args.workspace, model_ids=model_ids)
    if not payloads:
        print(f"No val_scores in {args.workspace}", file=sys.stderr)
        return 1

    malware_thresholds = _load_malware_thresholds()
    cascade_outcomes = simulate_cascade_batch(
        policy, payloads, malware_thresholds=malware_thresholds, early_exit=True
    )
    full_outcomes = simulate_cascade_batch(
        policy, payloads, malware_thresholds=malware_thresholds, early_exit=False
    )

    cascade_stats = summarize_outcomes(cascade_outcomes)
    full_stats = summarize_outcomes(full_outcomes)
    f1_delta = cascade_stats["f1"] - full_stats["f1"]
    model_reduction = 1.0 - (
        cascade_stats["avg_models_run"] / full_stats["avg_models_run"]
        if full_stats["avg_models_run"] > 0
        else 0.0
    )

    report = {
        "policy_name": policy.get("policy_name"),
        "n_apks": cascade_stats["n"],
        "cascade": cascade_stats,
        "full_pipeline": full_stats,
        "comparison": {
            "f1_delta": f1_delta,
            "accuracy_delta": cascade_stats["accuracy"] - full_stats["accuracy"],
            "avg_models_run_reduction": model_reduction,
            "pass_f1_within_0.01": abs(f1_delta) <= 0.01,
        },
        "notes": [
            "manifest_xgb and bytecnn have no thesis val_scores; tier 3/4 use available scores only.",
            "Device validation: run 400-APK eval twice (enabled=false vs true) and use compare_cascade_eval.py.",
        ],
        "alignment_warnings": (policy.get("calibration") or {}).get("alignment_warnings", []),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Policy: {report['policy_name']}")
    print(f"Aligned APKs: {report['n_apks']}")
    print("\n=== cascade (early exit) ===")
    print(f"  F1:        {cascade_stats['f1']:.4f}")
    print(f"  accuracy:  {cascade_stats['accuracy']:.4f}")
    print(f"  avg models run: {cascade_stats['avg_models_run']:.2f}")
    print(f"  early exit: {cascade_stats['early_exit_rate']:.1%}")
    print(f"  exit tiers: {cascade_stats['exit_tier_counts']}")
    print("\n=== full pipeline (no early exit) ===")
    print(f"  F1:        {full_stats['f1']:.4f}")
    print(f"  accuracy:  {full_stats['accuracy']:.4f}")
    print(f"  avg models run: {full_stats['avg_models_run']:.2f}")
    print("\n=== comparison ===")
    print(f"  F1 delta:  {f1_delta:+.4f}")
    print(f"  model run reduction: {model_reduction:.1%}")
    print(f"  pass (|F1 delta| <= 0.01): {report['comparison']['pass_f1_within_0.01']}")
    print(f"\nWrote report → {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
