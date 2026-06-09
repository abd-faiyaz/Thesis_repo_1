"""Offline cascade replay on aligned val scores (mirrors ScanOrchestrator.java)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from shared_calibration.cascade_policy import (
    VAL_SCORE_MODEL_ALIASES,
    inner_join_val_scores,
    load_val_scores_workspace,
    resolve_payload_for_model,
    resolve_val_score_model_id,
)

EXIT_LOW_BENIGN = "low_confident_benign"
EXIT_HIGH_MALWARE = "high_confident_malware"
EXIT_FINAL = "final_verdict"

DEFAULT_MALWARE_THRESHOLD = 0.5


@dataclass
class ApkCascadeOutcome:
    apk_id: str
    label: int
    exit_tier: int
    exit_reason: str
    decision: str
    final_score: float
    models_run: list[str]


def load_malware_threshold(thresholds_path: Path) -> float:
    payload = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
    if "malware_threshold" in payload:
        return float(payload["malware_threshold"])
    if "tuned_val" in payload:
        return float(payload["tuned_val"])
    if "mode_b" in payload:
        mode_b = payload["mode_b"]
        if isinstance(mode_b, dict) and "tuned_val" in mode_b:
            return float(mode_b["tuned_val"])
    return DEFAULT_MALWARE_THRESHOLD


def _score_for_model(
    scores_by_model: dict[str, np.ndarray],
    model_id: str,
    apk_idx: int,
) -> float | None:
    if model_id in scores_by_model:
        return float(scores_by_model[model_id][apk_idx])
    alias = VAL_SCORE_MODEL_ALIASES.get(model_id)
    if alias and alias in scores_by_model:
        return float(scores_by_model[alias][apk_idx])
    resolved = resolve_val_score_model_id(model_id)
    if resolved in scores_by_model:
        return float(scores_by_model[resolved][apk_idx])
    return None


def _weighted_score(
    entries: list[tuple[str, float]],
    weights: dict[str, float],
) -> float | None:
    total_w = 0.0
    weighted = 0.0
    for model_id, score in entries:
        if score is None:
            continue
        w = float(weights.get(model_id, 1.0))
        weighted += w * score
        total_w += w
    if total_w <= 0.0:
        return None
    return float(weighted / total_w)


def _conservative_malware_exit(
    entries: list[tuple[str, float]],
    malware_thresholds: dict[str, float],
) -> bool:
    for model_id, score in entries:
        if score is None:
            continue
        threshold = malware_thresholds.get(model_id, DEFAULT_MALWARE_THRESHOLD)
        if score >= threshold:
            return True
    return False


def _decision_for_final_tier(t_low: float, t_high: float, score: float) -> str:
    if score <= t_low:
        return "benign"
    if score >= t_high:
        return "malware"
    if t_low < t_high:
        return "uncertain"
    t_mid = (t_low + t_high) / 2.0
    return "malware" if score >= t_mid else "benign"


def _tier_from_policy(policy: dict[str, Any], tier_num: int) -> dict[str, Any] | None:
    for tier in policy.get("tiers", []):
        if int(tier["tier"]) == tier_num:
            return tier
    return None


def _fuse_tier4(
    scores_by_model: dict[str, np.ndarray],
    apk_idx: int,
    tier3_models: list[str],
    fusion_weights: dict[str, float],
) -> tuple[float, list[str]]:
    pool_models = list(tier3_models) + ["bytecnn"]
    entries: list[tuple[str, float]] = []
    for model_id in pool_models:
        score = _score_for_model(scores_by_model, model_id, apk_idx)
        if score is not None:
            entries.append((model_id, score))
    fused = _weighted_score(entries, fusion_weights)
    models = [model_id for model_id, _ in entries]
    return (fused if fused is not None else -1.0, models)


def simulate_apk(
    apk_idx: int,
    aligned_scores: dict[str, np.ndarray],
    labels: np.ndarray,
    apk_ids: list[str],
    policy: dict[str, Any],
    *,
    malware_thresholds: dict[str, float] | None = None,
    early_exit: bool = True,
) -> ApkCascadeOutcome:
    """Replay cascade tiers for one APK index."""
    malware_thresholds = malware_thresholds or {}
    model_weights = policy.get("model_weights") or {}
    fusion_weights = policy.get("fusion_weights") or {}
    tier3_pattern = policy.get("tier3_pattern_model", "early_fusion_dex_manifest")

    models_run: list[str] = []

    def tier_entries(tier_spec: dict[str, Any]) -> list[tuple[str, float]]:
        models = list(tier_spec.get("models", []))
        if tier_spec.get("tier") == 3:
            models = [m if m != "early_fusion_dex_manifest" else tier3_pattern for m in models]
            models = [m if m != "dual_branch_dex_manifest" else tier3_pattern for m in models]
        out: list[tuple[str, float]] = []
        for model_id in models:
            score = _score_for_model(aligned_scores, model_id, apk_idx)
            if score is not None:
                out.append((model_id, score))
                if model_id not in models_run:
                    models_run.append(model_id)
        return out

    # Tier 1
    tier1 = _tier_from_policy(policy, 1)
    if tier1 is not None:
        entries = tier_entries(tier1)
        agg = _weighted_score(entries, model_weights)
        if agg is not None and early_exit:
            t_low = float(tier1["t_low"])
            t_high = float(tier1["t_high"])
            if agg <= t_low:
                return ApkCascadeOutcome(
                    apk_ids[apk_idx],
                    int(labels[apk_idx]),
                    1,
                    EXIT_LOW_BENIGN,
                    "benign",
                    agg,
                    list(models_run),
                )
            conservative = bool(tier1.get("conservative_malware_or"))
            if agg >= t_high or (
                conservative and _conservative_malware_exit(entries, malware_thresholds)
            ):
                return ApkCascadeOutcome(
                    apk_ids[apk_idx],
                    int(labels[apk_idx]),
                    1,
                    EXIT_HIGH_MALWARE,
                    "malware",
                    agg,
                    list(models_run),
                )

    # Tier 2
    tier2 = _tier_from_policy(policy, 2)
    if tier2 is not None:
        entries = tier_entries(tier2)
        if not entries and tier2.get("mlp_header_fallback"):
            score = _score_for_model(aligned_scores, "mlp_header", apk_idx)
            if score is not None:
                entries = [("mlp_header", score)]
                models_run.append("mlp_header")
        agg = _weighted_score(entries, model_weights)
        if agg is not None and early_exit:
            t_low = float(tier2["t_low"])
            t_high = float(tier2["t_high"])
            if agg <= t_low:
                return ApkCascadeOutcome(
                    apk_ids[apk_idx],
                    int(labels[apk_idx]),
                    2,
                    EXIT_LOW_BENIGN,
                    "benign",
                    agg,
                    list(models_run),
                )
            if agg >= t_high:
                return ApkCascadeOutcome(
                    apk_ids[apk_idx],
                    int(labels[apk_idx]),
                    2,
                    EXIT_HIGH_MALWARE,
                    "malware",
                    agg,
                    list(models_run),
                )

    # Tier 3
    tier3 = _tier_from_policy(policy, 3)
    tier3_models: list[str] = []
    if tier3 is not None:
        tier3_models = list(tier3.get("models", []))
        tier3_models = [
            tier3_pattern if m in {"early_fusion_dex_manifest", "dual_branch_dex_manifest"} else m
            for m in tier3_models
        ]
        entries = tier_entries(tier3)
        agg = _weighted_score(entries, model_weights)
        if agg is not None and early_exit:
            t_low = float(tier3["t_low"])
            t_high = float(tier3["t_high"])
            if agg <= t_low:
                return ApkCascadeOutcome(
                    apk_ids[apk_idx],
                    int(labels[apk_idx]),
                    3,
                    EXIT_LOW_BENIGN,
                    "benign",
                    agg,
                    list(models_run),
                )
            if agg >= t_high:
                return ApkCascadeOutcome(
                    apk_ids[apk_idx],
                    int(labels[apk_idx]),
                    3,
                    EXIT_HIGH_MALWARE,
                    "malware",
                    agg,
                    list(models_run),
                )

    # Tier 4 final fusion
    tier4 = _tier_from_policy(policy, 4) or {"t_low": 0.5, "t_high": 0.5}
    final_score, tier4_models = _fuse_tier4(
        aligned_scores, apk_idx, tier3_models, fusion_weights
    )
    for model_id in tier4_models:
        if model_id not in models_run:
            models_run.append(model_id)
    decision = _decision_for_final_tier(
        float(tier4.get("t_low", 0.5)),
        float(tier4.get("t_high", 0.5)),
        final_score,
    )
    return ApkCascadeOutcome(
        apk_ids[apk_idx],
        int(labels[apk_idx]),
        4,
        EXIT_FINAL,
        decision,
        final_score,
        list(models_run),
    )


def simulate_cascade_batch(
    policy: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    *,
    malware_thresholds: dict[str, float] | None = None,
    early_exit: bool = True,
) -> list[ApkCascadeOutcome]:
    cascade_models = sorted(
        {
            model_id
            for tier in policy.get("tiers", [])
            for model_id in tier.get("models", [])
        }
        | {"mlp_header", "bytecnn", "manifest_xgb"}
    )
    available = [
        model_id
        for model_id in cascade_models
        if resolve_payload_for_model(model_id, payloads) is not None
        or resolve_val_score_model_id(model_id) in payloads
    ]
    aligned = inner_join_val_scores(payloads, required_models=available)
    outcomes: list[ApkCascadeOutcome] = []
    for apk_idx in range(len(aligned.apk_ids)):
        outcomes.append(
            simulate_apk(
                apk_idx,
                aligned.scores_by_model,
                aligned.labels,
                aligned.apk_ids,
                policy,
                malware_thresholds=malware_thresholds,
                early_exit=early_exit,
            )
        )
    return outcomes


def summarize_outcomes(outcomes: list[ApkCascadeOutcome]) -> dict[str, Any]:
    labels = np.asarray([row.label for row in outcomes], dtype=int)
    preds = np.asarray(
        [
            1 if row.decision == "malware" else 0 if row.decision == "benign" else -1
            for row in outcomes
        ],
        dtype=int,
    )
    scored = preds >= 0
    n = int(scored.sum())
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    y_true = labels[scored]
    y_pred = preds[scored]
    avg_models = float(np.mean([len(row.models_run) for row in outcomes])) if outcomes else 0.0
    early = sum(1 for row in outcomes if row.exit_tier < 4)
    exit_by_tier: dict[int, int] = {}
    for row in outcomes:
        exit_by_tier[row.exit_tier] = exit_by_tier.get(row.exit_tier, 0) + 1

    return {
        "n": n,
        "n_uncertain": int((preds < 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "avg_models_run": avg_models,
        "early_exit_rate": float(early / len(outcomes)) if outcomes else 0.0,
        "exit_tier_counts": {str(k): v for k, v in sorted(exit_by_tier.items())},
    }


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
