"""Cross-model tier calibration for cascade_policy.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shared_calibration.calibrate import calibrate_cascade_thresholds
from shared_calibration.thresholds import cascade_band_from_calibration

# Legacy XGB/CNN validation accuracies used until external val scores exist.
LEGACY_FUSION_WEIGHTS: dict[str, float] = {
    "manifest_xgb": 0.9748,
    "bytecnn": 0.9607843,
}

VAL_SCORE_MODEL_ALIASES: dict[str, str] = {
    "mldp_dexheader_cascade_mode_a": "mldp_dexheader_cascade",
    "mldp_dexheader_cascade_mode_b": "mldp_dexheader_cascade",
}

DEFAULT_TIER_BANDS: dict[int, tuple[float, float]] = {
    1: (0.15, 0.85),
    2: (0.20, 0.80),
    3: (0.30, 0.70),
    4: (0.50, 0.50),
}

DEFAULT_TIER_SPEC: list[dict[str, Any]] = [
    {
        "tier": 1,
        "models": ["mldp_pruned_permission", "broadcast_mldp_hybrid"],
        "aggregation": "weighted_avg",
        "conservative_malware_or": True,
    },
    {
        "tier": 2,
        "models": ["mldp_dexheader_cascade_mode_b"],
        "mlp_header_fallback": True,
    },
    {
        "tier": 3,
        "models": ["early_fusion_dex_manifest", "manifest_xgb"],
    },
    {
        "tier": 4,
        "models": ["bytecnn"],
        "final": True,
    },
]


@dataclass
class AlignedValSet:
    apk_ids: list[str]
    labels: np.ndarray
    scores_by_model: dict[str, np.ndarray]
    model_metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)


def val_scores_filename(model_id: str) -> str:
    return f"{model_id}_val_scores.json"


def resolve_val_score_model_id(model_id: str) -> str:
    return VAL_SCORE_MODEL_ALIASES.get(model_id, model_id)


def load_val_scores(path: Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in payload:
        raise ValueError(f"Not a val_scores payload: {path}")
    return payload


def load_val_scores_workspace(
    workspace: Path,
    *,
    model_ids: list[str] | None = None,
    external_scores: dict[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    workspace = Path(workspace)
    external_scores = external_scores or {}
    loaded: dict[str, dict[str, Any]] = {}

    for model_id in model_ids or []:
        source_key = resolve_val_score_model_id(model_id)
        if model_id in external_scores:
            loaded[model_id] = load_val_scores(external_scores[model_id])
            continue
        candidate = workspace / val_scores_filename(source_key)
        if candidate.is_file():
            loaded[model_id] = load_val_scores(candidate)
    return loaded


def _rows_to_map(payload: dict[str, Any]) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    for row in payload.get("rows", []):
        apk_id = str(row["apk_id"]).lower()
        out[apk_id] = (int(row["label"]), float(row["score"]))
    return out


def resolve_payload_for_model(
    model_id: str,
    payloads: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    if model_id in payloads:
        return model_id, payloads[model_id]
    alias = VAL_SCORE_MODEL_ALIASES.get(model_id)
    if alias and alias in payloads:
        return model_id, payloads[alias]
    return None


def inner_join_val_scores(
    payloads: dict[str, dict[str, Any]],
    *,
    required_models: list[str] | None = None,
) -> AlignedValSet:
    if not payloads:
        raise ValueError("No val_scores payloads to align")

    required = required_models or list(payloads.keys())
    resolved: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for model_id in required:
        hit = resolve_payload_for_model(model_id, payloads)
        if hit is None:
            missing.append(model_id)
        else:
            resolved[hit[0]] = hit[1]
    if missing:
        raise ValueError(f"Missing val_scores for: {', '.join(missing)}")

    maps = {model_id: _rows_to_map(payload) for model_id, payload in resolved.items()}
    common_ids = None
    for model_id, row_map in maps.items():
        ids = set(row_map)
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        raise ValueError("Inner join produced zero aligned APK ids")

    apk_ids = sorted(common_ids)
    labels = np.asarray([maps[required[0]][apk_id][0] for apk_id in apk_ids], dtype=int)
    scores_by_model: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float | None]] = {}
    for model_id in required:
        scores_by_model[model_id] = np.asarray(
            [maps[model_id][apk_id][1] for apk_id in apk_ids],
            dtype=np.float64,
        )
        metrics = payloads[model_id].get("metrics", {})
        model_metrics[model_id] = {
            "f1": float(metrics.get("f1", 1.0) or 1.0),
            "accuracy": float(metrics.get("accuracy", 1.0) or 1.0),
        }
    return AlignedValSet(
        apk_ids=apk_ids,
        labels=labels,
        scores_by_model=scores_by_model,
        model_metrics=model_metrics,
    )


def try_inner_join_val_scores(
    payloads: dict[str, dict[str, Any]],
    *,
    required_models: list[str],
) -> AlignedValSet | None:
    try:
        return inner_join_val_scores(payloads, required_models=required_models)
    except ValueError:
        return None


def load_per_model_cascade_band(path: Path, model_id: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if model_id == "mldp_dexheader_cascade_mode_b":
        return load_mode_b_bands_from_thresholds(path)
    cascade = payload.get("cascade")
    if isinstance(cascade, dict) and "t_low" in cascade:
        return {
            "t_low": float(cascade["t_low"]),
            "t_high": float(cascade["t_high"]),
            "val_false_omission_rate_at_t_low": float(
                cascade.get("val_false_omission_rate_at_t_low", 0.0)
            ),
            "val_false_alarm_rate_at_t_high": float(
                cascade.get("val_false_alarm_rate_at_t_high", 0.0)
            ),
            "val_step1_exit_rate": float(cascade.get("val_step1_exit_rate", 0.0)),
        }
    raise ValueError(f"No cascade band in {path} for {model_id}")


def merge_disjoint_tier_bands(bands: list[dict[str, float]]) -> dict[str, float]:
    """Conservative merge when tier members were calibrated on disjoint val APK sets."""
    t_low = max(float(b["t_low"]) for b in bands)
    t_high = min(float(b["t_high"]) for b in bands)
    if t_high <= t_low:
        t_high = min(1.0, t_low + 0.05)
    return {
        "t_low": t_low,
        "t_high": t_high,
        "val_false_omission_rate_at_t_low": max(
            float(b.get("val_false_omission_rate_at_t_low", 0.0)) for b in bands
        ),
        "val_false_alarm_rate_at_t_high": max(
            float(b.get("val_false_alarm_rate_at_t_high", 0.0)) for b in bands
        ),
        "val_step1_exit_rate": min(float(b.get("val_step1_exit_rate", 0.0)) for b in bands),
    }


def model_weights_from_payloads(
    payloads: dict[str, dict[str, Any]],
    *,
    fallback_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for model_id, payload in payloads.items():
        metrics = payload.get("metrics", {})
        weights[model_id] = max(float(metrics.get("f1", 1.0) or 1.0), 1e-6)
    for model_id, value in (fallback_weights or LEGACY_FUSION_WEIGHTS).items():
        weights.setdefault(model_id, float(value))
    return weights


def f1_weight_for_model(
    model_id: str,
    aligned: AlignedValSet,
    *,
    fallback_weights: dict[str, float] | None = None,
) -> float:
    if model_id in aligned.model_metrics:
        return max(float(aligned.model_metrics[model_id].get("f1", 1.0) or 1.0), 1e-6)
    fallbacks = fallback_weights or LEGACY_FUSION_WEIGHTS
    if model_id in fallbacks:
        return float(fallbacks[model_id])
    return 1.0


def tier_member_weights(
    models: list[str],
    aligned: AlignedValSet,
    *,
    fallback_weights: dict[str, float] | None = None,
    override_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    overrides = override_weights or {}
    weights: dict[str, float] = {}
    for model_id in models:
        if model_id in overrides:
            weights[model_id] = float(overrides[model_id])
        else:
            weights[model_id] = f1_weight_for_model(
                model_id, aligned, fallback_weights=fallback_weights
            )
    return weights


def weighted_tier_scores(
    aligned: AlignedValSet,
    models: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    available = [model_id for model_id in models if model_id in aligned.scores_by_model]
    if not available:
        raise ValueError(f"No aligned scores for tier models: {models}")
    total_w = sum(weights[model_id] for model_id in available)
    if total_w <= 0:
        raise ValueError(f"Non-positive total weight for tier models: {models}")
    combined = np.zeros(len(aligned.apk_ids), dtype=np.float64)
    for model_id in available:
        combined += weights[model_id] * aligned.scores_by_model[model_id]
    return combined / total_w


def load_mode_b_bands_from_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mode_b = payload.get("mode_b") or payload.get("cascade") or {}
    if "stage1_t_low" in mode_b:
        return {
            "t_low": float(mode_b["stage1_t_low"]),
            "t_high": float(mode_b["stage1_t_high"]),
            "val_false_omission_rate_at_t_low": float(
                mode_b.get("val_false_omission_rate_at_t_low", 0.0)
            ),
            "val_false_alarm_rate_at_t_high": float(
                mode_b.get("val_false_alarm_rate_at_t_high", 0.0)
            ),
            "val_step1_exit_rate": float(mode_b.get("val_step1_exit_rate", 0.0)),
        }
    if "t_low" in mode_b:
        return cascade_band_from_calibration(mode_b)
    raise ValueError(f"No mode_b / cascade bands in {path}")


def calibrate_tier_bands(
    labels: np.ndarray,
    tier_scores: np.ndarray,
    *,
    target_false_omission_rate: float = 0.02,
    target_false_alarm_at_thigh: float = 0.02,
) -> dict[str, float]:
    calibration = calibrate_cascade_thresholds(
        labels,
        tier_scores,
        target_false_omission_rate=target_false_omission_rate,
        target_false_alarm_at_thigh=target_false_alarm_at_thigh,
    )
    return cascade_band_from_calibration(calibration)


def simulate_cumulative_exits(
    aligned: AlignedValSet,
    tier_specs: list[dict[str, Any]],
    tier_bands: dict[int, dict[str, float]],
    tier_scores: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    n = len(aligned.apk_ids)
    if n == 0:
        return []

    remaining = np.ones(n, dtype=bool)
    report: list[dict[str, Any]] = []
    exited_before = 0

    for tier_spec in tier_specs:
        tier_num = int(tier_spec["tier"])
        bands = tier_bands.get(tier_num)
        scores = tier_scores.get(tier_num)
        if bands is None or scores is None:
            continue
        if scores.size == 0 or scores.shape[0] != n:
            continue

        t_low = float(bands["t_low"])
        t_high = float(bands["t_high"])
        active_scores = scores[remaining]
        exits = (active_scores <= t_low) | (active_scores >= t_high)
        if bool(tier_spec.get("final")):
            exits = np.ones(active_scores.shape[0], dtype=bool)

        n_active = int(remaining.sum())
        n_exit = int(exits.sum()) if n_active else 0
        tier_exit_rate = float(n_exit / n_active) if n_active else 0.0
        exited_before += n_exit
        cumulative_exit_rate = float(exited_before / n) if n else 0.0

        report.append(
            {
                "tier": tier_num,
                "n_active": n_active,
                "n_exit": n_exit,
                "tier_exit_rate": tier_exit_rate,
                "cumulative_exit_rate": cumulative_exit_rate,
                "t_low": t_low,
                "t_high": t_high,
            }
        )
        if n_active:
            remaining_idx = np.where(remaining)[0]
            remaining[remaining_idx[exits]] = False

    return report


def tier_f1_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    t_low: float,
    t_high: float,
    final_tier: bool,
) -> float:
    from sklearn.metrics import f1_score

    if final_tier:
        t_mid = (t_low + t_high) / 2.0
        preds = (scores >= t_mid).astype(int)
    else:
        preds = np.zeros_like(labels)
        preds[scores >= t_high] = 1
        preds[scores <= t_low] = 0
        uncertain = (scores > t_low) & (scores < t_high)
        preds[uncertain] = (scores[uncertain] >= 0.5).astype(int)
    return float(f1_score(labels, preds, zero_division=0))


def build_cascade_policy(
    *,
    tier_specs: list[dict[str, Any]] | None = None,
    payloads: dict[str, dict[str, Any]],
    target_false_omission_rate: float = 0.02,
    target_false_alarm_at_thigh: float = 0.02,
    policy_name: str = "cascade_v1",
    tier3_pattern_model: str = "early_fusion_dex_manifest",
    enabled: bool = False,
    reuse_tier_bands: dict[int, dict[str, float]] | None = None,
    per_model_thresholds: dict[str, Path] | None = None,
    weight_overrides: dict[str, float] | None = None,
    fallback_weights: dict[str, float] | None = None,
    aligned: AlignedValSet | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build cascade_policy.json payload and feasibility report."""
    specs = tier_specs or DEFAULT_TIER_SPEC
    reuse_tier_bands = reuse_tier_bands or {}
    per_model_thresholds = per_model_thresholds or {}
    tier_bands: dict[int, dict[str, float]] = {}
    tier_scores: dict[int, np.ndarray] = {}
    tier_aligned: dict[int, AlignedValSet] = {}
    tier_calibration_rows: list[dict[str, Any]] = []
    alignment_warnings: list[str] = []

    payload_weights = model_weights_from_payloads(payloads, fallback_weights=fallback_weights)
    model_weights: dict[str, float] = dict(payload_weights)
    fusion_weights: dict[str, float] = dict(fallback_weights or LEGACY_FUSION_WEIGHTS)

    all_models = sorted(
        {model_id for tier in specs for model_id in tier.get("models", [])}
    )
    for model_id in all_models:
        if model_id not in model_weights:
            model_weights[model_id] = float(fusion_weights.get(model_id, 1.0))
        if model_id in {"manifest_xgb", "bytecnn", "early_fusion_dex_manifest", "dual_branch_dex_manifest"}:
            fusion_weights[model_id] = model_weights[model_id]

    output_tiers: list[dict[str, Any]] = []
    for tier_spec in specs:
        tier_num = int(tier_spec["tier"])
        models = list(tier_spec.get("models", []))
        tier_entry = {key: value for key, value in tier_spec.items() if key != "models"}
        tier_entry["models"] = models

        score_models = [
            m for m in models if resolve_payload_for_model(m, payloads) is not None
        ]
        aligned_tier = (
            try_inner_join_val_scores(payloads, required_models=score_models)
            if score_models
            else None
        )
        tier_f1: float | None = None
        n_aligned = 0

        if tier_num in reuse_tier_bands:
            bands = dict(reuse_tier_bands[tier_num])
            source = "reused_thresholds"
            if aligned_tier is None:
                raise ValueError(
                    f"Tier {tier_num} reuse requires aligned val scores for: {score_models}"
                )
            weights = tier_member_weights(
                score_models,
                aligned_tier,
                fallback_weights=fallback_weights,
                override_weights=weight_overrides,
            )
            scores = weighted_tier_scores(aligned_tier, score_models, weights)
            n_aligned = len(aligned_tier.apk_ids)
            tier_aligned[tier_num] = aligned_tier
        elif aligned_tier is not None and len(score_models) >= 1:
            weights = tier_member_weights(
                score_models,
                aligned_tier,
                fallback_weights=fallback_weights,
                override_weights=weight_overrides,
            )
            scores = weighted_tier_scores(aligned_tier, score_models, weights)
            bands = calibrate_tier_bands(
                aligned_tier.labels,
                scores,
                target_false_omission_rate=target_false_omission_rate,
                target_false_alarm_at_thigh=target_false_alarm_at_thigh,
            )
            source = "calibrated" if len(score_models) > 1 else "calibrated_single_model"
            n_aligned = len(aligned_tier.apk_ids)
            tier_aligned[tier_num] = aligned_tier
            tier_f1 = tier_f1_at_threshold(
                aligned_tier.labels,
                scores,
                t_low=bands["t_low"],
                t_high=bands["t_high"],
                final_tier=bool(tier_spec.get("final")),
            )
        elif len(score_models) >= 2:
            disjoint_bands: list[dict[str, float]] = []
            for model_id in score_models:
                path = per_model_thresholds.get(model_id)
                if path is None or not Path(path).is_file():
                    raise ValueError(
                        f"Tier {tier_num} has disjoint val APK sets; provide thresholds for {model_id}"
                    )
                disjoint_bands.append(load_per_model_cascade_band(Path(path), model_id))
            bands = merge_disjoint_tier_bands(disjoint_bands)
            source = "merged_per_model_disjoint_val"
            msg = (
                f"Tier {tier_num}: zero APK overlap across {score_models}; "
                "merged per-model cascade bands (re-run with shared val splits for joint calibration)."
            )
            alignment_warnings.append(msg)
            scores = np.array([], dtype=np.float64)
        elif len(score_models) == 1:
            path = per_model_thresholds.get(score_models[0])
            if path is not None and Path(path).is_file():
                bands = load_per_model_cascade_band(Path(path), score_models[0])
                source = "per_model_thresholds"
            elif aligned_tier is not None:
                scores = weighted_tier_scores(
                    aligned_tier,
                    score_models,
                    tier_member_weights(
                        score_models,
                        aligned_tier,
                        fallback_weights=fallback_weights,
                        override_weights=weight_overrides,
                    ),
                )
                bands = calibrate_tier_bands(
                    aligned_tier.labels,
                    scores,
                    target_false_omission_rate=target_false_omission_rate,
                    target_false_alarm_at_thigh=target_false_alarm_at_thigh,
                )
                source = "calibrated_single_model"
                n_aligned = len(aligned_tier.apk_ids)
                tier_aligned[tier_num] = aligned_tier
                tier_f1 = tier_f1_at_threshold(
                    aligned_tier.labels,
                    scores,
                    t_low=bands["t_low"],
                    t_high=bands["t_high"],
                    final_tier=bool(tier_spec.get("final")),
                )
            else:
                raise ValueError(f"Tier {tier_num}: no val scores or thresholds for {score_models[0]}")
            if source == "per_model_thresholds":
                scores = np.array([], dtype=np.float64)
        else:
            defaults = DEFAULT_TIER_BANDS.get(tier_num, (0.15, 0.85))
            bands = {
                "t_low": defaults[0],
                "t_high": defaults[1],
                "val_false_omission_rate_at_t_low": 0.0,
                "val_false_alarm_rate_at_t_high": 0.0,
                "val_step1_exit_rate": 0.0,
            }
            source = "placeholder_no_val_scores"
            scores = np.array([], dtype=np.float64)
            alignment_warnings.append(
                f"Tier {tier_num}: no val scores for {models}; using placeholder bands"
            )

        tier_bands[tier_num] = bands
        tier_scores[tier_num] = scores
        tier_entry["t_low"] = bands["t_low"]
        tier_entry["t_high"] = bands["t_high"]
        output_tiers.append(tier_entry)

        tier_weights = {
            model_id: (
                weight_overrides.get(model_id)
                if weight_overrides and model_id in weight_overrides
                else model_weights.get(model_id, 1.0)
            )
            for model_id in models
        }
        row: dict[str, Any] = {
            "tier": tier_num,
            "models": models,
            "source": source,
            "n_aligned_apks": n_aligned,
            "weights": tier_weights,
            **bands,
        }
        if tier_f1 is not None:
            row["tier_f1"] = tier_f1
        tier_calibration_rows.append(row)

    reference_aligned = aligned
    if reference_aligned is None:
        dex_models = [
            m
            for m in (
                "broadcast_mldp_hybrid",
                "mldp_dexheader_cascade_mode_b",
                "mldp_dexheader_cascade",
                "mlp_header",
                "early_fusion_dex_manifest",
            )
            if resolve_payload_for_model(m, payloads) is not None
        ]
        reference_aligned = try_inner_join_val_scores(payloads, required_models=dex_models)

    cumulative_report: list[dict[str, Any]] = []
    if reference_aligned is not None:
        cumulative_report = simulate_cumulative_exits(
            reference_aligned, specs, tier_bands, tier_scores
        )

    policy = {
        "policy_name": policy_name,
        "enabled": enabled,
        "tier3_pattern_model": tier3_pattern_model,
        "tiers": output_tiers,
        "model_weights": {key: float(value) for key, value in sorted(model_weights.items())},
        "fusion_weights": {key: float(value) for key, value in sorted(fusion_weights.items())},
        "calibration": {
            "n_aligned_apks": len(reference_aligned.apk_ids) if reference_aligned else 0,
            "alignment_key": "sha256",
            "alignment_warnings": alignment_warnings,
            "targets": {
                "target_false_omission_rate": target_false_omission_rate,
                "target_false_alarm_at_thigh": target_false_alarm_at_thigh,
            },
            "tiers": tier_calibration_rows,
            "cumulative_exit": cumulative_report,
        },
    }

    report = {
        "policy_name": policy_name,
        "n_aligned_apks": len(reference_aligned.apk_ids) if reference_aligned else 0,
        "alignment_warnings": alignment_warnings,
        "feasibility": cumulative_report,
        "tier_calibration": tier_calibration_rows,
    }
    return policy, report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
