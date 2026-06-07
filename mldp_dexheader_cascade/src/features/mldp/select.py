"""Orchestrate PRNR → SPR → PMAR and freeze MLDP permission set S."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.constants import PUBLISHED_MLDP_PERMISSIONS
from src.features.mldp.pmar import collapse_by_implications, mine_association_rules
from src.features.mldp.prnr import compute_prnr, prnr_to_json
from src.features.mldp.spr import rank_by_support


def run_mldp_selection(
    cfg: PipelineConfig,
    *,
    train_transactions: list[set[str]],
    train_labels: list[int],
) -> tuple[list[str], dict[str, Any]]:
    mldp_cfg = cfg.mldp
    drop_t = float(mldp_cfg.get("prnr_drop_abs_threshold", 0.05))
    skew = bool(mldp_cfg.get("skew_correction", True))
    spr_top = int(mldp_cfg.get("spr_keep_top", 25))
    pmar_sup = float(mldp_cfg.get("pmar_min_support", 0.10))
    pmar_conf = float(mldp_cfg.get("pmar_min_confidence", 0.965))
    max_s = int(mldp_cfg.get("max_permissions", 30))
    min_s = int(mldp_cfg.get("min_permissions", 8))
    use_fallback = bool(mldp_cfg.get("fallback_published_list", True))

    prnr = compute_prnr(
        train_transactions,
        train_labels,
        drop_abs_threshold=drop_t,
        skew_correction=skew,
    )
    spr_kept, spr_stats = rank_by_support(
        prnr.survivors, train_transactions, keep_top=spr_top
    )

    malware_tx = [tx for tx, y in zip(train_transactions, train_labels) if y == 1]
    rule_records, implications = mine_association_rules(
        malware_tx,
        spr_kept,
        min_support=pmar_sup,
        min_confidence=pmar_conf,
    )
    pmar_kept, pmar_removed = collapse_by_implications(
        spr_kept, implications, prnr.r_scores
    )

    target_hint = int(mldp_cfg.get("target_size_hint", 22))
    selected = pmar_kept
    spr_fallback_used = False
    if len(selected) < target_hint and len(spr_kept) >= min_s:
        selected = spr_kept[: min(len(spr_kept), target_hint, max_s)]
        spr_fallback_used = True

    fallback_used = False
    if len(selected) < min_s and use_fallback:
        selected = list(PUBLISHED_MLDP_PERMISSIONS)
        fallback_used = True
    elif len(selected) > max_s:
        raise ValueError(
            f"MLDP selected |S|={len(selected)} exceeds max {max_s} (M1 guard)"
        )

    trace: dict[str, Any] = {
        "method": mldp_cfg.get("method", "prnr_spr_pmar"),
        "association_algorithm": "pairwise_count",
        "stages": {
            "full_vocab": len(prnr.malware_counts | prnr.benign_counts),
            "after_prnr": len(prnr.survivors),
            "after_spr": len(spr_kept),
            "after_pmar": len(pmar_kept),
            "final_S": len(selected),
        },
        "spr_fallback_used": spr_fallback_used,
        "fallback_published_list_used": fallback_used,
        "prnr": prnr_to_json(prnr),
        "spr_support": spr_stats,
        "pmar_rules": rule_records,
        "pmar_removed": pmar_removed,
        "selected_permissions": selected,
    }
    return selected, trace


def save_mldp_artifacts(
    processed_dir: Path,
    selected: list[str],
    trace: dict[str, Any],
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = processed_dir / "mldp_permission_vocab.json"
    trace_path = processed_dir / "mldp_trace.json"

    vocab_path.write_text(
        json.dumps({"tokens": selected, "size": len(selected)}, indent=2) + "\n",
        encoding="utf-8",
    )
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
