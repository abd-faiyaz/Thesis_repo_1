"""Run full MLDP pipeline on train split only."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import PipelineConfig
from src.features.permission_vector import save_selected_permissions
from src.mldp.association_rules import mine_rule_permissions
from src.mldp.prnr import compute_prnr, prnr_to_json
from src.mldp.support_filter import filter_by_support
from src.mldp.validation import validate_selected_set, write_selection_validation


def run_mldp_selection(
    cfg: PipelineConfig,
    *,
    train_transactions: list[set[str]],
    train_labels: list[int],
) -> list[str]:
    mldp_cfg = cfg.mldp
    prnr_cfg = mldp_cfg.get("prnr", {})
    sup_cfg = mldp_cfg.get("support", {})
    assoc_cfg = mldp_cfg.get("association", {})
    max_s = int(mldp_cfg.get("max_permissions", 40))
    min_s = int(mldp_cfg.get("min_permissions", 5))

    prnr_result = compute_prnr(
        train_transactions,
        train_labels,
        epsilon=float(prnr_cfg.get("epsilon", 1e-6)),
        min_rate_delta=float(prnr_cfg.get("min_rate_delta", 0.02)),
        top_k=int(prnr_cfg.get("top_k", 80)),
    )

    prnr_path = cfg.paths.mldp_dir / "prnr_scores.json"
    prnr_path.write_text(json.dumps(prnr_to_json(prnr_result), indent=2) + "\n", encoding="utf-8")

    supported, support_stats = filter_by_support(
        prnr_result.ranked,
        train_transactions,
        min_support=float(sup_cfg.get("min_support", 0.01)),
        max_support=float(sup_cfg.get("max_support", 0.95)),
    )
    support_path = cfg.paths.mldp_dir / "support_stats.json"
    support_path.write_text(
        json.dumps({"supported": supported, "stats": support_stats}, indent=2) + "\n",
        encoding="utf-8",
    )

    malware_tx = [
        tx
        for tx, label in zip(train_transactions, train_labels)
        if label == 1
    ]
    rule_perms, rule_records = mine_rule_permissions(
        malware_tx,
        supported,
        min_support=float(assoc_cfg.get("min_support", 0.05)),
        min_confidence=float(assoc_cfg.get("min_confidence", 0.70)),
        min_lift=float(assoc_cfg.get("min_lift", 1.2)),
        max_stored_rules=int(assoc_cfg.get("max_stored_rules", 200)),
    )

    rules_path = cfg.paths.mldp_dir / "association_rules.json"
    rules_path.write_text(json.dumps(rule_records, indent=2) + "\n", encoding="utf-8")

    selected = list(rule_perms)
    fallback_used = False
    if len(selected) < min_s:
        fallback_used = True
        for perm in supported:
            if perm not in selected:
                selected.append(perm)
            if len(selected) >= min_s:
                break

    prnr_order = {p: i for i, p in enumerate(prnr_result.ranked)}
    selected.sort(key=lambda p: prnr_order.get(p, 10_000))
    selected = selected[:max_s]

    metadata = {
        "fallback_used": fallback_used,
        "n_prnr_candidates": len(prnr_result.ranked),
        "n_after_support": len(supported),
        "n_from_rules": len(rule_perms),
        "max_permissions": max_s,
        "association_rule_mode": "malware_only_itemsets",
        "association_rule_note": (
            "FP-Growth on malware-only transactions; permissions from high-confidence "
            "rules among PRNR/support candidates (implicit malware consequent)."
        ),
    }
    save_selected_permissions(cfg.paths.selected_permissions, selected, metadata=metadata)

    validation = validate_selected_set(selected, metadata, cfg)
    validation_path = write_selection_validation(cfg, validation)
    for warning in validation.get("warnings", []):
        print(f"  MLDP warning: {warning}")
    print(f"  selection validation → {validation_path} (passed={validation['passed']})")

    return selected


def load_transactions_for_split(transactions_dir: Path, split: str) -> tuple[list[set[str]], list[int], list[str]]:
    split_dir = transactions_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing transactions for split={split}: {split_dir}")

    transactions: list[set[str]] = []
    labels: list[int] = []
    apk_ids: list[str] = []

    for path in sorted(split_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        transactions.append(set(data["permissions"]))
        labels.append(int(data["label"]))
        apk_ids.append(data["apk_id"])
    return transactions, labels, apk_ids
