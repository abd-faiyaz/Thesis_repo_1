"""Validate MLDP frozen permission set S after selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import PipelineConfig


def validate_selected_set(
    selected: list[str],
    metadata: dict[str, Any],
    cfg: PipelineConfig,
) -> dict[str, Any]:
    mldp_cfg = cfg.mldp
    expected_min = int(mldp_cfg.get("expected_s_min", 20))
    expected_max = int(mldp_cfg.get("expected_s_max", 40))
    s_size = len(selected)
    fallback_used = bool(metadata.get("fallback_used", False))

    in_range = expected_min <= s_size <= expected_max
    warnings: list[str] = []
    if not in_range:
        warnings.append(
            f"|S|={s_size} outside expected [{expected_min}, {expected_max}] — "
            "tune mldp.prnr / mldp.support / mldp.association thresholds"
        )
    if fallback_used:
        warnings.append(
            "fallback_used=true — rule mining yielded fewer than min_permissions; "
            "PRNR/support backfill applied"
        )

    return {
        "s_size": s_size,
        "expected_s_min": expected_min,
        "expected_s_max": expected_max,
        "in_expected_range": in_range,
        "fallback_used": fallback_used,
        "passed": in_range and not fallback_used,
        "warnings": warnings,
        **metadata,
    }


def write_selection_validation(
    cfg: PipelineConfig,
    validation: dict[str, Any],
) -> Path:
    out_path = cfg.paths.mldp_dir / "selection_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return out_path
