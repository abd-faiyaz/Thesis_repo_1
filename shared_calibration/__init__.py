"""Shared threshold tuning and cascade band calibration for thesis pipelines."""

from shared_calibration.calibrate import (
    calibrate_cascade_thresholds,
    calibrate_t_high,
    calibrate_t_low,
    false_alarm_rate,
    false_omission_rate,
)
from shared_calibration.metrics import tune_threshold
from shared_calibration.val_scores import (
    apk_ids_from_paths,
    build_split_scores_payload,
    default_canonical_val_path,
    find_repo_root,
    load_canonical_val_ids,
    score_rows_from_arrays,
    split_scores_filename,
    sync_val_scores_to_workspace,
    write_split_scores,
    write_split_scores_bundle,
)
from shared_calibration.cascade_policy import (
    DEFAULT_TIER_SPEC,
    build_cascade_policy,
    inner_join_val_scores,
    load_mode_b_bands_from_thresholds,
)
from shared_calibration.thresholds import (
    build_thresholds_payload,
    build_val_thresholds_payload,
    cascade_band_from_calibration,
    format_cascade_band_summary,
    get_cascade_targets,
    read_saved_thresholds,
    read_thresholds_payload,
    write_export_thresholds,
    write_thresholds,
)

__all__ = [
    "DEFAULT_TIER_SPEC",
    "build_cascade_policy",
    "inner_join_val_scores",
    "load_mode_b_bands_from_thresholds",
    "tune_threshold",
    "false_omission_rate",
    "false_alarm_rate",
    "calibrate_t_low",
    "calibrate_t_high",
    "calibrate_cascade_thresholds",
    "cascade_band_from_calibration",
    "build_thresholds_payload",
    "build_val_thresholds_payload",
    "get_cascade_targets",
    "format_cascade_band_summary",
    "read_thresholds_payload",
    "write_export_thresholds",
    "write_thresholds",
    "read_saved_thresholds",
    "build_split_scores_payload",
    "write_split_scores",
    "write_split_scores_bundle",
    "split_scores_filename",
    "sync_val_scores_to_workspace",
    "load_canonical_val_ids",
    "default_canonical_val_path",
    "find_repo_root",
    "score_rows_from_arrays",
    "apk_ids_from_paths",
]
