"""Live archive mirroring when PA_RUN_ID is set (see run_pattern_a.sh PA_ARCHIVE=1)."""

from __future__ import annotations

import sys
from pathlib import Path

PROFILE_KEY = "early_fusion_dex_manifest"
_MODEL_ROOT = Path(__file__).resolve().parent.parent


def _ensure_scripts() -> None:
    d = _MODEL_ROOT
    while d != d.parent:
        scripts = d / "scripts"
        if (scripts / "thesis_run_logging.py").is_file():
            sd = str(scripts)
            if sd not in sys.path:
                sys.path.insert(0, sd)
            return
        d = d.parent


_ensure_scripts()
from thesis_run_logging import RunArchive  # noqa: E402


def get_archive() -> RunArchive:
    return RunArchive.for_profile(_MODEL_ROOT, PROFILE_KEY)


def log_epoch(**kwargs) -> None:
    arc = get_archive()
    if arc.enabled:
        arc.log_epoch(**kwargs)


def after_train(checkpoint: Path, meta: dict) -> None:
    arc = get_archive()
    if not arc.enabled:
        return
    arc.log_training_run_info(meta)
    arc.mirror_checkpoint(checkpoint)


def after_eval(written_path: Path) -> None:
    arc = get_archive()
    if not arc.enabled:
        return
    arc.mirror_file(written_path, f"metrics/{written_path.name}")
    arc.mirror_test_results()
    arc.finalize_manifest()


def after_export() -> None:
    arc = get_archive()
    if arc.enabled:
        arc.mirror_export_bundle()


def after_parity(report_path: Path | None = None) -> None:
    arc = get_archive()
    if arc.enabled:
        arc.mirror_parity_report(report_path)
        arc.finalize_manifest()
