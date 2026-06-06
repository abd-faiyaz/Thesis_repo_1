"""Load and resolve paths from YAML configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PACKAGE_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class PathsConfig:
    apk_root: Path
    artifacts: Path
    processed: Path
    checkpoints: Path
    export: Path
    dataset_index: Path
    splits_dir: Path
    failed_apks_log: Path
    permission_vocab: Path
    latest_checkpoint: Path


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    paths: PathsConfig
    raw: dict[str, Any]

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.raw.get("pipeline", {})

    @property
    def preprocessing(self) -> dict[str, Any]:
        return self.raw.get("preprocessing", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def export(self) -> dict[str, Any]:
        return self.raw.get("export", {})


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    cfg_file = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not cfg_file.is_absolute():
        cfg_file = (_PACKAGE_ROOT / cfg_file).resolve()

    with cfg_file.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    root = _PACKAGE_ROOT
    paths_raw = raw.get("paths", {})

    apk_root = os.environ.get("APK_ROOT", paths_raw.get("apk_root", "data/apks"))
    paths = PathsConfig(
        apk_root=_resolve_path(root, apk_root),
        artifacts=_resolve_path(root, paths_raw.get("artifacts", "artifacts")),
        processed=_resolve_path(root, paths_raw.get("processed", "artifacts/processed")),
        checkpoints=_resolve_path(root, paths_raw.get("checkpoints", "artifacts/checkpoints")),
        export=_resolve_path(root, paths_raw.get("export", "artifacts/export")),
        dataset_index=_resolve_path(root, paths_raw.get("dataset_index", "artifacts/dataset_index.csv")),
        splits_dir=_resolve_path(root, paths_raw.get("splits_dir", "artifacts/splits")),
        failed_apks_log=_resolve_path(root, paths_raw.get("failed_apks_log", "artifacts/failed_apks.log")),
        permission_vocab=_resolve_path(
            root, paths_raw.get("permission_vocab", "artifacts/permission_vocab.json")
        ),
        latest_checkpoint=_resolve_path(
            root, paths_raw.get("latest_checkpoint", "artifacts/checkpoints/linregdroid.pth")
        ),
    )
    return PipelineConfig(root=root, paths=paths, raw=raw)


def ensure_artifact_dirs(cfg: PipelineConfig) -> None:
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
    cfg.paths.splits_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.export.mkdir(parents=True, exist_ok=True)
    cfg.paths.dataset_index.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.failed_apks_log.parent.mkdir(parents=True, exist_ok=True)
