"""Load YAML configuration and ensure artifact directories exist."""

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
    shared_manifest_csv: Path | None
    deployed_mlp_header_bundle: Path
    artifacts: Path
    manifests_dir: Path
    dataset_index: Path
    splits_dir: Path
    processed: Path
    checkpoints: Path
    metrics: Path
    export: Path
    failed_index_log: Path
    failed_apks_log: Path


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    paths: PathsConfig
    raw: dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.raw.get("model_id", "mldp_dexheader_cascade"))

    @property
    def domain(self) -> str:
        return str(self.raw.get("domain", "manifest_mldp_perm_dex_header"))

    @property
    def splits(self) -> dict[str, Any]:
        return self.raw.get("splits", {})

    @property
    def indexing(self) -> dict[str, Any]:
        return self.raw.get("indexing", {})

    @property
    def features(self) -> dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def dex(self) -> dict[str, Any]:
        return self.features.get("dex", {})

    @property
    def mldp(self) -> dict[str, Any]:
        return self.raw.get("mldp", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def cascade(self) -> dict[str, Any]:
        return self.raw.get("cascade", {})

    @property
    def baseline(self) -> dict[str, Any]:
        return self.raw.get("baseline", {})

    @property
    def export(self) -> dict[str, Any]:
        return self.raw.get("export", {})

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def preprocess(self) -> dict[str, Any]:
        return self.raw.get("preprocess", {})


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
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

    shared_raw = paths_raw.get("shared_manifest_csv")
    shared_manifest = _resolve_path(root, shared_raw) if shared_raw else None

    paths = PathsConfig(
        apk_root=_resolve_path(root, apk_root),
        shared_manifest_csv=shared_manifest,
        deployed_mlp_header_bundle=_resolve_path(
            root,
            paths_raw.get(
                "deployed_mlp_header_bundle",
                "../vigidroid/app/src/main/assets/models/mlp_header",
            ),
        ),
        artifacts=_resolve_path(root, paths_raw.get("artifacts", "artifacts")),
        manifests_dir=_resolve_path(
            root, paths_raw.get("manifests_dir", "artifacts/manifests")
        ),
        dataset_index=_resolve_path(
            root, paths_raw.get("dataset_index", "artifacts/manifests/apk_index.csv")
        ),
        splits_dir=_resolve_path(root, paths_raw.get("splits_dir", "artifacts/splits")),
        processed=_resolve_path(root, paths_raw.get("processed", "artifacts/processed")),
        checkpoints=_resolve_path(
            root, paths_raw.get("checkpoints", "artifacts/checkpoints")
        ),
        metrics=_resolve_path(root, paths_raw.get("metrics", "artifacts/metrics")),
        export=_resolve_path(
            root,
            paths_raw.get("export", "artifacts/export/mldp_dexheader_cascade"),
        ),
        failed_index_log=_resolve_path(
            root, paths_raw.get("failed_index_log", "artifacts/failed_index.log")
        ),
        failed_apks_log=_resolve_path(
            root, paths_raw.get("failed_apks_log", "artifacts/failed_apks.log")
        ),
    )
    return PipelineConfig(root=root, paths=paths, raw=raw)


def ensure_artifact_dirs(cfg: PipelineConfig) -> None:
    for path in (
        cfg.paths.manifests_dir,
        cfg.paths.splits_dir,
        cfg.paths.processed,
        cfg.paths.checkpoints,
        cfg.paths.metrics,
        cfg.paths.export,
        cfg.paths.failed_index_log.parent,
        cfg.paths.failed_apks_log.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
