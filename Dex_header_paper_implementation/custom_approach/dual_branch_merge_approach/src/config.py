"""Load and resolve paths from YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PACKAGE_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class PathsConfig:
    apk_root: Path
    dataset_index: Path
    processed_dir: Path
    checkpoint_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    failed_apks_log: Path
    normalization_stats: Path
    vocab: Path
    splits_dir: Path
    class_balance: Path
    pipeline_log: Path
    artifacts_bundle: Path

    @property
    def shards_train_dir(self) -> Path:
        return self.processed_dir / "shards" / "train"

    @property
    def shards_val_dir(self) -> Path:
        return self.processed_dir / "shards" / "val"

    @property
    def shards_test_dir(self) -> Path:
        return self.processed_dir / "shards" / "test"

    @property
    def processed_ids_log(self) -> Path:
        return self.processed_dir / "processed_ids.txt"

    @property
    def manifest_train(self) -> Path:
        return self.processed_dir / "manifest_train.json"

    @property
    def manifest_val(self) -> Path:
        return self.processed_dir / "manifest_val.json"

    @property
    def manifest_test(self) -> Path:
        return self.processed_dir / "manifest_test.json"


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved configuration for the dual-branch pipeline."""

    root: Path
    paths: PathsConfig
    raw: dict[str, Any]

    @property
    def preprocessing(self) -> dict[str, Any]:
        return self.raw.get("preprocessing", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw.get("evaluation", {})


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    """Load YAML config and resolve relative paths against the package root."""
    cfg_file = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not cfg_file.is_absolute():
        cfg_file = (_PACKAGE_ROOT / cfg_file).resolve()

    with cfg_file.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    root = _PACKAGE_ROOT
    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        apk_root=_resolve_path(root, paths_raw["apk_root"]),
        dataset_index=_resolve_path(root, paths_raw["dataset_index"]),
        processed_dir=_resolve_path(root, paths_raw["processed_dir"]),
        checkpoint_dir=_resolve_path(root, paths_raw["checkpoint_dir"]),
        latest_checkpoint=_resolve_path(root, paths_raw["latest_checkpoint"]),
        best_checkpoint=_resolve_path(root, paths_raw["best_checkpoint"]),
        failed_apks_log=_resolve_path(root, paths_raw["failed_apks_log"]),
        normalization_stats=_resolve_path(root, paths_raw["normalization_stats"]),
        vocab=_resolve_path(root, paths_raw["vocab"]),
        splits_dir=_resolve_path(root, paths_raw["splits_dir"]),
        class_balance=_resolve_path(root, paths_raw["class_balance"]),
        pipeline_log=_resolve_path(root, paths_raw["pipeline_log"]),
        artifacts_bundle=_resolve_path(root, paths_raw["artifacts_bundle"]),
    )
    return PipelineConfig(root=root, paths=paths, raw=raw)


def ensure_artifact_dirs(cfg: PipelineConfig) -> None:
    """Create output directories used by preprocessing and training."""
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.shards_train_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.shards_val_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.shards_test_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.splits_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.failed_apks_log.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.dataset_index.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.vocab.parent.mkdir(parents=True, exist_ok=True)
