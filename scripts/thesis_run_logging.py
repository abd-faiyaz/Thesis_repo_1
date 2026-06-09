#!/usr/bin/env python3
"""Live mirror from artifacts/ → output_archives/<run_id>/ during pipeline runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchiveProfile:
    profile_key: str
    model_id: str
    display_name: str
    env_run_id: str
    checkpoint_rel: str
    export_rel: str
    metrics_subdir: str = "metrics"
    corpus_source: str = "dataset_index_csv"  # or apk_index_summary
    package_dir: str = ""
    test_results_candidates: tuple[str, ...] = (
        "artifacts/metrics/test_results.json",
        "artifacts/checkpoints/test_results.json",
    )


ARCHIVE_PROFILES: dict[str, ArchiveProfile] = {
    "mlp_header": ArchiveProfile(
        profile_key="mlp_header",
        model_id="mlp_header",
        display_name="Base Model 1 (MLP-H)",
        env_run_id="BM1_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/latest_checkpoint.pth",
        export_rel="artifacts/export/mlp_header",
    ),
    "early_fusion_dex_manifest": ArchiveProfile(
        profile_key="early_fusion_dex_manifest",
        model_id="early_fusion_dex_manifest",
        display_name="Early-Fusion Dex+Manifest",
        env_run_id="PA_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/best.pt",
        export_rel="artifacts/export/early_fusion_dex_manifest",
        metrics_subdir="checkpoints",
        corpus_source="dataset_index_csv",
        package_dir="Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach",
        test_results_candidates=("artifacts/checkpoints/test_results.json",),
    ),
    "dual_branch_dex_manifest": ArchiveProfile(
        profile_key="dual_branch_dex_manifest",
        model_id="dual_branch_dex_manifest",
        display_name="Dual-Branch Dex+Manifest",
        env_run_id="PB_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/best.pt",
        export_rel="artifacts/export/dual_branch_dex_manifest",
        metrics_subdir="checkpoints",
        corpus_source="dataset_index_csv",
        package_dir="Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach",
        test_results_candidates=("artifacts/checkpoints/test_results.json",),
    ),
    "broadcast_mldp_hybrid": ArchiveProfile(
        profile_key="broadcast_mldp_hybrid",
        model_id="broadcast_mldp_hybrid",
        display_name="Broadcast + MLDP Hybrid",
        env_run_id="BMH_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/best.pt",
        export_rel="artifacts/export/broadcast_mldp_hybrid",
        corpus_source="apk_index_summary",
        package_dir="broadcast_mldp_hybrid",
    ),
    "mldp_dexheader_cascade": ArchiveProfile(
        profile_key="mldp_dexheader_cascade",
        model_id="mldp_dexheader_cascade",
        display_name="MLDP + Dex Header Cascade",
        env_run_id="MDH_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/mode_a_best.pt",
        export_rel="artifacts/export/mldp_dexheader_cascade",
        corpus_source="apk_index_summary",
        package_dir="mldp_dexheader_cascade",
        test_results_candidates=("artifacts/metrics/test_results.json",),
    ),
    "linregdroid_permission": ArchiveProfile(
        profile_key="linregdroid_permission",
        model_id="linregdroid_permission",
        display_name="LinRegDroid (permission-only)",
        env_run_id="LR_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/linregdroid.pth",
        export_rel="artifacts/export/linregdroid_permission",
        package_dir="linear",
    ),
    "mldp_pruned_permission": ArchiveProfile(
        profile_key="mldp_pruned_permission",
        model_id="mldp_pruned_permission",
        display_name="MLDP-pruned permission classifier",
        env_run_id="MLDP_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/mldp_pruned.pth",
        export_rel="artifacts/export/mldp_pruned_permission",
        package_dir="permission_extractor",
    ),
    "dexheader_broadcast_fusion": ArchiveProfile(
        profile_key="dexheader_broadcast_fusion",
        model_id="dexheader_broadcast_fusion",
        display_name="Dex Header + Broadcast Fusion",
        env_run_id="DBF_RUN_ID",
        checkpoint_rel="artifacts/checkpoints/best.pt",
        export_rel="artifacts/export/dexheader_broadcast_fusion",
        corpus_source="dataset_index_csv",
        package_dir="dexheader_broadcast_fusion",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(model_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(model_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class RunArchive:
    """Write to artifacts/ (live) and mirror copies into output_archives/<run_id>/."""

    def __init__(self, model_root: Path, profile: ArchiveProfile):
        self.model_root = model_root.resolve()
        self.profile = profile

    @classmethod
    def for_profile(cls, model_root: Path, profile_key: str) -> RunArchive:
        return cls(model_root, ARCHIVE_PROFILES[profile_key])

    @property
    def run_id(self) -> str | None:
        value = os.environ.get(self.profile.env_run_id, "").strip()
        return value or None

    @property
    def enabled(self) -> bool:
        return self.run_id is not None

    def artifacts_dir(self) -> Path:
        return self.model_root / "artifacts"

    def metrics_dir(self) -> Path:
        path = self.artifacts_dir() / self.profile.metrics_subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def archive_dir(self) -> Path | None:
        run_id = self.run_id
        if not run_id:
            return None
        root = self.model_root / "output_archives" / run_id
        for sub in ("logs", "metrics", "corpus_stats", "figures", "config", "export", "parity"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        return root

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.metrics_dir() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.mirror_file(path, f"metrics/{filename}")
        return path

    def append_jsonl(self, filename: str, record: dict[str, Any]) -> Path:
        path = self.metrics_dir() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.mirror_file(path, f"metrics/{filename}")
        return path

    def mirror_file(self, src: Path, archive_rel: str) -> Path | None:
        archive = self.archive_dir()
        if archive is None or not src.is_file():
            return None
        dest = archive / archive_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def mirror_tree(self, src_dir: Path, archive_subdir: str) -> Path | None:
        archive = self.archive_dir()
        if archive is None or not src_dir.is_dir():
            return None
        dest = archive / archive_subdir
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_dir, dest)
        return dest

    def mirror_checkpoint(self, checkpoint_path: Path | None = None) -> Path | None:
        ckpt = checkpoint_path or (self.model_root / self.profile.checkpoint_rel)
        if not ckpt.is_file():
            return None
        return self.mirror_file(ckpt, f"checkpoints/{ckpt.name}")

    def mirror_export_bundle(self) -> Path | None:
        export_dir = self.model_root / self.profile.export_rel
        return self.mirror_tree(export_dir, "export")

    def mirror_parity_report(self, report_path: Path | None = None) -> Path | None:
        candidates = [
            report_path,
            self.metrics_dir() / "parity_report.json",
            self.artifacts_dir() / "parity" / "parity_report.json",
        ]
        for path in candidates:
            if path and path.is_file():
                return self.mirror_file(path, "parity/parity_report.json")
        return None

    def mirror_test_results(self) -> Path | None:
        for rel in self.profile.test_results_candidates:
            path = self.model_root / rel
            if path.is_file():
                return self.mirror_file(path, "metrics/test_results.json")
        return None

    def update_manifest(self, patch: dict[str, Any]) -> Path | None:
        archive = self.archive_dir()
        if archive is None:
            return None
        manifest_path = archive / "RUN_MANIFEST.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(patch)
        manifest.setdefault("run_id", self.run_id)
        manifest.setdefault("model_id", self.profile.model_id)
        manifest.setdefault("display_name", self.profile.display_name)
        manifest["updated_at"] = _utc_now()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path

    def log_training_run_info(self, payload: dict[str, Any]) -> Path:
        body = {"timestamp": _utc_now(), "run_id": self.run_id, **payload}
        return self.write_json("training_run_info.json", body)

    def log_training_meta(self, payload: dict[str, Any]) -> Path:
        body = {"timestamp": _utc_now(), "run_id": self.run_id, **payload}
        return self.write_json("training_meta.json", body)

    def log_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float | None = None,
        val_metrics: dict[str, float] | None = None,
        learning_rate: float | None = None,
    ) -> Path:
        record: dict[str, Any] = {
            "timestamp": _utc_now(),
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": train_loss,
        }
        if val_loss is not None:
            record["val_loss"] = val_loss
        if learning_rate is not None:
            record["learning_rate"] = learning_rate
        if val_metrics:
            record.update(val_metrics)
        return self.append_jsonl("epochs.jsonl", record)

    def finalize_manifest(self) -> Path | None:
        if not self.enabled:
            return None
        patch: dict[str, Any] = {
            "git_commit": _git_commit(self.model_root),
            "artifact_paths": {
                "checkpoint": self.profile.checkpoint_rel,
                "export_dir": self.profile.export_rel,
                "metrics_dir": f"artifacts/{self.profile.metrics_subdir}",
            },
        }
        for rel in self.profile.test_results_candidates:
            path = self.model_root / rel
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                metrics = data.get("metrics")
                if metrics is None and "accuracy" in data:
                    metrics = {
                        "accuracy": data.get("accuracy"),
                        "f1": data.get("f1"),
                        "roc_auc": data.get("roc_auc"),
                    }
                patch["final_test_metrics"] = metrics
                patch["n_test_samples"] = data.get("n_samples")
                patch["eval_split"] = data.get("split", "test")
                break
        train_info = self.metrics_dir() / "training_run_info.json"
        if train_info.is_file():
            patch["training"] = json.loads(train_info.read_text(encoding="utf-8"))
        else:
            meta = self.metrics_dir() / "training_meta.json"
            if meta.is_file():
                patch["training"] = json.loads(meta.read_text(encoding="utf-8"))
        for name in ("preprocess_summary.json", "corpus_stats.json"):
            pre = self.metrics_dir() / name
            if pre.is_file():
                patch["preprocessing"] = json.loads(pre.read_text(encoding="utf-8"))
                break
        return self.update_manifest(patch)
