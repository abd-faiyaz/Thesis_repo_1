#!/usr/bin/env python3
"""Shared helpers for thesis pipeline run archives (output_archives/<run_id>/)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_SCRIPTS = Path(__file__).resolve().parent
if str(_REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_REPO_SCRIPTS))

from thesis_run_logging import ARCHIVE_PROFILES, ArchiveProfile

CANONICAL_APK_ROOT = "/mnt/Files/FromLaptop/thesis_full_dataset"

# Alias for backward compatibility in this module
PROFILES: dict[str, ArchiveProfile] = ARCHIVE_PROFILES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def archive_dir_for(root: Path, run_id: str) -> Path:
    return root / "output_archives" / run_id


def resolve_run_id(root: Path, profile: ArchiveProfile, run_id: str | None) -> str:
    if run_id:
        return run_id
    env_id = os.environ.get(profile.env_run_id, "").strip()
    if env_id:
        return env_id
    latest = root / "output_archives" / "LATEST_RUN.txt"
    if latest.is_file():
        return latest.read_text(encoding="utf-8").strip()
    raise ValueError(f"Run id required (--run-id or {profile.env_run_id})")


def bootstrap_archive(
    root: Path,
    profile: ArchiveProfile,
    run_id: str,
    *,
    config_snapshot: Path | None = None,
    apk_root: str | None = None,
) -> Path:
    archive = archive_dir_for(root, run_id)
    for sub in ("logs", "metrics", "corpus_stats", "figures", "config", "export", "parity"):
        (archive / sub).mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "model_id": profile.model_id,
        "display_name": profile.display_name,
        "created_at": _utc_now(),
        "git_commit": _git_commit(root),
        "apk_root": apk_root or os.environ.get("APK_ROOT", CANONICAL_APK_ROOT),
    }
    (archive / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if config_snapshot and config_snapshot.is_file():
        dest = archive / "config" / "default.yaml.snapshot"
        shutil.copy2(config_snapshot, dest)

    (root / "output_archives" / "LATEST_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    return archive


def _copy_file(src: Path, dest: Path) -> None:
    if not src.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_tree(src: Path, dest: Path) -> None:
    if not src.is_dir():
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _copy_glob(src_dir: Path, pattern: str, dest_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    for path in sorted(src_dir.glob(pattern)):
        if path.is_file():
            _copy_file(path, dest_dir / path.name)


def sync_artifacts(root: Path, profile: ArchiveProfile, run_id: str) -> Path:
    archive = archive_dir_for(root, run_id)
    metrics_dest = archive / "metrics"
    metrics_dest.mkdir(parents=True, exist_ok=True)

    metric_sources = [root / "artifacts" / "metrics"]
    if profile.metrics_subdir != "metrics":
        metric_sources.append(root / "artifacts" / profile.metrics_subdir)

    metric_names = (
        "epochs.jsonl",
        "test_results.json",
        "evaluation_results.json",
        "training_run_info.json",
        "training_meta.json",
        "preprocess_summary.json",
        "checkpoint_summary.json",
        "corpus_stats.json",
        "parity_report.json",
        "thresholds.json",
        "metrics_val.json",
        "metrics_test.json",
        "val_scores.json",
    )
    for metrics_src in metric_sources:
        for name in metric_names:
            _copy_file(metrics_src / name, metrics_dest / name)

    _copy_file(root / profile.checkpoint_rel, archive / "checkpoints" / Path(profile.checkpoint_rel).name)
    _copy_tree(root / profile.export_rel, archive / "export")

    parity_src = metrics_src / "parity_report.json"
    if parity_src.is_file():
        _copy_file(parity_src, archive / "parity" / "parity_report.json")

    return archive


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except PermissionError as exc:
        hint = (
            f"Cannot write {path} ({exc}).\n"
            "This usually means the file or parent directory was created by root "
            "(e.g. an earlier run with sudo or inside a root-owned container).\n"
            f"Fix: sudo chown -R \"${{USER}}:${{USER}}\" {path.parent.parent}"
        )
        raise PermissionError(hint) from exc


def export_corpus_stats(root: Path, profile: ArchiveProfile, run_id: str | None = None) -> dict[str, Any]:
    label_dist: dict[str, int] = {"total": 0, "benign": 0, "malware": 0}
    year_counts: dict[str, int] = {}

    if profile.corpus_source == "apk_index_summary":
        summary_path = root / "artifacts/manifests/apk_index_summary.json"
        if summary_path.is_file():
            summary = _load_json(summary_path)
            splits = summary.get("splits") or {}
            for split_payload in splits.values():
                label_dist["benign"] += int(split_payload.get("benign", 0))
                label_dist["malware"] += int(split_payload.get("malware", 0))
                label_dist["total"] += int(split_payload.get("total", 0))
            payload = {
                "source": str(summary_path),
                "label_distribution": label_dist,
                "year_folder_counts": year_counts,
                "split_summary": splits,
                "split_policy": summary.get("split_policy"),
            }
        else:
            payload = {"source": str(summary_path), "error": "summary not found"}
    else:
        index_path = root / "artifacts/dataset_index.csv"
        if index_path.is_file():
            with index_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    label_dist["total"] += 1
                    label = str(row.get("label", "")).strip()
                    if label in {"0", "benign"}:
                        label_dist["benign"] += 1
                    elif label in {"1", "malware"}:
                        label_dist["malware"] += 1
                    year = (row.get("year") or "").strip()
                    if year:
                        year_counts[year] = year_counts.get(year, 0) + 1
            payload = {
                "source": str(index_path),
                "label_distribution": label_dist,
                "year_folder_counts": dict(sorted(year_counts.items())),
            }
        else:
            payload = {"source": str(index_path), "error": "dataset index not found"}

    payload["timestamp"] = _utc_now()
    _write_json(root / "artifacts" / "metrics" / "corpus_stats.json", payload)

    if run_id:
        archive = archive_dir_for(root, run_id)
        _write_json(archive / "corpus_stats" / "label_distribution.json", label_dist)
        _write_json(archive / "corpus_stats" / "year_folder_counts.json", payload.get("year_folder_counts", {}))
        if payload.get("split_summary"):
            _write_json(archive / "corpus_stats" / "split_summary.json", payload["split_summary"])

    return payload


def finalize_archive(
    root: Path,
    profile: ArchiveProfile,
    run_id: str,
    *,
    apk_root: str | None = None,
) -> Path:
    sync_artifacts(root, profile, run_id)
    archive = archive_dir_for(root, run_id)

    entries: list[tuple[str, Path]] = []

    def add(label: str, path: Path) -> None:
        if path.is_file():
            entries.append((label, path))

    add("checkpoint", root / profile.checkpoint_rel)
    metrics_dir = root / "artifacts" / "metrics"
    for name in (
        "epochs.jsonl",
        "test_results.json",
        "evaluation_results.json",
        "training_run_info.json",
        "training_meta.json",
        "corpus_stats.json",
        "parity_report.json",
    ):
        add(f"metrics/{name}", metrics_dir / name)

    for rel in (
        "metrics/test_results.json",
        "metrics/evaluation_results.json",
        "metrics/epochs.jsonl",
        "metrics/corpus_stats.json",
        "metrics/parity_report.json",
        "logs/pipeline_full.log",
        "config/default.yaml.snapshot",
        "corpus_stats/label_distribution.json",
        "corpus_stats/year_folder_counts.json",
        "export/export_manifest.json",
        "export/model.onnx",
        "parity/parity_report.json",
    ):
        add(f"archive/{rel}", archive / rel)

    manifest_path = archive / "RUN_MANIFEST.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)

    manifest.update(
        {
            "run_id": run_id,
            "model_id": profile.model_id,
            "display_name": profile.display_name,
            "finalized_at": _utc_now(),
            "canonical_apk_root": apk_root or CANONICAL_APK_ROOT,
            "apk_root": apk_root or CANONICAL_APK_ROOT,
            "git_commit": manifest.get("git_commit") or _git_commit(root),
        }
    )

    test_path = archive / "metrics" / "test_results.json"
    if test_path.is_file():
        test_data = _load_json(test_path)
        manifest["final_test_metrics"] = test_data.get("metrics")
        manifest["n_test_samples"] = test_data.get("n_samples")
        manifest["eval_split"] = test_data.get("split")

    train_info = archive / "metrics" / "training_run_info.json"
    if train_info.is_file():
        manifest["training"] = _load_json(train_info)
    else:
        meta = archive / "metrics" / "training_meta.json"
        if meta.is_file():
            manifest["training"] = _load_json(meta)

    corpus_stats = archive / "metrics" / "corpus_stats.json"
    if corpus_stats.is_file():
        manifest["preprocessing"] = _load_json(corpus_stats)

    manifest["artifact_paths"] = {
        "checkpoint": profile.checkpoint_rel,
        "export_dir": profile.export_rel,
        "metrics_dir": "artifacts/metrics",
    }
    manifest["verify_checksums"] = (
        f"cd {root} && sha256sum -c output_archives/{run_id}/RUN_MANIFEST.sha256"
    )

    index: dict[str, dict[str, str]] = {}
    lines: list[str] = []
    for label, path in entries:
        digest = _sha256(path)
        rel_repo = _rel(path, root)
        lines.append(f"{digest}  {rel_repo}")
        index[label] = {"path": rel_repo, "sha256": digest}

    manifest["archive_phase1"] = {
        "completed_at": _utc_now(),
        "checksum_file": "RUN_MANIFEST.sha256",
        "artifact_index": index,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    manifest_rel = _rel(manifest_path, root)
    lines.append(f"{_sha256(manifest_path)}  {manifest_rel}")

    checksum_path = archive / "RUN_MANIFEST.sha256"
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "output_archives" / "LATEST_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    return checksum_path


def _load_eval_metrics(archive: Path) -> dict[str, Any]:
    for name in ("test_results.json", "metrics_test.json", "evaluation_results.json"):
        path = archive / "metrics" / name
        if not path.is_file():
            continue
        data = _load_json(path)
        if name == "evaluation_results.json":
            splits = data.get("splits") or {}
            for key in ("temporal_holdout", "test"):
                if key in splits:
                    return {
                        "metrics": splits[key]["metrics"],
                        "confusion_matrix": splits[key].get("confusion_matrix"),
                        "n_samples": splits[key].get("n_samples"),
                        "split": key,
                    }
            continue
        return data
    raise FileNotFoundError(f"No eval metrics under {archive / 'metrics'}")


def _features_section(profile: ArchiveProfile) -> str:
    key = profile.profile_key
    if key == "early_fusion_dex_manifest":
        return """| Item | Value |
|------|-------|
| Domain | `dex_header_manifest` |
| Dex header | 104-D (bytes 8–111, normalized), multidex **sum** |
| Manifest | BoW multihot, lexicon 4380 + UNK → 4381-D |
| Fusion | Concat(H, I) → ASCNN → MLP head |
| Input length | 4485 (padded to 4488 for ASCNN) |"""
    if key == "dual_branch_dex_manifest":
        return """| Item | Value |
|------|-------|
| Domain | `dex_header_manifest_dual` |
| Dex branch | MLP(H) on 104-D header (multidex sum) |
| Manifest branch | ASCNN on 4381-D BoW |
| Fusion | Dual-branch merge head (late fusion) |"""
    if key == "linregdroid_permission":
        return """| Item | Value |
|------|-------|
| Domain | `manifest_permissions` |
| Features | Full permission vocab (~173-D multihot) |
| Model | Single sklearn MLR (LinRegDroid1: clamp(ŷ,0,1) ≥ threshold) |
| Split | Stratified dev 2020–2021 + **temporal_holdout** 2022–2023 (primary test) |"""
    if key == "mldp_pruned_permission":
        return """| Item | Value |
|------|-------|
| Domain | `manifest_permissions_mldp` |
| Features | MLDP-pruned permission set S (≈20–40 dims) |
| Model | LinearSVC or Tiny MLP (best val F1) |
| Split | Same as LinRegDroid (stratified dev + temporal holdout test) |"""
    if key == "broadcast_mldp_hybrid":
        return """| Item | Value |
|------|-------|
| Domain | `manifest_mldp_perm_receiver_actions` |
| Permissions | MLDP-pruned set S (train-only) |
| Receivers | Static manifest broadcast system actions |
| Model | Early-fusion tiny MLP (64 hidden) |
| Split | Train 2020–2021; val/test stratified from 2022–2023 |"""
    if key == "mldp_dexheader_cascade":
        return """| Item | Value |
|------|-------|
| Domain | `manifest_mldp_perm_dex_header` |
| Permissions | MLDP-pruned set S (≈22, train-only) |
| Dex header | 104-D (bytes 8–111 /255, multidex sum, corpus min–max) |
| Mode A | Early fusion `x = [x_S ‖ H]` → MLP(d→64→1) |
| Mode B | Stage-1 MLDP logistic; Stage-2 deployed MLP(H) on uncertain band |
| Split | Train 2020–2021; val 10% from train; test all 2022–2023 |"""
    return f"| model_id | `{profile.model_id}` |"


def _split_policy_note(profile: ArchiveProfile) -> str:
    key = profile.profile_key
    if key in ("early_fusion_dex_manifest", "dual_branch_dex_manifest", "mlp_header"):
        return (
            "Temporal split: **train 2020–2021**, val ~10% from train years, "
            "**test 2022–2023** (reported metrics on test only)."
        )
    if key in ("linregdroid_permission", "mldp_pruned_permission"):
        return (
            "Stratified **train/val/dev_test** from 2020–2021; primary test is "
            "**temporal_holdout** (2022–2023) in `test_results.json`."
        )
    if key == "broadcast_mldp_hybrid":
        return "Train 2020–2021; val and test are disjoint stratified halves of 2022–2023."
    if key == "mldp_dexheader_cascade":
        return (
            "Temporal split: **train 2020–2021**, val 10% holdout from train years, "
            "**test all 2022–2023**. MLDP S and dex min/max fit on train only."
        )
    if key == "dexheader_broadcast_fusion":
        return (
            "Temporal split: **train 2020–2021**, val from train years, "
            "**test 2022–2023**. Dex header + broadcast receiver fusion."
        )
    return "See config snapshot for split policy."


def generate_thesis_snippet(root: Path, profile: ArchiveProfile, run_id: str) -> str:
    archive = archive_dir_for(root, run_id)
    manifest = _load_json(archive / "RUN_MANIFEST.json")
    labels_path = archive / "corpus_stats" / "label_distribution.json"
    labels = _load_json(labels_path) if labels_path.is_file() else {}
    eval_metrics = _load_eval_metrics(archive)
    metrics = eval_metrics.get("metrics") or {}
    cm = eval_metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    if not metrics and "mode_a" in eval_metrics:
        mode_a = eval_metrics.get("mode_a") or {}
        metrics = {
            "accuracy": mode_a.get("accuracy"),
            "f1": mode_a.get("f1"),
            "roc_auc": mode_a.get("roc_auc"),
        }
        cm = mode_a.get("confusion_matrix") or cm
    elif not metrics and "accuracy" in eval_metrics:
        metrics = {
            "accuracy": eval_metrics.get("accuracy"),
            "f1": eval_metrics.get("f1"),
            "roc_auc": eval_metrics.get("roc_auc"),
        }
    eval_split = eval_metrics.get("split", "test")

    pre = manifest.get("preprocessing") or {}
    train = manifest.get("training") or {}
    apk_root = manifest.get("canonical_apk_root") or manifest.get("apk_root", "—")
    git_commit = (manifest.get("git_commit") or "—")[:12]
    pkg = profile.package_dir or root.name

    year_counts = {}
    year_path = archive / "corpus_stats" / "year_folder_counts.json"
    if year_path.is_file():
        year_counts = _load_json(year_path)
    year_rows = "\n".join(f"| {y} | {c:,} |" for y, c in sorted(year_counts.items()))

    parity_line = "Parity not run."
    parity_path = archive / "parity" / "parity_report.json"
    if parity_path.is_file():
        parity = _load_json(parity_path)
        pt_onnx = parity.get("pytorch_vs_onnx") or {}
        max_diff = pt_onnx.get("max_abs_diff", pt_onnx.get("max_delta"))
        if max_diff is not None:
            parity_line = (
                f"PyTorch vs ONNX max abs diff = {float(max_diff):.2e} "
                f"({'PASS' if parity.get('passed') else 'FAIL'})."
            )

    export_manifest_path = archive / "export" / "export_manifest.json"
    onnx_line = "ONNX export not found in archive."
    if export_manifest_path.is_file():
        exp = _load_json(export_manifest_path)
        inputs = exp.get("inputs") or [{}]
        onnx_line = (
            f"`model.onnx` opset {exp.get('opset_version', '—')}, "
            f"input shape {inputs[0].get('shape', '—')} → malware probability."
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    loss_val = eval_metrics.get("loss", manifest.get("final_test_loss", "—"))
    threshold = eval_metrics.get("threshold", 0.5)
    train_samples = train.get("n_train", train.get("train_samples", "—"))
    selected = train.get("selected_model", train.get("variant", "—"))

    return f"""# {profile.display_name} — Thesis snippet

**Run:** `{run_id}` · **Generated:** {generated}  
**Archive:** `output_archives/{run_id}/`  
**Live artifacts:** `artifacts/` (latest working copy)  
**Reproducibility:** `RUN_MANIFEST.json` + `RUN_MANIFEST.sha256`

---

## 1. Dataset

| Item | Value |
|------|-------|
| APK root (canonical) | `{apk_root}` |
| Benign / malware | {labels.get('benign', 0):,} / {labels.get('malware', 0):,} |
| Total indexed | {labels.get('total', 0):,} |
| Labeling | Parent folder (`benign` vs `malware`) |

| Year | APK count |
|------|-----------|
{year_rows or "| — | — |"}

**Split policy:** {_split_policy_note(profile)}

---

## 2. Features & model

{_features_section(profile)}

**Deployment:** {onnx_line}

---

## 3. Training

| Item | Value |
|------|-------|
| Checkpoint | `{profile.checkpoint_rel}` |
| Train samples | {train_samples} |
| Selected variant | {selected} |
| Git commit | `{git_commit}` |

Config snapshot: `output_archives/{run_id}/config/default.yaml.snapshot`.

---

## 4. Test results ({eval_split} split)

| Metric | Value |
|--------|-------|
| Accuracy | {float(metrics.get('accuracy') or 0):.4f} |
| F1 (malware) | {float(metrics.get('f1') or 0):.4f} |
| ROC-AUC | {float(metrics.get('roc_auc') or 0):.4f} |
| Loss | {loss_val} |
| Threshold | {threshold} |
| Test samples | {eval_metrics.get('n_samples', 0):,} |

**Confusion matrix** (rows=true, cols=predicted; benign first):

| | Pred benign | Pred malware |
|---|-------------|--------------|
| True benign | {cm[0][0]} | {cm[0][1]} |
| True malware | {cm[1][0]} | {cm[1][1]} |

Figures: `output_archives/{run_id}/figures/` (`loss_curves.png`, `metrics_vs_epoch.png`, `roc_curve_val.png`, `confusion_matrix_val.png`, corpus plots).

---

## 5. Export & parity

| Check | Result |
|-------|--------|
| ONNX bundle | `output_archives/{run_id}/export/` |
| Parity | {parity_line} |

---

## 6. Limitations

- Corpus ~13.5k APKs; do not claim full MSFDroid-scale numbers without retraining.
- Failed APKs: see `artifacts/failed_apks.log` if any.
- **Live vs archive:** `artifacts/` holds the latest run; `output_archives/{run_id}/` is an immutable snapshot when archive mode is enabled.

---

## Appendix: verify archive

```bash
cd {pkg}
sha256sum -c output_archives/{run_id}/RUN_MANIFEST.sha256
```
"""


def plot_results(root: Path, profile: ArchiveProfile, run_id: str) -> list[Path]:
    from plot_thesis_results import generate_figures

    return generate_figures(root, profile, run_id)


def _add_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        required=True,
        help="Model archive profile key",
    )


def _root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Model package root (default: caller script parent/..)",
    )


def _run_id_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", type=str, default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thesis run archive utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_boot = sub.add_parser("bootstrap")
    _add_profile_arg(p_boot)
    _root_arg(p_boot)
    p_boot.add_argument("--run-id", type=str, required=True)
    p_boot.add_argument("--config", type=Path, default=None)
    p_boot.add_argument("--apk-root", type=str, default=None)

    p_stats = sub.add_parser("export-corpus-stats")
    _add_profile_arg(p_stats)
    _root_arg(p_stats)
    _run_id_arg(p_stats)

    p_sync = sub.add_parser("sync")
    _add_profile_arg(p_sync)
    _root_arg(p_sync)
    p_sync.add_argument("--run-id", type=str, required=True)

    p_fin = sub.add_parser("finalize")
    _add_profile_arg(p_fin)
    _root_arg(p_fin)
    _run_id_arg(p_fin)
    p_fin.add_argument("--apk-root", type=str, default=None)

    p_plot = sub.add_parser("plot")
    _add_profile_arg(p_plot)
    _root_arg(p_plot)
    _run_id_arg(p_plot)

    p_snip = sub.add_parser("snippet")
    _add_profile_arg(p_snip)
    _root_arg(p_snip)
    _run_id_arg(p_snip)
    p_snip.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    root = (args.root or Path.cwd()).resolve()

    if args.command == "bootstrap":
        bootstrap_archive(
            root,
            profile,
            args.run_id,
            config_snapshot=args.config,
            apk_root=args.apk_root,
        )
        print(f"Bootstrapped output_archives/{args.run_id}/")
        return 0

    if args.command == "export-corpus-stats":
        stats = export_corpus_stats(root, profile, getattr(args, "run_id", None))
        print(json.dumps(stats, indent=2))
        return 0

    run_id = resolve_run_id(root, profile, getattr(args, "run_id", None))

    if args.command == "sync":
        sync_artifacts(root, profile, run_id)
        print(f"Synced artifacts → output_archives/{run_id}/")
        return 0

    if args.command == "finalize":
        out = finalize_archive(root, profile, run_id, apk_root=args.apk_root)
        print(f"Wrote {out}")
        return 0

    if args.command == "plot":
        paths = plot_results(root, profile, run_id)
        for path in paths:
            print(f"Wrote {path}")
        return 0

    if args.command == "snippet":
        text = generate_thesis_snippet(root, profile, run_id)
        out = args.out or archive_dir_for(root, run_id) / "THESIS_SNIPPET.md"
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
