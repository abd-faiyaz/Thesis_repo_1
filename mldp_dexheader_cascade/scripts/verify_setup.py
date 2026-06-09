#!/usr/bin/env python3
"""P0 — Verify environment, config, corpus, and deployed MLP(H) bundle."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("mlxtend", "mlxtend"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("torch", "torch"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("pyaxmlparser", "pyaxmlparser"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    print("MLDP + Dex header cascade — environment check (P0)\n")
    errors: list[str] = []
    warnings: list[str] = []

    for module, pip_name in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            errors.append(f"  - {pip_name}")

    if errors:
        print("Missing dependencies:\n" + "\n".join(errors))
        print("\nInstall: source scripts/activate_thesis_env.sh")
        print("         pip install -r ../requirements-thesis-all.txt")
        return 1

    from src.config import ensure_artifact_dirs, load_config

    cfg = load_config()
    ensure_artifact_dirs(cfg)

    bundle = cfg.paths.deployed_mlp_header_bundle
    onnx_path = bundle / "model.onnx"
    norm_path = bundle / "features" / "normalization_header.json"
    for required in (onnx_path, norm_path):
        if not required.is_file():
            errors.append(f"  - deployed MLP(H) missing: {required}")

    shared = cfg.paths.shared_manifest_csv
    if shared is not None and not shared.is_file():
        warnings.append(
            f"  - shared manifest not found ({shared}); P1 will build a local index"
        )

    apk_root = cfg.paths.apk_root
    if not apk_root.is_dir():
        errors.append(f"  - apk_root not found: {apk_root}")
    else:
        apk_count = sum(1 for _ in apk_root.rglob("*.apk"))
        if apk_count == 0:
            errors.append(f"  - no .apk files under {apk_root}")

    if errors:
        print("Errors:\n" + "\n".join(errors))
        return 1

    if warnings:
        print("Warnings:\n" + "\n".join(warnings) + "\n")

    apk_count = sum(1 for _ in apk_root.rglob("*.apk"))
    print("Dependencies OK.")
    print(f"  model_id: {cfg.model_id}")
    print(f"  domain:   {cfg.domain}")
    print(f"  apk_root: {apk_root} ({apk_count} APKs)")
    print(f"  train_years: {cfg.splits.get('train_years')}")
    print(f"  holdout_years: {cfg.splits.get('holdout_years', cfg.splits.get('test_years'))}")
    print(f"  val_fraction_of_holdout: {cfg.splits.get('val_fraction_of_holdout')}")
    print(f"  manifest_backend: {cfg.features.get('manifest_backend')}")
    print(f"  dex multidex_mode: {cfg.dex.get('multidex_mode')}")
    print(f"  deployed MLP(H): {onnx_path}")
    print(f"    sha256: {_sha256(onnx_path)}")
    print(f"    normalization: {norm_path}")
    if shared is not None and shared.is_file():
        print(f"  shared manifest: {shared}")
    print("\nP0 exit criteria met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
