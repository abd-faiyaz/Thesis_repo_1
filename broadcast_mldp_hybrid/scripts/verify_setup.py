#!/usr/bin/env python3
"""P0 — Verify environment, config, corpus, and system_actions.json."""

from __future__ import annotations

import importlib
import json
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


def main() -> int:
    print("Broadcast + MLDP hybrid — environment check (P0)\n")
    errors: list[str] = []
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

    actions_path = cfg.paths.system_actions_file
    if not actions_path.is_file():
        print(f"ERROR: system_actions.json missing: {actions_path}")
        print("  Run: python scripts/build_system_actions.py")
        return 1

    payload = json.loads(actions_path.read_text(encoding="utf-8"))
    actions = payload.get("actions") or []
    if not actions:
        print(f"ERROR: system_actions.json has no actions: {actions_path}")
        return 1

    apk_root = cfg.paths.apk_root
    if not apk_root.is_dir():
        print(f"ERROR: apk_root not found: {apk_root}")
        print("  Set paths.apk_root in config/default.yaml or export APK_ROOT=...")
        return 1

    apk_count = sum(1 for _ in apk_root.rglob("*.apk"))
    if apk_count == 0:
        print(f"ERROR: no .apk files under {apk_root}")
        return 1

    print("Dependencies OK.")
    print(f"  model_id: {cfg.model_id}")
    print(f"  domain:   {cfg.domain}")
    print(f"  apk_root: {apk_root} ({apk_count} APKs)")
    print(f"  system_actions: {len(actions)} entries ({actions_path.relative_to(ROOT)})")
    print(f"  train_years:    {cfg.splits.get('train_years')}")
    print(f"  holdout_years:  {cfg.splits.get('holdout_years')}")
    print(f"  val_fraction_of_holdout: {cfg.splits.get('val_fraction_of_holdout')}")
    print("\nP0 exit criteria met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
