#!/usr/bin/env python3
"""Verify LinRegDroid environment and config."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("torch", "torch"),
    ("pyaxmlparser", "pyaxmlparser"),
]


def main() -> int:
    print("LinRegDroid permission classifier — environment check\n")
    print(f"Project root: {ROOT}\n")

    errors: list[str] = []
    for module, pip_name in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            errors.append(f"  - {pip_name} (import {module})")

    if errors:
        print("Missing dependencies:")
        print("\n".join(errors))
        print("\nInstall: pip install -r requirements.txt")
        return 1

    from src.config import ensure_artifact_dirs, load_config
    from src.features.permission_vector import normalize_permission

    cfg = load_config()
    ensure_artifact_dirs(cfg)
    sample = normalize_permission("android.permission.SEND_SMS")
    if sample != "permissions::send_sms":
        print(f"Unexpected normalization: {sample}")
        return 1

    apk_root = cfg.paths.apk_root
    if not apk_root.is_dir():
        print(f"ERROR: apk_root not found: {apk_root}")
        print("  Set paths.apk_root in config/default.yaml or export APK_ROOT=/path/to/corpus")
        return 1

    apk_count = sum(1 for _ in apk_root.rglob("*.apk"))
    if apk_count == 0:
        print(f"ERROR: no .apk files under {apk_root}")
        return 1

    print("Dependencies OK.")
    print(f"  apk_root:     {apk_root} ({apk_count} APKs)")
    print(f"  model_id:     {cfg.pipeline.get('model_id')}")
    print(f"  vocab path:   {cfg.paths.permission_vocab}")
    print(f"  VigiDroid token example: android.permission.SEND_SMS → {sample}")
    print("\nSetup verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
