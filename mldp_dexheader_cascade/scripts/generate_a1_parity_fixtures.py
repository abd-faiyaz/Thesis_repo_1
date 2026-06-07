#!/usr/bin/env python3
"""Generate androidTest A1 parity APKs + extraction fixtures (3 samples)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features.manifest_decode import decode_manifest
from src.features.normalization import load_normalization_header, transform_vector
from src.features.vectorize import vectorize_cascade
from src.preprocessing.apk_extract import extract_apk_raw_header

REPO_ROOT = ROOT.parent
ANDROID_TEST_PARITY = (
    REPO_ROOT
    / "vigidroid/app/src/androidTest/assets/models/mldp_dexheader_cascade/parity_samples"
)
MAIN_ASSETS = (
    REPO_ROOT / "vigidroid/app/src/main/assets/models/mldp_dexheader_cascade"
)
JVM_FIXTURES = (
    REPO_ROOT
    / "vigidroid/app/src/test/resources/mldp_dexheader_cascade_a1_fixtures.json"
)
NUM_SAMPLES = 3


def _copy_features(export_dir: Path, dest: Path) -> None:
    src = export_dir / "features"
    if not src.is_dir():
        raise FileNotFoundError(f"Missing export features: {src}")
    dest_features = dest / "features"
    if dest_features.exists():
        shutil.rmtree(dest_features)
    shutil.copytree(src, dest_features)


def main() -> int:
    cfg = load_config()
    export_dir = cfg.paths.export
    index_path = export_dir / "parity_samples" / "index.json"
    val_path = cfg.paths.processed / "features_val.pt"

    if not index_path.is_file():
        print(f"ERROR: missing {index_path}", file=sys.stderr)
        return 1
    if not val_path.is_file():
        print(f"ERROR: missing {val_path}", file=sys.stderr)
        return 1

    import torch

    mldp_vocab = json.loads(
        (cfg.paths.processed / "mldp_permission_vocab.json").read_text(encoding="utf-8")
    )["tokens"]
    mins, maxs, _ = load_normalization_header(cfg.paths.processed / "normalization_header.json")
    mode, pattern = (
        str(cfg.dex.get("multidex_mode", "sum")),
        str(cfg.dex.get("dex_pattern", r"^classes(\d*)\.dex$")),
    )
    val = torch.load(val_path, weights_only=False)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    apk_dir = ANDROID_TEST_PARITY / "apks"
    manifest_dir = ANDROID_TEST_PARITY / "manifests"
    apk_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    MAIN_ASSETS.mkdir(parents=True, exist_ok=True)
    _copy_features(export_dir, MAIN_ASSETS)
    thresholds = export_dir / "thresholds.json"
    if thresholds.is_file():
        shutil.copy2(thresholds, MAIN_ASSETS / "thresholds.json")

    fixtures: list[dict] = []
    for sample in index["samples"][:NUM_SAMPLES]:
        sid = sample["sample_id"]
        idx = int(sample["index"])
        apk = Path(val["paths"][idx])
        if not apk.is_file():
            print(f"ERROR: APK not found: {apk}", file=sys.stderr)
            return 1

        sample_dir = export_dir / "parity_samples" / sid
        expected_x_s = np.load(sample_dir / "x_S.npy").astype(np.float32)
        expected_h = np.load(sample_dir / "H.npy").astype(np.float32)
        expected_x = np.load(sample_dir / "x.npy").astype(np.float32)

        parsed = decode_manifest(apk)
        raw_h = extract_apk_raw_header(apk, mode=mode, pattern=pattern)
        h_norm = transform_vector(raw_h, mins=mins, maxs=maxs)
        x_s, h, x = vectorize_cascade(parsed.permissions, h_norm, mldp_vocab=mldp_vocab)

        for name, got, want in (
            ("x_S", x_s, expected_x_s),
            ("H", h, expected_h),
            ("x", x, expected_x),
        ):
            diff = float(np.max(np.abs(got - want)))
            if diff > 1e-6:
                print(f"ERROR: {sid} {name} mismatch max_diff={diff}", file=sys.stderr)
                return 1

        apk_dest = apk_dir / f"{sid}.apk"
        shutil.copy2(apk, apk_dest)

        manifest_out = manifest_dir / f"{sid}.xml"
        with manifest_out.open("wb") as out:
            subprocess.run(
                ["unzip", "-p", str(apk), "AndroidManifest.xml"],
                check=True,
                stdout=out,
            )

        fixtures.append(
            {
                "sample_id": sid,
                "val_index": idx,
                "apk_asset": f"models/mldp_dexheader_cascade/parity_samples/apks/{sid}.apk",
                "manifest_asset": f"models/mldp_dexheader_cascade/parity_samples/manifests/{sid}.xml",
                "permissions": list(parsed.permissions),
                "expected_x_s": x_s.astype(np.float32).tolist(),
                "expected_h": h.astype(np.float32).tolist(),
                "expected_x": x.astype(np.float32).tolist(),
            }
        )
        print(f"  {sid} OK apk={apk_dest.stat().st_size} bytes")

    payload = {
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "tolerance": 1e-4,
        "S": len(mldp_vocab),
        "H": int(cfg.dex.get("feature_dim", 104)),
        "d": len(mldp_vocab) + int(cfg.dex.get("feature_dim", 104)),
        "fixtures": fixtures,
    }
    out_path = ANDROID_TEST_PARITY / "parity_extraction_fixtures.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    JVM_FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    JVM_FIXTURES.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(fixtures)} fixtures)")
    print(f"Wrote {JVM_FIXTURES}")
    print(f"Deployed features → {MAIN_ASSETS / 'features'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
