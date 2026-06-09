# Pattern B (`dual_branch_dex_manifest`) — Android steps A1 → A4

**Model:** MLP(header) + ASCNN(manifest) fused inside one ONNX (late fusion exported as single graph)  
**Model ID:** `dual_branch_dex_manifest`  
**Status:** **Not implemented** in `vigidroid/`. Complete BM1 first; Pattern A BoW extractor is reusable.

**Rule:** Do **not** change existing ByteCNN / XGBoost. Add Pattern B as a **new** `stages[]` entry.

**Order:** A1 → A2 → **A4** → **A3**

**Note:** ONNX inputs are the **same shape as Pattern A** (`header` 104 + `bow` 4381). Only `model.onnx` and asset paths differ.

---

## Paths

| What | Path |
|------|------|
| PC export bundle | `Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/export/dual_branch_dex_manifest/` |
| Android assets target | `vigidroid/app/src/main/assets/models/dual_branch_dex_manifest/` |
| Python reference | `.../dual_branch_merge_approach/src/features/` |

---

## Phase 0 — Copy export bundle

```bash
cd /mnt/Files/thesis_vigidroid

mkdir -p vigidroid/app/src/main/assets/models/dual_branch_dex_manifest

cp -r Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/export/dual_branch_dex_manifest/* \
  vigidroid/app/src/main/assets/models/dual_branch_dex_manifest/

ls vigidroid/app/src/main/assets/models/dual_branch_dex_manifest/
```

**Optional PC parity check:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach
../../../thesis_venv/bin/python scripts/parity_check_onnx.py \
  --bundle artifacts/export/dual_branch_dex_manifest
```

---

## Phase A1 — Feature extractor

**Same features as Pattern A** (same training shards: header + manifest BoW).

### Tasks to implement in `vigidroid/`

- [ ] **A1.1** Reuse `DexHeaderFeatureExtractor` from BM1
- [ ] **A1.2** Reuse `ManifestBowExtractor` from Pattern A but load vocab from:
  - `assets/models/dual_branch_dex_manifest/features/vocab.json`
- [ ] **A1.3** Do **not** reimplement fusion in Java — fusion is inside ONNX

If Pattern A A1 is done, Pattern B A1 is mostly **pointing at Pattern B asset paths**.

---

## Phase A2 — ONNX inference

### Tasks to implement in `vigidroid/`

- [ ] **A2.1** New `PatternBOnnxRunner.java` (or generic runner keyed by `model_id`)
  - Load `assets/models/dual_branch_dex_manifest/model.onnx`
  - Inputs: `header` `[1,104]`, `bow` `[1,4381]`
  - Output: `malware_probability`
- [ ] **A2.2** Independent from Pattern A session (separate model file)

**Compile:**

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew assembleDebug
```

---

## Phase A4 — Parity test (before A3)

### Tasks to implement in `vigidroid/`

- [ ] **A4.1** Convert parity NPZ → JSON (same script as Pattern A, different path):
  ```bash
  cd /mnt/Files/thesis_vigidroid
  thesis_venv/bin/python - <<'PY'
  import json, numpy as np
  from pathlib import Path
  p = Path("Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/export/dual_branch_dex_manifest/parity_samples")
  d = np.load(p / "sample_vectors.npz")
  out = {k: d[k].tolist() for k in d.files}
  out["index"] = json.loads((p / "index.json").read_text())
  (p / "parity_vectors.json").write_text(json.dumps(out))
  print("keys:", list(d.files))
  PY
  cp Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/export/dual_branch_dex_manifest/parity_samples/parity_vectors.json \
     vigidroid/app/src/main/assets/models/dual_branch_dex_manifest/parity_samples/
  ```
- [ ] **A4.2** `PatternBParityTest.java` — 8 samples, tolerance `1e-4`

### Run on phone

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew installDebug
./gradlew connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.msh.vigidroid.PatternBParityTest
```

---

## Phase A3 — Scans + metrics

### Tasks to implement in `vigidroid/`

- [ ] **A3.1** Register `model_id`: `dual_branch_dex_manifest`, `domain`: `dex_header_manifest_dual`
- [ ] **A3.2** Per APK: header + bow → Pattern B ONNX → append `stages[]`
- [ ] **A3.3** Legacy ByteCNN / XGBoost unchanged

### APK location on phone

All `.apk` files in **`/sdcard/Download/`** — scanned together when you start a scan.

```bash
adb push /path/to/sample.apk /sdcard/Download/
```

### Build, run, pull metrics

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew installDebug
adb shell am start -n com.msh.vigidroid/.MainActivity

cd /mnt/Files/thesis_vigidroid
bash Shared_pipeline_Files/tools/pull_device_metrics.sh
```

---

## Suggested implementation order (all 3 models)

1. **BM1** — full A1–A4  
2. **Pattern A** — BoW extractor + A2/A4/A3  
3. **Pattern B** — reuse A1, new ONNX path + parity test  

You can enable **one model at a time** in scans (config flag) or run all registered models per APK (heavier).

---

## Done checklist

- [ ] Bundle in `assets/models/dual_branch_dex_manifest/`
- [ ] A4 passes (8/8) on POCO F3
- [ ] A3 writes Pattern B stages to `all_scan_metrics.json`
- [ ] CNN/XGB unaffected

---

**Confirm this plan before implementation.**
