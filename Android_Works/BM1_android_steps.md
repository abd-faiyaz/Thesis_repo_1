# BM1 (`mlp_header`) — Android steps A1 → A4

**Model:** Dex header only (MSFDroid Base Model 1)  
**Model ID:** `mlp_header`  
**Status:** **Implemented** in `vigidroid/` (2026-06-06). Build in Android Studio on POCO F3.

**Rule:** Do **not** remove or change existing ByteCNN / XGBoost wiring. Add BM1 as a **new** stage in `stages[]`.

**Order:** A1 → A2 → **A4** (parity test) → **A3** (real scans). Do not trust bulk scans until A4 passes.

---

## Paths (this repo)

| What | Path |
|------|------|
| PC export bundle | `Dex_header_paper_implementation/only_base1_model/artifacts/export/mlp_header/` |
| Android assets target | `vigidroid/app/src/main/assets/models/mlp_header/` |
| Python feature reference | `Dex_header_paper_implementation/only_base1_model/src/features/` |
| App project (open in Android Studio) | `vigidroid/` |

---

## Your workflow (after implementation)

1. **Android Studio** → open `vigidroid/` → **Run ▶** on POCO F3 (installs app)
2. **A4** → right-click `MlpHeaderParityTest` → **Run** (must pass 8/8 before trusting scans)
3. **A3** → put APKs in `Internal storage/Download/` → open app → **Start scan**
4. **Pull metrics** → `bash Shared_pipeline_Files/tools/pull_device_metrics.sh`

---

## Phase 0 — Copy export bundle to the app ✅

**Done.** Assets live at `vigidroid/app/src/main/assets/models/mlp_header/` (includes `parity_samples/parity_vectors.json`).

To refresh after re-export on PC:

```bash
cd /mnt/Files/thesis_vigidroid
cp -r Dex_header_paper_implementation/only_base1_model/artifacts/export/mlp_header/* \
  vigidroid/app/src/main/assets/models/mlp_header/
# Re-run parity_vectors.json conversion if sample_vectors.npz changed (see A4 section)
```

**Optional — confirm PC export is healthy before Android work:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
../../thesis_venv/bin/python scripts/parity_check_onnx.py \
  --bundle artifacts/export/mlp_header
# Expect: passed: true
```

---

## Phase A1 — Feature extractor (APK → 104 numbers)

**Goal:** Java code that matches Python P2 for BM1.

### Code (implemented)

- [x] **A1.1** `DexHeaderFeatureExtractor.java`
  - Open APK as ZIP
  - Find all entries whose **basename** matches `classes.dex`, `classes2.dex`, … (sort: `classes.dex` first, then numeric suffix)
  - Per Dex file: validate magic `dex\n`, read bytes **8–111**, divide each byte by `255.0f` → 104 floats
  - **Sum** all per-Dex vectors element-wise → one 104-d vector (multidex mode `sum`)
  - If no Dex or bad magic → fail this APK (log error, skip)
- [x] **A1.2** Loads `normalization_header.json`, min–max transform
- [x] **A1.3** `app/src/test/.../DexHeaderFeatureExtractorTest.java`

**Reference Python files (read-only):**

- `src/features/dex_header.py`
- `src/features/multidex.py`
- `src/features/apk_extract.py`
- `src/features/normalization.py`

**No run command yet** — code only. Verify later via A4.

---

## Phase A2 — ONNX inference

**Goal:** Load BM1 model and return malware probability.

### Code (implemented)

- [x] **A2.1** `MlpHeaderOnnxRunner.java` + `ModelAssetHelper.java`
  - Copy `assets/models/mlp_header/model.onnx` → app cache
  - Create `OrtSession` (same ONNX Runtime dependency already in `app/build.gradle.kts`)
  - Input tensor name: `features`, shape `[1, 104]`, `float32`
  - Output name: `malware_probability` → single float in `[0, 1]`
- [x] **A2.2** Reads `export_manifest.json` for input name
- [x] **A2.3** BM1 init failure is non-fatal; CNN/XGB unchanged

**No phone run yet** — build must compile:

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew assembleDebug
```

---

## Phase A4 — Parity test (do BEFORE A3)

**Goal:** Phone ONNX output matches PC `parity_samples/` (tolerance `1e-4`).

### Code (implemented)

- [x] **A4.1** `parity_samples/` in assets (includes `parity_vectors.json`)
- [x] **A4.2** JSON loader in `MlpHeaderParityTest` (regenerate if needed):
  ```bash
  cd /mnt/Files/thesis_vigidroid
  thesis_venv/bin/python - <<'PY'
  import json, numpy as np
  from pathlib import Path
  p = Path("Dex_header_paper_implementation/only_base1_model/artifacts/export/mlp_header/parity_samples")
  d = np.load(p / "sample_vectors.npz")
  out = {
      "vectors": d["vectors"].tolist(),
      "expected_scores": d["expected_scores"].tolist(),
      "sample_ids": [str(x) for x in d["sample_ids"].tolist()],
  }
  (p / "parity_vectors.json").write_text(json.dumps(out))
  print("wrote", p / "parity_vectors.json")
  PY
  cp Dex_header_paper_implementation/only_base1_model/artifacts/export/mlp_header/parity_samples/parity_vectors.json \
     vigidroid/app/src/main/assets/models/mlp_header/parity_samples/
  ```
- [x] **A4.3** `app/src/androidTest/.../MlpHeaderParityTest.java`
  - For each of 8 samples: run `MlpHeaderOnnxRunner` → compare to expected
  - Fail if any `abs(phone - expected) >= 1e-4`

### Commands — run on phone (USB debugging on)

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid

# Install debug APK + run BM1 parity test only
./gradlew installDebug
./gradlew connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.msh.vigidroid.MlpHeaderParityTest
```

**Or in Android Studio:** open `vigidroid/` → right-click `MlpHeaderParityTest` → **Run**.

**Pass criteria:** all 8 tests green. If red, fix A1/A2 before A3.

---

## Phase A3 — Real scans + metrics

**Goal:** Each APK scan runs BM1 and appends one `stages[]` entry (domain, timings, score).

### Code (implemented)

- [x] **A3.1** `ScanService.initMlpHeaderPipeline()` — `model_id` `mlp_header`, domain `dex_header_d3`
- [x] **A3.2** Each APK in `Download/`: extract → ONNX → `stages[]` in `all_scan_metrics.json`
- [x] **A3.3** CNN/XGB ensemble **unchanged**; BM1 is an extra stage only

### Where to put APKs on the phone

Current app scans **`Download/`** on internal storage (all `.apk` files). No per-file picker.

1. Copy APKs to phone: `Internal storage/Download/*.apk`  
   ```bash
   adb push /path/to/your/sample.apk /sdcard/Download/
   ```
2. Grant storage permission in app (first launch).

### Commands — build, install, scan

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew installDebug
adb shell am start -n com.msh.vigidroid/.MainActivity
# Tap Start scan in the app (manual trigger)
```

### Pull metrics to PC

```bash
cd /mnt/Files/thesis_vigidroid
bash Shared_pipeline_Files/tools/pull_device_metrics.sh
# Output: Shared_pipeline_Files/results/device/all_scan_metrics.json
```

Check BM1 stage in JSON: `stages[]` entry with `domain` / model id for `mlp_header`.

---

## Phase 5 — Android Studio release build (optional)

For a shareable APK (not debug):

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew assembleRelease
# APK: app/build/outputs/apk/release/app-release-unsigned.apk
```

For thesis work, **debug install via Run ▶** is enough.

---

## Checklist before you call BM1 “done”

- [ ] Phase 0: bundle in `assets/models/mlp_header/`
- [ ] A1 + A2 code merged, `assembleDebug` succeeds
- [ ] A4: 8/8 parity tests pass on POCO F3
- [ ] A3: scan Download folder APKs; `all_scan_metrics.json` has BM1 stages with scores + timings
- [ ] Metrics pulled to PC
- [ ] ByteCNN + XGBoost still work unchanged

---

**New Java files:** `DexHeaderFeatureExtractor`, `MlpHeaderOnnxRunner`, `ModelAssetHelper`, `MlpHeaderParityTest`, `DexHeaderFeatureExtractorTest`.
