# Pattern A (`pattern_a_combined`) — Android steps A1 → A4

**Model:** Dex header + manifest BoW → single ASCNN (`concat(H, I)`)  
**Model ID:** `pattern_a_combined`  
**Status:** **Not implemented** in `vigidroid/`. Complete BM1 first (shared Dex header code).

**Rule:** Do **not** remove or change existing ByteCNN / XGBoost wiring. Add Pattern A as a **new** `stages[]` entry.

**Order:** A1 → A2 → **A4** → **A3**

---

## Paths

| What | Path |
|------|------|
| PC export bundle | `Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/export/pattern_a_combined/` |
| Android assets target | `vigidroid/app/src/main/assets/models/pattern_a_combined/` |
| Python reference | `.../full_combined_pipeline_approach/src/features/` |

---

## Phase 0 — Copy export bundle

```bash
cd /mnt/Files/thesis_vigidroid

mkdir -p vigidroid/app/src/main/assets/models/pattern_a_combined

cp -r Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/export/pattern_a_combined/* \
  vigidroid/app/src/main/assets/models/pattern_a_combined/

ls vigidroid/app/src/main/assets/models/pattern_a_combined/
# model.onnx  export_manifest.json  thresholds.json  features/  parity_samples/
```

**Optional PC parity check:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
../../../thesis_venv/bin/python scripts/parity_check_onnx.py \
  --bundle artifacts/export/pattern_a_combined
```

---

## Phase A1 — Feature extractor (APK → header + BoW)

**Goal:** Two float vectors matching Python P2.

| Output | Size | Source |
|--------|------|--------|
| `header` | 104 | Same as BM1 (multidex sum + min–max from `normalization_header.json`) |
| `bow` | 4381 | Manifest permissions + intents → multihot using `features/vocab.json` |

### Tasks to implement in `vigidroid/`

- [ ] **A1.1** Reuse BM1 `DexHeaderFeatureExtractor` for **header** (do not duplicate)
- [ ] **A1.2** New `ManifestBowExtractor.java`
  - Parse `AndroidManifest.xml` from APK ZIP (reuse / extend `AxmlReader.java` if possible)
  - Token rules must match Python `manifest_bow.py` + `build_lexicon.py`:
    - Permissions → normalized tokens
    - Intent actions → normalized tokens
    - Multihot: index `i` = 1.0 if token in vocab else UNK bucket
  - Load frozen `assets/models/pattern_a_combined/features/vocab.json` (4380 + UNK → length 4381)
- [ ] **A1.3** JVM unit tests: known manifest → expected sparse indices (compare one APK against Python offline if needed)

**Reference Python:**

- `src/features/manifest_bow.py`
- `src/features/dex_header.py`, `multidex.py`, `apk_extract.py`

---

## Phase A2 — ONNX inference

**Goal:** Two inputs → one malware probability.

### Tasks to implement in `vigidroid/`

- [ ] **A2.1** New `PatternAOnnxRunner.java`
  - Load `assets/models/pattern_a_combined/model.onnx`
  - Inputs (from `export_manifest.json`):
    - `header`: `[1, 104]` float32
    - `bow`: `[1, 4381]` float32
  - Output: `malware_probability` float32
- [ ] **A2.2** Missing bundle → skip stage; do not break legacy models

**Compile check:**

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew assembleDebug
```

---

## Phase A4 — Parity test (before A3)

### Tasks to implement in `vigidroid/`

- [ ] **A4.1** Convert parity NPZ to JSON (header + bow per sample):
  ```bash
  cd /mnt/Files/thesis_vigidroid
  thesis_venv/bin/python - <<'PY'
  import json, numpy as np
  from pathlib import Path
  p = Path("Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/export/pattern_a_combined/parity_samples")
  d = np.load(p / "sample_vectors.npz")
  # Keys may include header_vectors, bow_vectors — inspect: print(d.files)
  out = {k: d[k].tolist() if hasattr(d[k], 'tolist') else d[k] for k in d.files}
  idx = json.loads((p / "index.json").read_text())
  out["index"] = idx
  (p / "parity_vectors.json").write_text(json.dumps(out))
  print("keys:", list(d.files))
  PY
  cp Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/export/pattern_a_combined/parity_samples/parity_vectors.json \
     vigidroid/app/src/main/assets/models/pattern_a_combined/parity_samples/
  ```
- [ ] **A4.2** Instrumented test `PatternAParityTest.java` — 8 samples, tolerance `1e-4`

### Run on phone

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew installDebug
./gradlew connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.msh.vigidroid.PatternAParityTest
```

---

## Phase A3 — Scans + metrics

### Tasks to implement in `vigidroid/`

- [ ] **A3.1** Register `model_id`: `pattern_a_combined`, `domain`: `dex_header_manifest`
- [ ] **A3.2** Per APK in scan loop: extract header + bow → ONNX → append `stages[]` with timings + score
- [ ] **A3.3** Keep ByteCNN / XGBoost stages unchanged

### Put APKs on phone

```bash
adb push /path/to/sample.apk /sdcard/Download/
```

App scans **all** `.apk` in `Download/` when you tap Start scan.

### Build, run, pull metrics

```bash
cd /mnt/Files/thesis_vigidroid/vigidroid
./gradlew installDebug
adb shell am start -n com.msh.vigidroid/.MainActivity

cd /mnt/Files/thesis_vigidroid
bash Shared_pipeline_Files/tools/pull_device_metrics.sh
```

---

## Done checklist

- [ ] Bundle in `assets/models/pattern_a_combined/`
- [ ] A1 header matches BM1; BoW matches Python vocab
- [ ] A4 passes on device (8/8)
- [ ] A3 logs Pattern A stages in `all_scan_metrics.json`
- [ ] Legacy CNN/XGB still work

---

**Prerequisite:** BM1 Android steps complete (shared Dex header extractor).  
**Confirm this plan before implementation.**
