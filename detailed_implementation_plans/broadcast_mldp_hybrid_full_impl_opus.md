# Broadcast + MLDP Permission Hybrid — Full Implementation Plan (Opus)

**Cross-paper hybrid:**
- **#7** Ghasempour, Mohd Sani & Abari — *Permission Extraction Framework for Android Malware Detection*, IJACSA 11(11), 2020 → **MLDP permission pruning**
- **#12** Mohsen, Bisgin, Scott & Strait — *Detecting Android Malwares By Mining Statically Registered Broadcast Receivers*, IEEE CIC, 2017 → **broadcast receiver system actions**

**Verified rough plan:** `detailed_implementation_plans/broadcast_mldp_hybrid_opus.html`
**Pipeline alignment:** `sendable/Source_papers/Pipeline_full_concept.html` (P0–P8 offline, A1–A4 on-device)
**Thesis category:** Hybrid (with modifications) — manifest-only, early fusion
**Sibling reference plan:** `detailed_implementation_plans/simple_1_brd_rec_perm.md` (the single-paper broadcast+permission model — reuse its parser & Android scaffolding)

---

## 0. Up-front assignments

| Field | Value | Notes |
|-------|-------|-------|
| **`model_id`** | `broadcast_mldp_hybrid` | Used in `artifacts/export/`, Android `assets/models/`, metrics JSON |
| **`domain`** | `manifest_mldp_perm_receiver_actions` | Distinct from `manifest_perm_receiver_actions` (simple_1) and Dex-header domains |
| **On-device feasible** | **Yes** | Manifest-only parse; target `<100 ms` extract, `<20 KB` ONNX |
| **Fusion type** | **Early fusion** | `x = [x_S ‖ x_R]` → single classifier head |
| **Deployment head** | Tiny MLP `d→64→1` (logistic `d→1` fallback) | ONNX opset 14 |
| **Paper baseline** | RBF C-SVM (`γ=0.1, C=10`) + Decision Tree | sklearn, offline only, **not** exported |
| **Project folder** | `broadcast_mldp_hybrid/` | Self-contained training workspace |
| **Plan id** | `hybrid_1_brd_mldp` | First hybrid implementation in thesis queue |

### Architecture summary (target)

```
Raw APK
  → AndroidManifest.xml  (binary XML decode)
       ├─► MLDP permission block   x_S ∈ {0,1}^|S|   (only perms in frozen set S, |S|≈22)
       └─► Receiver system-action  x_R ∈ {0,1}^R     (static <receiver> actions ∩ system-action allow-list, R≈30–80)
  → x = concat(x_S, x_R) ∈ {0,1}^(|S|+R)   d ≈ 50–100
  → Tiny MLP (d→64→1) → P(malware) ∈ [0,1]
```

### Cross-check corrections folded into this plan (from the opus rough plan)

| # | Correction | Where enforced |
|---|------------|----------------|
| M1 | `|S| ≈ 16–25 (≈22)` not 20–40 | P2 vocab build, config caps |
| M2 | MLDP = **Variant 1** PRNR→SPR→PMAR (deterministic), published list as fallback | P2 `src/features/mldp/` |
| M3 | Receiver actions restricted to **Android system-action allow-list** | P2 `receivers.py`, A1 Java extractor, shipped `system_actions.json` |
| M4 | PRNR keeps both `R→+1` and `R→-1`, drops `R≈0` | P2 PRNR implementation |
| M5 | Fused dim `d ≈ 50–100` | P4 model, export manifest shape |
| M6 | Add RBF-SVM paper baseline (γ=0.1, C=10) | P5 `svm_baseline.py` |
| M7 | Class imbalance → `pos_weight` / balanced sampling | P5 training |
| M9 | Report honest temporal numbers (expect ~75–90%, not 97%) | P6 metrics + thesis text |

---

## 1. Dependencies and risks

### 1.1 External dependencies

- **APK corpus** on disk (`apk_root/`), year folders `2020–2023`, `benign/` + `malware/` — not in git.
- **Manifest decoder (Python):** `androguard` (paper #7 used Androguard) — lock version in P0. `axmlparserpy` or `aapt2 dump xmltree` acceptable if faster; must match the simple_1 decoder choice for shared parity tooling.
- **Manifest decoder (Android):** existing `AxmlReader` in VigiDroid; align tag/attribute traversal with Python.
- **Android system-action allow-list:** a static JSON file enumerating Android OS broadcast actions (built once, shipped to device). Source: Android SDK `Intent` action constants across target API levels (compile like #12's "system actions of Android OS releases in one file").
- **Association-rule mining:** `mlxtend` (Apriori) for PMAR, or a small hand-rolled Apriori on ~25 items (cheap).
- **Training:** PyTorch 2.x, scikit-learn (SVM/DT baseline), ONNX 1.x + onnxruntime, opset **14**.

### 1.2 Risk register

| Risk | Mitigation |
|------|------------|
| Train/serve skew (Java ≠ Python manifest parse) | Shared golden APK set; **P8 + A4** parity on `parity_samples/` |
| **System-action allow-list drift** (Python vs Java) | Single `system_actions.json` shipped in export bundle; both sides load the SAME file (M3) |
| Vocabulary / S leakage from test years | Build `S` and `A` **only** from `split=train` APKs in P2 |
| MLDP non-determinism | Use Variant 1 (PRNR→SPR→PMAR) with fixed thresholds + seed; persist `mldp_trace.json` |
| Obfuscated / encrypted manifests | Log to `failed_apks.log`; exclude from counts (both papers note this limitation) |
| All-zero feature rows | Keep in dataset; model must handle them (common for tiny apps) |
| Class imbalance | `pos_weight` in BCE; optionally Mohsen-style balanced subsampling for the SVM baseline |
| Overstating accuracy | Report temporal-split numbers; cite #12 independent-set drop in thesis |

---

## 2. Project layout

```
broadcast_mldp_hybrid/
├── config/
│   └── default.yaml
├── requirements.txt
├── scripts/
│   ├── verify_setup.py
│   ├── index_dataset.py            # P1
│   ├── build_system_actions.py     # P2 prep: compile Android system-action allow-list
│   ├── run_mldp.sh                 # P2: PRNR→SPR→PMAR → freeze S
│   ├── run_preprocess.sh           # P2 vectorize wrapper
│   ├── run_train.sh                # P5
│   ├── run_evaluate.sh             # P6
│   └── export_onnx.py              # P7
├── assets/
│   └── system_actions.json         # checked-in allow-list (M3) — source of truth
├── src/
│   ├── config.py
│   ├── constants.py                # label names, manifest tag/attr constants
│   ├── indexing/
│   │   └── build_manifest.py       # P1 CSV/JSON index
│   ├── features/
│   │   ├── manifest_decode.py      # APK → parsed manifest dict (perms + static receiver actions)
│   │   ├── permissions.py          # raw permission extraction
│   │   ├── receivers.py            # static <receiver> actions ∩ system_actions.json  (M3)
│   │   ├── mldp/
│   │   │   ├── prnr.py             # Permission Ranking w/ Negative Rate (M4)
│   │   │   ├── spr.py              # Support-based Permission Ranking
│   │   │   ├── pmar.py             # Apriori association-rule collapse
│   │   │   └── select.py           # orchestrate → freeze S, write mldp_trace.json
│   │   ├── vocab.py                # build/freeze S (perms) and A (system actions)
│   │   └── vectorize.py            # x_S, x_R, concat → x
│   ├── preprocessing/
│   │   └── preprocess_apks.py      # P2 batch job
│   ├── data/
│   │   ├── store.py
│   │   ├── dataset.py
│   │   └── dataloaders.py
│   ├── models/
│   │   ├── tiny_mlp.py             # deployment head (d→64→1)
│   │   └── logistic_head.py        # fallback (d→1)
│   └── training/
│       ├── svm_baseline.py         # paper-faithful RBF-SVM + DT (M6)
│       ├── train.py                # MLP/logistic + ablations
│       ├── evaluate.py             # P6
│       └── parity_onnx.py          # P8
└── artifacts/
    ├── manifests/                  # P1 index
    ├── processed/                  # P2 shards + vocab + mldp_trace.json
    ├── checkpoints/
    ├── metrics/
    └── export/broadcast_mldp_hybrid/
```

**Android (sibling repo `vigidroid/`):**

```
app/src/main/assets/models/broadcast_mldp_hybrid/
├── model.onnx
├── export_manifest.json
├── thresholds.json
├── features/
│   ├── mldp_permission_vocab.json     # frozen S (ordered)
│   ├── receiver_action_vocab.json     # frozen A (ordered, system actions)
│   ├── system_actions.json            # allow-list (same file used in Python)  (M3)
│   └── feature_layout.json            # {"order":["mldp_perms","receiver_actions"],"S":..,"R":..}
└── parity_samples/
```

---

## 3. Configuration contract (P0)

### 3.1 `config/default.yaml`

```yaml
model_id: broadcast_mldp_hybrid
domain: manifest_mldp_perm_receiver_actions

paths:
  apk_root: /path/to/apk_corpus    # EDIT per machine
  train_years: [2020, 2021]
  test_years: [2022, 2023]

splits:
  val_fraction_of_train: 0.10      # early stopping / threshold tuning only

features:
  manifest_backend: androguard     # androguard | axmlparserpy | aapt2  (lock in P0)
  normalize_permission_names: true # strip "android.permission." prefix consistently
  receiver_scope: static_manifest_only
  receiver_system_actions_only: true        # M3 — hard requirement
  system_actions_file: assets/system_actions.json

mldp:                              # M2 — Variant 1 (deterministic)
  method: prnr_spr_pmar            # prnr_spr_pmar | published_list | proposed_pca (not recommended)
  prnr_drop_abs_threshold: 0.05    # |R(Pj)| <= t  → dropped as non-discriminative  (M4)
  spr_keep_top: 25                 # support-based survivors
  pmar_min_support: 0.10           # Apriori
  pmar_min_confidence: 0.965       # Apriori (paper #7)
  target_size_hint: 22             # |S| sanity bound (M1); error if |S| > 30
  fallback_published_list: true    # if mining degenerate, use Table I (22 perms)

classifier:
  deployment: tiny_mlp             # tiny_mlp | logistic
  tiny_mlp_hidden: 64
  paper_baseline_svm: true         # M6
  svm_C: 10.0                      # #12 optimum
  svm_gamma: 0.1                   # #12 optimum
  svm_kernel: rbf

training:
  batch_size: 256
  epochs: 60
  learning_rate: 0.005
  weight_decay: 0.0001
  pos_weight: auto                 # N_neg / N_pos on train  (M7)
  early_stop_patience: 6
  seed: 42

export:
  onnx_opset: 14
  parity_num_samples: 10
  parity_max_delta: 1.0e-4
```

### 3.2 P0 deliverables / exit criteria

| Deliverable | Exit criterion |
|-------------|----------------|
| `requirements.txt` | `pip install -r requirements.txt` succeeds |
| `verify_setup.py` | Imports torch/sklearn/onnx/mlxtend/androguard; loads YAML; asserts `apk_root` exists; asserts `system_actions.json` parses and is non-empty |
| `ensure_artifact_dirs()` | Creates `artifacts/{manifests,processed,checkpoints,metrics,export}` |
| README stub | Paper links, train years, run order |

**Do not start P2 until P0 passes.**

---

## Phase P1 — Dataset indexing

### Goal
Machine-readable index of all APKs with `label`, `year`, `split`, integrity fields — no corpus copy.

### Tasks
1. Walk `apk_root/{year}/{benign|malware}/**/*.apk`.
2. Per file: SHA-256 (skip unreadable zips, log reason); record `apk_path`, `sha256`, `label` (0/1), `year`, `split`.
   - `split = train` if `year ∈ {2020,2021}`, `test` if `year ∈ {2022,2023}`.
3. Deduplicate by `sha256` (keep first, log dups).
4. Write `artifacts/manifests/apk_index.csv` (+ optional `.json`).

### Output schema
`apk_path, sha256, label, year, split[, apk_size_bytes, num_dex_files]`

### Exit criteria
- [ ] Row counts per year/label/split printed
- [ ] No 2022/2023 APK marked `train`
- [ ] Duplicate-hash report + `failed_index.log` produced

---

## Phase P2 — Feature extraction (manifest → MLDP set S, vocab A, tensors)

This is the **critical phase** — it owns all three corrections M1–M4. Three sub-stages: (2a) parse, (2b) freeze S and A from train only, (2c) vectorize all.

### 2a. Manifest parsing (`src/features/manifest_decode.py`)

**Input:** APK path → **Output:** `{permissions: List[str], receiver_actions: List[str]}`

Steps per APK:
1. Open APK as ZIP; read `AndroidManifest.xml`.
2. Decode binary XML (backend from config).
3. **Permissions:** every `<uses-permission android:name>` (document whether `uses-permission-sdk-23` is included; default: include, normalized).
4. **Receiver actions:** every `<receiver>` → child `<intent-filter>` → child `<action android:name>`. Collect raw names.
5. Normalize permission names (strip `android.permission.`); dedup within APK (set semantics).

**Explicit exclusions (paper alignment):** no `classes.dex`, no dynamic (context-registered) receivers, no activities/services unless debugging.

### 2b-i. System-action allow-list (`scripts/build_system_actions.py`) — **M3**

- Compile the set of Android OS broadcast action constants (the `android.intent.action.*`, `android.net.conn.CONNECTIVITY_CHANGE`, `android.provider.Telephony.SMS_RECEIVED`, `android.intent.action.BOOT_COMPLETED`, `USER_PRESENT`, `PHONE_STATE`, `NEW_OUTGOING_CALL`, … families) across target API levels.
- Persist to `assets/system_actions.json` as `{ "actions": [ "<fully.qualified.ACTION>", ... ] }`.
- This file is **the single source of truth** shipped unchanged to Android (parity guarantee).

### 2b-ii. MLDP selection → freeze S (`src/features/mldp/`) — **M2, M4**

Run **once on `split=train` APKs only**:
1. Build `M` (malware) and `B` (benign) binary permission matrices over the **full** permission vocabulary observed in train.
2. **Skew correction:** scale supports so `|M|`/`|B|` imbalance does not bias ranking (#7 eq. 1/3).
3. **PRNR** (`prnr.py`): `R(Pj) = (S_M(Pj) − S_B(Pj)) / (S_M(Pj) + S_B(Pj))`, range `[-1,1]`.
   Drop permissions with `|R(Pj)| <= prnr_drop_abs_threshold` (≈0; non-discriminative). **Keep both extremes** (M4). ~135 → ~95.
4. **SPR** (`spr.py`): rank survivors by overall support; keep top `spr_keep_top` (≈25).
5. **PMAR** (`pmar.py`): Apriori (`min_support=0.10`, `min_confidence=0.965`); collapse permissions that imply each other (keep one representative). ~25 → **~22**.
6. Freeze ordered `S`; write:
   - `artifacts/processed/mldp_permission_vocab.json` → `{"tokens":[...], "size":|S|}`
   - `artifacts/processed/mldp_trace.json` → per-stage counts, `R(Pj)` table, dropped/kept lists, rules found.
7. **Guards:** if `|S| > 30` → raise (violates M1). If mining degenerate (e.g. `|S| < 8`) and `fallback_published_list: true` → use paper Table I (22 perms) and record fallback in trace.

### 2b-iii. Receiver vocabulary A (`src/features/vocab.py`) — **M3**
1. Over `split=train`, collect receiver actions ∩ `system_actions.json`.
2. Count document frequency; keep all that appear (optionally `min_doc_freq`).
3. Sort lexicographically → ordered `A`; write `receiver_action_vocab.json` `{"tokens":[...],"size":R}`.
4. Write `feature_layout.json` `{"order":["mldp_perms","receiver_actions"],"S":|S|,"R":R,"total":|S|+R}`.

### 2c. Vectorization (`src/features/vectorize.py`)

For APK *i*:

```
x_S[i,j] = 1 if perm_j ∈ S and declared, else 0
x_R[i,k] = 1 if action_k ∈ A and present in static receivers, else 0
x[i]     = concat(x_S[i], x_R[i]) ∈ {0,1}^(|S|+R)
```

Store as **float32** (0.0/1.0) for ONNX runtime compatibility.

### 2d. Batch preprocess (`src/preprocessing/preprocess_apks.py`)
1. Load `S`, `A`, `system_actions.json` (must exist).
2. For each APK (train + test): parse → vectorize → append to shard buffers; failures → `artifacts/failed_apks.log`.
3. Write `features_train.pt`, `features_test.pt`, optional `features_val.pt` (each: `x`, `y`, `paths`, `sha256`).
4. Emit `preprocessing_meta.json`: `preprocessing_version` (git hash/date), `|S|`, `R`, `d`, counts, failures.

### Exit criteria
- [ ] `|S| ≈ 16–25` (≤30) and `R ≈ 30–80` logged; `mldp_trace.json` present (M1, M2)
- [ ] Receiver vector built **only** from system-action allow-list (M3) — spot-check 5 APKs vs `aapt2`/apktool dump
- [ ] PRNR table shows both malware- and benign-indicative perms retained (M4)
- [ ] No test APK influenced `S` or `A`
- [ ] train/test counts = P1 minus failures

---

## Phase P3 — DataLoaders

### Tasks
1. `HybridManifestDataset.__getitem__` → `(x: float32[d], y: long)`.
2. `build_dataloaders()`:
   - `train_loader` (train split), `val_loader` (10% holdout from train), `test_loader` (2022+2023).
3. `batch_size` from config; `num_workers`/`pin_memory` per machine.

### Exit criteria
- [ ] One batch smoke test: shapes `[B, d]`, labels in `{0,1}`
- [ ] Per-split class balance printed (feeds `pos_weight`)

---

## Phase P4 — Model definition

### 4.1 Deployment: Tiny MLP (`src/models/tiny_mlp.py`)
```
Linear(d → 64) → ReLU → Dropout(0.2) → Linear(64 → 1)
# forward returns logits; sigmoid applied in export graph OR documented for Java
```
`BCEWithLogitsLoss(pos_weight=...)`.

### 4.2 Fallback: Logistic (`src/models/logistic_head.py`)
`nn.Linear(d, 1)` — use if MLP gives no val gain over logistic; smallest ONNX.

### 4.3 Paper baseline: RBF-SVM + DT (`src/training/svm_baseline.py`) — **M6**
- `SVC(kernel="rbf", C=10, gamma=0.1, class_weight="balanced")` and `DecisionTreeClassifier`.
- Fit on train (numpy), evaluate on val + test. Save `svm_rbf.joblib`, `svm_metrics.json`, `dt_metrics.json`.
- **Not** exported to ONNX (out of scope unless `skl2onnx` requested).

### Exit criteria
- [ ] MLP forward on dummy `[1, d]` works; param count `< few×d×64` (well under 20 KB ONNX)
- [ ] SVM baseline fits and scores on a small subset

---

## Phase P5 — Training

### Tasks
1. **MLP/logistic (PyTorch):** AdamW (config lr/wd); `BCEWithLogitsLoss(pos_weight=auto)` (M7); per-epoch train/val loss, val F1, val AUC; early stop on val F1; save `artifacts/checkpoints/best.pt` (`model_state`, `S`, `A`, `d`, `config_hash`).
2. **SVM/DT baseline** (parallel script): metrics for thesis "paper-faithful" table (M6).
3. **Ablations (required for thesis narrative, mirrors #12 three scenarios):**
   - **MLDP perms only** (`x_S`, first `|S|` dims)
   - **Receiver actions only** (`x_R`, last `R` dims)
   - **Full fusion** (`x`)

### Exit criteria
- [ ] Full-fusion val F1 ≥ max(perm-only, receiver-only) — qualitative match to #12 "combined > either"
- [ ] `best.pt` reloadable; training log saved
- [ ] SVM/DT baseline JSON exists

---

## Phase P6 — Evaluation (test split 2022 + 2023)

### Metrics
Accuracy, F1 (malware = positive), ROC-AUC, confusion matrix (TN, FP, FN, TP). Threshold: default 0.5; optional tuned on **val only** → `thresholds.json`.

### Output `artifacts/metrics/test_results.json`
```json
{
  "model_id": "broadcast_mldp_hybrid",
  "split": "test",
  "train_years": [2020, 2021],
  "test_years": [2022, 2023],
  "n_samples": 0,
  "feature_dims": { "S": 0, "R": 0, "total": 0 },
  "metrics": { "accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0 },
  "confusion_matrix": [[0, 0], [0, 0]],
  "threshold": 0.5,
  "ablations": {
    "mldp_perms_only": { "f1": 0.0 },
    "receiver_actions_only": { "f1": 0.0 },
    "full_fusion": { "f1": 0.0 }
  },
  "paper_baselines": { "svm_rbf": { "f1": 0.0 }, "decision_tree": { "f1": 0.0 } }
}
```

### Exit criteria
- [ ] Eval reads only `features_test.pt` (never APKs)
- [ ] Ablation table present; confusion matrix optional plot
- [ ] **M9:** thesis text frames expected ~75–90% temporal F1, not #12's 97% duplicate-config headline

---

## Phase P7 — ONNX export bundle

### Directory `artifacts/export/broadcast_mldp_hybrid/`

### Tasks
1. Load `best.pt`; `model.eval()`; trace with example `[1, d]`.
2. Export `model.onnx`, opset 14, input `features` `[1, d]` float32, output `malware_prob` (sigmoid in graph — document).
3. Copy `mldp_permission_vocab.json`, `receiver_action_vocab.json`, **`system_actions.json`** (M3), `feature_layout.json` → `features/`.
4. `thresholds.json`: `{ "default": 0.5, "tuned_val": <float> }`.
5. `export_manifest.json`:
```json
{
  "model_id": "broadcast_mldp_hybrid",
  "domain": "manifest_mldp_perm_receiver_actions",
  "opset": 14,
  "inputs": [{ "name": "features", "shape": [1, "S_PLUS_R"], "dtype": "float32" }],
  "outputs": [{ "name": "malware_prob", "dtype": "float32" }],
  "preprocessing_version": "<date-or-git>",
  "multidex_mode": "n/a",
  "feature_extraction": {
    "apk_part": "AndroidManifest.xml",
    "fusion": "early_concat",
    "blocks": ["mldp_perms", "receiver_system_actions"],
    "mldp_size_S": 0,
    "receiver_size_R": 0,
    "receiver_system_actions_only": true
  }
}
```
6. `parity_samples/` (~10 APKs): each `x.npy` (or JSON) + `expected_prob.json` from PyTorch.

### Exit criteria
- [ ] ONNX `< 20 KB`
- [ ] Bundle copies cleanly to `vigidroid/app/src/main/assets/models/broadcast_mldp_hybrid/`
- [ ] `system_actions.json` present in bundle (parity prerequisite)

---

## Phase P8 — Parity (PyTorch vs ONNX)

### Script `src/training/parity_onnx.py`
1. Load checkpoint + ONNX session.
2. For each `parity_samples/`: run both → compare `malware_prob`.
3. Write `artifacts/metrics/parity_report.json` with per-sample delta + `max_delta`.

### Exit criteria
- [ ] `max_delta ≤ 1e-4` on all samples
- [ ] On failure: fix dtype (float32), row-major layout, sigmoid placement before A1

---

## Android phases A1–A4

### A1 — Feature extractor (`BroadcastMldpHybridExtractor`)
Semantics **must** match Python P2:
1. Open APK ZIP → `AndroidManifest.xml`; `AxmlReader` collects permission names + static receiver actions.
2. Load `mldp_permission_vocab.json`, `receiver_action_vocab.json`, **`system_actions.json`** from assets.
3. Build `float[d]` = `[ x_S (|S|) | x_R (R) ]`:
   - perm bit set iff normalized name ∈ S;
   - action bit set iff action ∈ (system_actions ∩ A) and present in static receivers (**M3**).
4. Timings: `parse_ms`, `vectorize_ms`.
- **Deliverable:** JVM/instrumented unit test on 3 APKs comparing Java vector to Python dump.

### A2 — ONNX inference
1. `ModelRegistry` entry `broadcast_mldp_hybrid`.
2. Load `model.onnx` + manifest; ORT session; input `[1, d]` float32; read `malware_prob`.
3. Apply `thresholds.json`.

### A3 — Scan orchestration
Append one `stages[]` entry per scan:
```json
{
  "domain": "manifest_mldp_perm_receiver_actions",
  "model_id": "broadcast_mldp_hybrid",
  "parse_ms": 0, "vectorize_ms": 0, "inference_ms": 0,
  "score": 0.0, "mem_delta_bytes": 0
}
```
**Suggested cascade position:** early, cheap manifest gate (complementary to Dex-header / byte models).

### A4 — Instrumented parity test
- Load each `parity_samples/` input → compare device score to `expected_prob` within `1e-4`.
- CI gate before release build.

### A1–A4 exit criteria
- [ ] End-to-end parity-APK scan passes on device/emulator
- [ ] Java `x_R` uses the **same** `system_actions.json` as Python (M3)
- [ ] p50 `parse+vectorize+inference` ms logged

---

## 4. Execution order and gates

```
P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → A1 → A2 → A3 → A4
            ↑ S and A frozen here (train only) — do NOT start A1 before P2 done on full train set
```

| Gate | Rule |
|------|------|
| G1 | P2 may not use any `split=test` APK to build `S`/`A` |
| G2 | P7 only after P5 produces `best.pt` |
| G3 | A1 only after `features/*.json` + `system_actions.json` exist in bundle |
| G4 | P8 green before copying bundle to VigiDroid |
| G5 | A4 green before reporting on-device numbers |

---

## 5. Thesis experiment hooks

| Thesis task | Contribution |
|-------------|--------------|
| Task 1 — Resource optimization | manifest parse + vectorize + infer ms; ONNX `< 20 KB` |
| Task 2 — Multistep | fast manifest gate; `t_low`/`t_high` on val holdout |
| Task 5 — Tradeoffs | ablation MLDP-only vs receiver-only vs fused; F1 vs latency plot |
| Task 6 — Feasibility | manifest-only, no Dex → high deployability |
| Paper fidelity table | RBF-SVM (γ=0.1,C=10) + DT vs deployed MLP |
| Ensemble (later) | export calibrated score; offline-learned weight |

### Comparison targets
| Config | Expected role |
|--------|---------------|
| MLDP perms only | strong solo (#7: ~94–97% F on their corpus; lower on temporal split) |
| Receiver actions only | weaker alone (#12: ~71–79%); lifts sensitivity |
| **MLDP + receivers (fusion)** | best manifest-only-per-parameter (this model) |
| vs #12 full perm+receiver | ~3× fewer dims, similar/better generalization |

---

## 6. Per-phase checklists

### P2 feature parity checklist (Python internal)
- [ ] Permissions from `<uses-permission>` only; normalized consistently
- [ ] Receiver actions from **static `<receiver>` ∩ system-action allow-list** (M3)
- [ ] Set semantics within APK (dup tags → one bit)
- [ ] `S` via PRNR→SPR→PMAR on train only; `mldp_trace.json` written (M2)
- [ ] `|S| ≤ 30` guard passed (M1)

### P8 / A4 parity checklist (train vs device)
- [ ] Same manifest bytes read
- [ ] Same `S`, `A`, `system_actions.json` (same order → same index)
- [ ] float32 0.0/1.0 features
- [ ] Same sigmoid/logits handling
- [ ] Score delta `≤ 1e-4` on all parity samples

---

## 7. Estimated effort

| Phase | Effort (solo, familiar stack) |
|-------|-------------------------------|
| P0–P1 | 0.5–1 day |
| P2 (parser + MLDP mining + system-action list) | 3–5 days |
| P3–P4 | 0.5 day |
| P5–P6 (+ SVM baseline + ablations) | 1.5–2.5 days |
| P7–P8 | 0.5–1 day |
| A1–A4 (Java parity, esp. system-action list) | 2–3 days |

**Critical path:** P2 ↔ A1 parser + `system_actions.json` consistency.

---

## 8. Open decisions (defaults proposed above)

1. MLDP method: **Variant 1 (PRNR→SPR→PMAR)** vs published-list vs Variant-2 PCA? (default: Variant 1)
2. `include uses-permission-sdk-23` in permission space? (default: yes, normalized)
3. Python manifest backend: `androguard` (paper-faithful) vs faster alternative? (default: androguard)
4. Deployment head: tiny MLP vs logistic if no val gain? (default: MLP, fall back to logistic)
5. SVM/DT baseline required for thesis table? (default: yes — it is the only paper-faithful classifier)
6. System-action allow-list source/API-level coverage? (default: union across target minSdk..targetSdk)
7. `apk_root` path on the training machine?
8. Cascade slot vs other manifest models (e.g. simple_1 broadcast+perm)?

---

## 9. References
- Verified rough plan: `detailed_implementation_plans/broadcast_mldp_hybrid_opus.html`
- Tutorial: `detailed_implementation_plans/broadcast_mldp_hybrid_tutorial.html`
- Pipeline guide: `sendable/Source_papers/Pipeline_full_concept.html`
- Sibling plan: `detailed_implementation_plans/simple_1_brd_rec_perm.md`
- Paper #7: `sendable/Source_papers/7_Permission Extraction Framework for Android Malware Detection.pdf`
- Paper #12: `sendable/Source_papers/12_Detecting Android Malwares By Mining Statically Registered Broadcast Receivers (Full paper).pdf`
- Model catalog / ranks: `todo_model_ranks.html`

---

*Document version: 2026-06-07 · Plan id: `hybrid_1_brd_mldp` · cross-checked against source PDFs #7 and #12.*
