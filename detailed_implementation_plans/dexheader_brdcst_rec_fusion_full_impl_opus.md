# Dex Header + Broadcast Receiver Fusion — Full Implementation Plan (Opus)

**Cross-paper hybrid:**
- **MSFDroid** — *Dex header structural bytes + MLP(H)* → **PDF absent from workspace**; the Dex-header block is verified against the **deployed implementation** in `Dex_header_paper_implementation/only_base1_model/` and the shipped bundle `vigidroid/app/src/main/assets/models/mlp_header/`.
- **#12** Mohsen, Bisgin, Scott & Strait — *Detecting Android Malwares By Mining Statically Registered Broadcast Receivers*, IEEE CIC, 2017 → **broadcast receiver system actions** (PDF present, fully verified via `sendable/Source_papers/broadcast_receiver_paper_tutorial.html`).

**Verified rough plan:** `detailed_implementation_plans/dexheader_brdcst_rec_fusion_opus.html`
**Pipeline alignment:** `sendable/Source_papers/Pipeline_full_concept.html` (P0–P8 offline, A1–A4 on-device)
**Thesis category:** Hybrid (with modifications) — Dex-header + manifest receivers, embedding (intermediate) fusion
**Reuse sources:**
- `Dex_header_paper_implementation/only_base1_model/` — Dex header extractor (`src/features/dex_header.py`, `multidex.py`), corpus min–max (`src/features/normalization.py`), MLP(H) (`src/models/mlp_header.py`), ONNX export (`scripts/export_onnx.py`)
- `detailed_implementation_plans/simple_1_brd_rec_perm.md` — broadcast receiver parser & Android scaffolding
- `detailed_implementation_plans/broadcast_mldp_hybrid_full_impl_opus.md` and `mldp_dexheader_cascade_full_impl_opus.md` — sibling hybrid plans (same conventions, same dex/receiver reuse)
- `vigidroid/app/src/main/java/com/msh/vigidroid/DexHeaderFeatureExtractor.java` + `MlpHeaderOnnxRunner.java` — Android dex-header parity baseline

---

## 0. Up-front assignments

| Field | Value | Notes |
|-------|-------|-------|
| **`model_id`** | `dexheader_broadcast_fusion` | Used in `artifacts/export/`, Android `assets/models/`, metrics JSON |
| **`domain`** | `dex_header_receiver_actions` | Distinct from `dex_header_d3` (BM1), `manifest_perm_receiver_actions` (simple_1), `manifest_mldp_perm_receiver_actions` (broadcast hybrid), `manifest_mldp_perm_dex_header` (cascade) |
| **On-device feasible** | **Yes** | Manifest receiver parse + 112-byte/dex header reads; target `< 150 ms` extract, `< 60 KB` ONNX |
| **Fusion type** | **Embedding (intermediate) fusion** | `z = [z_H ‖ z_R]` → joint FC head → single score (not score-vote late fusion) |
| **Header branch** | Deployed **MLP(H) trunk** `104→128→128` (BN+ReLU); take 128-d penultimate as `z_H` | warm-start from shipped weights optional |
| **Receiver branch** | `Linear(R→d_R)+ReLU`, `d_R≈32–64` | sparse binary |
| **Fusion head** | `(128+d_R)→64→1` + sigmoid (bare logistic fallback) | ONNX opset 14 |
| **Paper baseline** | RBF C-SVM (`γ=0.1, C=10`) on `[H ‖ R]` early-concat | sklearn, offline only, **not** exported |
| **Project folder** | `dexheader_broadcast_fusion/` | Self-contained training workspace |
| **Plan id** | `hybrid_3_dexheader_broadcast` | Third hybrid in thesis queue |

### Architecture summary (target)

```
Raw APK
  ├─► classes*.dex ──► /255 per dex ──► sum-pool ──► corpus min–max ──► H ∈ [0,1]^104
  │                                                          └─► MLP(H) trunk 104→128→128 ──► z_H ∈ R^128
  └─► AndroidManifest.xml ──► static <receiver> actions ∩ system-action allow-list ──► R ∈ {0,1}^R  (R≈30–80)
                                                          └─► Linear(R→d_R)+ReLU ──► z_R ∈ R^{d_R}

z = [z_H ‖ z_R] ∈ R^{128+d_R}  ──► FC (128+d_R)→64→1 ──► σ ──► P(malware) ∈ [0,1]
```

### Cross-check corrections folded into this plan (from the opus rough plan)

| # | Correction | Where enforced |
|---|------------|----------------|
| M1 | Dex normalization = **3 ordered stages**: `/255` per dex → sum-pool → corpus min–max (LAST) | P2 `dex_header.py` + `multidex.py` + `normalization.py` (reuse) |
| M2 | Pooled `H_raw ∈ [0, D_dex]`, only `[0,1]` after min–max | P2 vectorize docstring; metrics sanity check |
| M3 | min/max **frozen on train**, shipped as `normalization_header.json`; never recomputed on device | P2, P7, A1 |
| M4 | Receiver vocabulary `A` = **system actions only** (Android OS allow-list); ship `system_actions.json` to both sides | P2 `receivers.py`, A1 Java extractor, `assets/system_actions.json` |
| M5 | Header branch = deployed **MLP(H) trunk** `104→128→128`; `z_H` = 128-d penultimate; drop `→1→σ`; warm-start optional | P4 `header_tower.py`, P5 |
| M6 | Fusion = **embedding/intermediate fusion** (concat embeddings → joint FC), not score-vote late fusion | P4 model, export manifest `fusion` field |
| M7 | Fusion head = `(128+d_R)→64→1` (bare logistic fallback); `d_R≈32–64` so branches balance | P4 model, config |
| M8 | Class imbalance → `pos_weight` / balanced training | P5 |
| M9 | Honest accuracy framing: ground header baseline on deployed measured number; expect ~75–90% temporal, not paper headlines | P6 metrics + thesis text |
| M10 | Dex cost framed honestly (unzip + 112 B/dex header read, no bytecode scan) | thesis text, P2 docstring |
| M11 | Multidex `sum` default (deployed); `primary_only` ablation (MSFDroid-faithful) | P2 config, export manifest `multidex_mode` |

---

## 1. Dependencies and risks

### 1.1 External dependencies

- **APK corpus** on disk (`apk_root/`), year folders `2020–2023`, `benign/` + `malware/` — not in git.
- **Manifest decoder (Python):** `androguard` / `axmlparserpy` / `aapt2 dump xmltree` — pick one and lock version in P0; must match the `simple_1` decoder choice for shared parity tooling.
- **Dex header reader (Python):** reuse `only_base1_model/src/features/dex_header.py` + `multidex.py` + `normalization.py` (ZIP-enumerate `classes*.dex`, magic check, bytes 8–111, sum-pool, corpus min–max).
- **Android system-action allow-list:** a static JSON enumerating Android OS broadcast actions, built once and shipped (M4). Source: Android SDK `Intent`/`Telephony`/connectivity action constants across target API levels.
- **Deployed MLP(H):** `Dex_header_paper_implementation/only_base1_model/src/models/mlp_header.py` + shipped `vigidroid/.../models/mlp_header/` (for warm-start + header-only reference number, M5/M9).
- **Manifest decoder (Android):** existing `AxmlReader` in VigiDroid; align tag/attribute traversal with Python.
- **Dex header reader (Android):** existing `DexHeaderFeatureExtractor.java` in VigiDroid (reuse for the header branch).
- **Training:** PyTorch 2.x, scikit-learn (SVM baseline), ONNX 1.x + onnxruntime, opset **14**.

### 1.2 Risk register

| Risk | Mitigation |
|------|------------|
| Train/serve skew (Java ≠ Python features) | Shared golden APK set; **P8 + A4** parity on `parity_samples/` |
| **Dex normalization mismatch** (recompute on device vs frozen train stats) | Ship single `normalization_header.json`; both sides load it (M1/M3) |
| **System-action allow-list drift** (Python vs Java) | Single `system_actions.json` shipped in bundle; both sides load the SAME file (M4) |
| Vocabulary / norm leakage from test years | Build `A` and dex min/max **only** from `split=train` APKs in P2 |
| Two-input ONNX graph mismatch | Export one graph with two named inputs (`dex_header[1,104]`, `receiver[1,R]`); document in manifest |
| Obfuscated manifest / packed dex / bad magic | Log to `failed_apks.log`; exclude from counts |
| All-zero feature rows (tiny apps, no receivers) | Keep in dataset; model must handle them |
| Branch imbalance (104-byte side swamps sparse receivers) | Tune `d_R` on val; optional per-branch dropout/BN (M7) |
| Class imbalance | `pos_weight` in BCE; optional Mohsen-style balanced subsampling for SVM baseline (M8) |
| Multidex aggregation mismatch | Record `multidex_mode`; default `sum` both sides (M11) |
| Overstating accuracy | Report temporal-split numbers; ground header baseline on deployed measured value (M9) |

---

## 2. Project layout

```
dexheader_broadcast_fusion/
├── config/
│   └── default.yaml
├── requirements.txt
├── scripts/
│   ├── verify_setup.py
│   ├── index_dataset.py            # P1
│   ├── build_system_actions.py     # P2 prep: compile Android system-action allow-list (M4)
│   ├── run_preprocess.sh           # P2 vectorize wrapper (dex + manifest)
│   ├── run_train.sh                # P5
│   ├── run_evaluate.sh             # P6
│   └── export_onnx.py              # P7
├── assets/
│   └── system_actions.json         # checked-in allow-list (M4) — source of truth
├── src/
│   ├── config.py
│   ├── constants.py                # DEX_MAGIC, DEX_HEADER_SIZE=0x70, FEATURE_DIM=104, label names, manifest tags
│   ├── indexing/
│   │   └── build_manifest.py       # P1 CSV/JSON index
│   ├── features/
│   │   ├── dex_header.py           # bytes 8–111 / 255   (reuse from only_base1_model)
│   │   ├── multidex.py             # sum-pool across classes*.dex (reuse)
│   │   ├── normalization.py        # fit/transform corpus min–max (reuse)
│   │   ├── manifest_decode.py      # APK → parsed manifest dict (static receiver actions)
│   │   ├── receivers.py            # static <receiver> actions ∩ system_actions.json  (M4)
│   │   ├── vocab.py                # build/freeze receiver-action vocab A (train only)
│   │   └── vectorize.py            # H (104), R (binary) — kept separate (two-input model)
│   ├── preprocessing/
│   │   └── preprocess_apks.py      # P2 batch job (dex + manifest)
│   ├── data/
│   │   ├── store.py
│   │   ├── dataset.py
│   │   └── dataloaders.py
│   ├── models/
│   │   ├── header_tower.py         # MLP(H) trunk 104→128→128 → z_H (M5); warm-start loader
│   │   ├── receiver_tower.py       # Linear(R→d_R)+ReLU → z_R
│   │   ├── fusion_net.py           # two towers + joint FC head (128+d_R)→64→1  (M6/M7)
│   │   └── logistic_head.py        # bare-linear fusion fallback
│   └── training/
│       ├── svm_baseline.py         # paper-faithful RBF-SVM on [H‖R] early-concat (offline)
│       ├── train.py                # fusion net + ablations
│       ├── evaluate.py             # P6
│       └── parity_onnx.py          # P8
└── artifacts/
    ├── manifests/                  # P1 index
    ├── processed/                  # P2 shards + receiver vocab + dex min/max
    ├── checkpoints/
    ├── metrics/
    └── export/dexheader_broadcast_fusion/
```

**Android (sibling repo `vigidroid/`):**

```
app/src/main/assets/models/dexheader_broadcast_fusion/
├── model.onnx                       # two-input fusion graph
├── export_manifest.json
├── thresholds.json
├── features/
│   ├── receiver_action_vocab.json   # frozen A (ordered, system actions)
│   ├── system_actions.json          # allow-list (same file used in Python)  (M4)
│   ├── normalization_header.json    # dex corpus min/max (frozen train)       (M1/M3)
│   └── feature_layout.json          # {"dex_header":104,"receiver":R,"d_R":..,"fused":..}
└── parity_samples/
```

---

## 3. Configuration contract (P0)

### 3.1 `config/default.yaml`

```yaml
model_id: dexheader_broadcast_fusion
domain: dex_header_receiver_actions

paths:
  apk_root: /path/to/apk_corpus    # EDIT per machine
  train_years: [2020, 2021]
  test_years: [2022, 2023]
  deployed_mlp_header_bundle: ../vigidroid/app/src/main/assets/models/mlp_header  # warm-start + header ref (M5/M9)

splits:
  val_fraction_of_train: 0.10      # early stopping / threshold tuning only

features:
  manifest_backend: androguard     # androguard | axmlparserpy | aapt2  (lock in P0)
  receiver_scope: static_manifest_only
  receiver_system_actions_only: true        # M4 — hard requirement
  system_actions_file: assets/system_actions.json
  receiver_action_min_doc_freq: 1           # drop ultra-rare actions on train only
  dex:
    header_size: 112               # 0x70
    magic_len: 8
    feature_dim: 104               # 112 - 8
    multidex_mode: sum             # sum | mean | primary_only  (M11; sum = deployed default)
    dex_pattern: "^classes(\\d*)\\.dex$"

model:
  header_hidden: 128               # MLP(H) trunk width (matches deployed)  (M5)
  header_warm_start: true          # load deployed mlp_header trunk weights  (M5)
  receiver_embed_dim: 64           # d_R ≈ 32–64  (M7)
  fusion_hidden: 64                # joint head (128+d_R)→64→1  (M6/M7)
  fusion_head: mlp                 # mlp | logistic (bare-linear fallback)

training:
  batch_size: 64
  epochs: 60
  optimizer: adamw
  learning_rate: 0.005
  weight_decay: 0.0001
  pos_weight: auto                 # N_neg / N_pos on train  (M8)
  early_stop_patience: 6
  seed: 42

baseline:
  paper_svm: true                  # RBF-SVM on [H‖R] early-concat (offline only)
  svm_C: 10.0                      # #12 optimum
  svm_gamma: 0.1                   # #12 optimum
  svm_kernel: rbf

export:
  onnx_opset: 14
  parity_num_samples: 10
  parity_max_delta: 1.0e-4
```

### 3.2 P0 deliverables / exit criteria

| Deliverable | Exit criterion |
|-------------|----------------|
| `requirements.txt` | `pip install -r requirements.txt` succeeds |
| `verify_setup.py` | Imports torch/sklearn/onnx/androguard; loads YAML; asserts `apk_root` exists; asserts `system_actions.json` parses and is non-empty; asserts deployed `mlp_header` bundle reachable (warm-start) |
| `ensure_artifact_dirs()` | Creates `artifacts/{manifests,processed,checkpoints,metrics,export}` |
| README stub | Paper links, train years, run order, fusion architecture |

**Do not start P2 until P0 passes.**

---

## Phase P1 — Dataset indexing

### Goal
Machine-readable index of all APKs with `label`, `year`, `split`, integrity fields — no corpus copy.

### Tasks
1. Walk `apk_root/{year}/{benign|malware}/**/*.apk`.
2. Per file: SHA-256 (skip unreadable zips, log reason); record `apk_path`, `sha256`, `label` (0/1), `year`, `split`, optional `apk_size_bytes`, `num_dex_files`.
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

## Phase P2 — Feature extraction (dex min/max, receiver vocab A, tensors)

This is the **critical phase** — it owns corrections M1–M4, M10, M11. Sub-stages: (2a) dex header read, (2b) manifest receiver parse, (2c) system-action allow-list, (2d) freeze receiver vocab A from train, (2e) fit dex min/max from train, (2f) vectorize all, (2g) batch preprocess.

### 2a. Dex header read (`src/features/dex_header.py` + `multidex.py`) — **reuse from `only_base1_model`**
Per APK, **per dex** (`classes*.dex` via `^classes(\d*)\.dex$`, primary first):
1. Verify 8-byte magic (`dex\n035\0` / 037 / 038 / 039); on failure log to `failed_apks.log` and skip that dex.
2. Read bytes 8–111 → `h_b = byte_b / 255.0` → `float64[104]`.
3. **Sum-pool** across all valid dex → `H_raw ∈ [0, D_dex]^104` (`multidex_mode: sum`, M11).

> **M10 note:** only the 112-byte header per dex is parsed — no bytecode/string-pool scan. We still unzip and locate each dex, so cost ≈ constant *per dex*, not literally O(1).

### 2b. Manifest receiver parse (`src/features/manifest_decode.py` + `receivers.py`)
- Open APK ZIP → `AndroidManifest.xml`; decode binary XML (backend from config).
- Enumerate every static `<receiver>` → child `<intent-filter>` → child `<action android:name>`. Collect raw action names.
- Dedup within APK (set semantics).
- **Exclusions:** no permissions, no activities/services, no dynamically (context-) registered receivers, no `classes.dex` bytecode.

### 2c. System-action allow-list (`scripts/build_system_actions.py`) — **M4**
- Compile the set of Android OS broadcast action constants (`android.intent.action.*` like `BOOT_COMPLETED`, `USER_PRESENT`; `android.provider.Telephony.SMS_RECEIVED`; `android.net.conn.CONNECTIVITY_CHANGE`; `PHONE_STATE`; `NEW_OUTGOING_CALL`; …) across target API levels.
- Persist to `assets/system_actions.json` as `{ "actions": [ "<fully.qualified.ACTION>", ... ] }`.
- **Single source of truth** shipped unchanged to Android (parity guarantee).

### 2d. Receiver vocabulary A (`src/features/vocab.py`) — **M4**
Run **once on `split=train` APKs only**:
1. Collect receiver actions ∩ `system_actions.json`.
2. Count document frequency; keep those ≥ `receiver_action_min_doc_freq`.
3. Sort lexicographically → ordered `A`; write `receiver_action_vocab.json` `{"tokens":[...],"size":R}`.

### 2e. Dex corpus min–max (`src/features/normalization.py`) — **M1, M3** (reuse)
1. Over `split=train` APKs, stack the sum-pooled `H_raw` vectors.
2. `fit_minmax` → per-dimension `mins`, `maxs` (constant dims → denom 1).
3. Write `artifacts/processed/normalization_header.json` → `{"feature_dim":104,"mins":[...],"maxs":[...],"multidex_mode":"sum","dex_pattern":"..."}`.
   - **Critical (M3):** this file is the parity source of truth; the device loads the same file. If warm-starting from the deployed MLP(H), confirm the deployed normalization is comparable, or fine-tune end-to-end so the trunk adapts to this corpus's stats (document choice).

### 2f. Vectorization (`src/features/vectorize.py`)
For APK *i* (keep `H` and `R` **separate** — two-input model):
```
H[i] = minmax( sum_pool( bytes[8:112]/255 over classes*.dex ) )   ∈ [0,1]^104     # M1
R[i,k] = 1 if action_k ∈ A and present in static receivers, else 0   ∈ {0,1}^R     # M4
```
Store as **float32** (0.0/1.0 for R).

### 2g. Batch preprocess (`src/preprocessing/preprocess_apks.py`)
1. Load `A`, dex `mins/maxs`, `system_actions.json` (must exist).
2. For each APK (train + test): read dex + parse manifest → vectorize → append to shard buffers; failures → `artifacts/failed_apks.log`.
3. Write `features_train.pt`, `features_test.pt`, optional `features_val.pt` (each: `H`, `R`, `y`, `paths`, `sha256`).
4. Emit `preprocessing_meta.json`: `preprocessing_version` (git hash/date), `R`, `multidex_mode`, counts, failures.

### Exit criteria
- [ ] Dex `H` built via `/255` → sum-pool → corpus min–max **in this order** (M1); spot-check 5 APKs vs deployed BM1 extractor output
- [ ] `normalization_header.json` written; `feature_dim == 104`; `H ∈ [0,1]` (sanity: pre-norm max can exceed 1) (M2)
- [ ] Receiver vector built **only** from system-action allow-list (M4) — spot-check 5 APKs vs `aapt2`/apktool dump
- [ ] `R ≈ 30–80` logged
- [ ] No test APK influenced `A` or dex min/max
- [ ] train/test counts = P1 minus failures

---

## Phase P3 — DataLoaders

### Tasks
1. `FusionDataset.__getitem__` → `(H: float32[104], R: float32[R], y: long)`.
2. `build_dataloaders()`: `train_loader` (train split), `val_loader` (10% holdout from train), `test_loader` (2022+2023).
3. `batch_size` from config; `num_workers`/`pin_memory` per machine.

### Exit criteria
- [ ] One batch smoke test: shapes `[B,104]`, `[B,R]`; labels in `{0,1}`
- [ ] Per-split class balance printed (feeds `pos_weight`)

---

## Phase P4 — Model definition

### 4.1 Header tower (`src/models/header_tower.py`) — **M5**
```
block1: Linear(104→128) → BatchNorm1d(128) → ReLU
block2: Linear(128→128) → BatchNorm1d(128) → ReLU
# z_H = block2 output (128-d). NO →1→σ head here.
```
- `header_warm_start: true` → load `block1`/`block2` weights from the deployed `mlp_header` checkpoint (skip its `head`).
- Mirrors deployed `mlp_header.py` exactly so the branch has a guaranteed parity baseline.

### 4.2 Receiver tower (`src/models/receiver_tower.py`)
```
Linear(R → d_R) → ReLU       # d_R = receiver_embed_dim ≈ 32–64  (M7)
# optional BatchNorm1d(d_R) / Dropout(0.2)
```

### 4.3 Fusion net (`src/models/fusion_net.py`) — **M6, M7**
```
z_H = header_tower(H)          # 128
z_R = receiver_tower(R)        # d_R
z   = concat(z_H, z_R)         # 128 + d_R
# fusion_head = mlp:
Linear(128+d_R → 64) → ReLU → Dropout(0.2) → Linear(64 → 1)
# forward returns logits; sigmoid applied in export graph OR documented for Java
```
`BCEWithLogitsLoss(pos_weight=...)` (M8). `fusion_head: logistic` → bare `Linear(128+d_R → 1)` fallback.

### 4.4 Paper baseline — RBF-SVM (`src/training/svm_baseline.py`)
- Early-concat the raw features `[H ‖ R]` (numpy); `SVC(kernel="rbf", C=10, gamma=0.1, class_weight="balanced")`.
- Fit on train, evaluate on val + test. Save `svm_rbf.joblib`, `svm_metrics.json`. **Not** exported to ONNX.

### Exit criteria
- [ ] Fusion forward on dummy `(H=[1,104], R=[1,R])` works; param count well under 60 KB ONNX
- [ ] Warm-start loads deployed trunk weights without shape errors
- [ ] SVM baseline fits on a small subset

---

## Phase P5 — Training

### Tasks
1. **Fusion net (PyTorch):** AdamW (config lr/wd); `BCEWithLogitsLoss(pos_weight=auto)` (M8); per-epoch train/val loss, val F1, val AUC; early stop on val F1; save `artifacts/checkpoints/best.pt` (`model_state`, `R`, `d_R`, `config_hash`, `warm_start`).
2. **SVM baseline** (parallel script): metrics for the paper-faithful table.
3. **Ablations (thesis narrative, mirrors #12 three scenarios + header reference):**
   - **Header only** (`z_H` → 1; equals deployed MLP(H) — report its existing/measured number for reference, M9)
   - **Receivers only** (`z_R` → 1)
   - **Fusion** (`z`)

### Exit criteria
- [ ] Fusion val F1 ≥ max(header-only, receiver-only) — qualitative "fusion ≥ either"
- [ ] `best.pt` reloadable; training logs saved; record whether warm-start helped
- [ ] SVM baseline JSON exists

---

## Phase P6 — Evaluation (test split 2022 + 2023)

### Metrics
Accuracy, F1 (malware = positive), ROC-AUC, confusion matrix (TN, FP, FN, TP). Threshold: default 0.5; optional tuned on **val only** → `thresholds.json`.

### Output `artifacts/metrics/test_results.json`
```json
{
  "model_id": "dexheader_broadcast_fusion",
  "split": "test",
  "train_years": [2020, 2021],
  "test_years": [2022, 2023],
  "n_samples": 0,
  "feature_dims": { "dex_header": 104, "receiver": 0, "d_R": 0, "fused": 0 },
  "metrics": { "accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0 },
  "confusion_matrix": [[0, 0], [0, 0]],
  "threshold": 0.5,
  "ablations": {
    "header_only": { "f1": 0.0, "note": "ref: deployed MLP(H)" },
    "receivers_only": { "f1": 0.0 },
    "fusion": { "f1": 0.0 }
  },
  "paper_baseline": { "svm_rbf_concat": { "f1": 0.0 } }
}
```

### Exit criteria
- [ ] Eval reads only `features_test.pt` (never APKs)
- [ ] Ablation table present; confusion matrix optional plot
- [ ] **M9:** thesis text frames expected ~75–90% temporal F1; header-only number is the *measured* deployed value, not an MSFDroid headline

---

## Phase P7 — ONNX export bundle

### Directory `artifacts/export/dexheader_broadcast_fusion/`

### Tasks
1. Load `best.pt`; `model.eval()`; trace with two example inputs `(H=[1,104], R=[1,R])`.
2. Export `model.onnx`, opset 14, **two named inputs** `dex_header [1,104] float32`, `receiver [1,R] float32`, output `malware_prob` (sigmoid in graph — document).
3. Copy features: `receiver_action_vocab.json`, **`system_actions.json`** (M4), **`normalization_header.json`** (M1/M3), `feature_layout.json` → `features/`.
4. `thresholds.json`: `{ "default": 0.5, "tuned_val": <float> }`.
5. `export_manifest.json`:
```json
{
  "model_id": "dexheader_broadcast_fusion",
  "domain": "dex_header_receiver_actions",
  "opset": 14,
  "inputs": [
    { "name": "dex_header", "shape": [1, 104], "dtype": "float32" },
    { "name": "receiver", "shape": [1, "R"], "dtype": "float32" }
  ],
  "outputs": [{ "name": "malware_prob", "dtype": "float32" }],
  "preprocessing_version": "<date-or-git>",
  "multidex_mode": "sum",
  "feature_extraction": {
    "apk_parts": ["classes*.dex(header)", "AndroidManifest.xml"],
    "fusion": "embedding_concat_then_fc",
    "branches": ["dex_header_104", "receiver_system_actions"],
    "dex_feature_dim": 104,
    "dex_normalization": "per_byte_div255 -> multidex_sum -> corpus_minmax",
    "receiver_size_R": 0,
    "receiver_embed_dim_dR": 0,
    "receiver_system_actions_only": true
  }
}
```
6. `parity_samples/` (~10 APKs): each `H.npy`, `R.npy` + `expected_prob.json` from PyTorch.

### Exit criteria
- [ ] ONNX `< 60 KB`
- [ ] Both `system_actions.json` and `normalization_header.json` present in bundle (parity prerequisites)
- [ ] Bundle copies cleanly to `vigidroid/app/src/main/assets/models/dexheader_broadcast_fusion/`

---

## Phase P8 — Parity (PyTorch vs ONNX)

### Script `src/training/parity_onnx.py`
1. Load checkpoint + ONNX session.
2. For each `parity_samples/`: feed `(H, R)` to both → compare `malware_prob`.
3. Write `artifacts/metrics/parity_report.json` with per-sample delta + `max_delta`.

### Exit criteria
- [ ] `max_delta ≤ 1e-4` on all samples
- [ ] On failure: check float32 dtype, row-major layout, sigmoid placement, two-input name/order, and that frozen `normalization_header.json` was applied

---

## Android phases A1–A4

### A1 — Feature extractor (`DexheaderBroadcastFusionExtractor`)
Semantics **must** match Python P2:
1. Open APK ZIP.
2. **Dex branch:** read every `classes*.dex` header (reuse `DexHeaderFeatureExtractor`): bytes 8–111 `/255`, sum-pool, then `transformMinMax` with the shipped `normalization_header.json` (M1/M3) → `float[104]`.
3. **Receiver branch:** `AxmlReader` collects static `<receiver>` action names; load `receiver_action_vocab.json` + `system_actions.json`; set bit iff action ∈ (system_actions ∩ A) and present (M4) → `float[R]`.
4. Timings: `parse_ms`, `dex_ms`, `vectorize_ms`.
- **Deliverable:** instrumented unit test on 3 APKs comparing Java `H` and `R` to Python dumps.

### A2 — ONNX inference
1. `ModelRegistry` entry `dexheader_broadcast_fusion`.
2. Load `model.onnx` + manifest; ORT session; feed two inputs `dex_header [1,104]`, `receiver [1,R]` float32; read `malware_prob`.
3. Apply `thresholds.json`.

### A3 — Scan orchestration
Append one `stages[]` entry per scan:
```json
{
  "domain": "dex_header_receiver_actions",
  "model_id": "dexheader_broadcast_fusion",
  "parse_ms": 0, "dex_ms": 0, "vectorize_ms": 0, "inference_ms": 0,
  "score": 0.0, "mem_delta_bytes": 0
}
```
**Suggested cascade position:** a Step-2 structural+behavioral detector after a cheaper manifest gate (e.g. `simple_1` / broadcast hybrid) when uncertain.

### A4 — Instrumented parity test
- Load each `parity_samples/` input `(H, R)` → compare device score to `expected_prob` within `1e-4`.
- CI gate before release build.

### A1–A4 exit criteria
- [ ] End-to-end parity-APK scan passes on device/emulator
- [ ] Java `H` uses the **same** `normalization_header.json`; Java `R` uses the **same** `system_actions.json` as Python (M1/M3/M4)
- [ ] p50 `dex+parse+vectorize+inference` ms logged

---

## 4. Execution order and gates

```
P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → A1 → A2 → A3 → A4
            ↑ A and dex min/max frozen here (train only) — do NOT start A1 before P2 done on full train set
```

| Gate | Rule |
|------|------|
| G1 | P2 may not use any `split=test` APK to build `A` or dex min/max |
| G2 | P7 only after P5 produces `best.pt` |
| G3 | A1 only after `features/*.json` + `system_actions.json` + `normalization_header.json` exist in bundle |
| G4 | P8 green before copying bundle to VigiDroid |
| G5 | A4 green before reporting on-device numbers |

---

## 5. Thesis experiment hooks

| Thesis task | Contribution |
|-------------|--------------|
| Task 1 — Resource optimization | dex 112-B/dex header read + manifest parse + infer ms; ONNX `< 60 KB` |
| Task 2 — Multistep | Step-2 structural+behavioral detector; `t_low`/`t_high` on val holdout |
| Task 5 — Tradeoffs | ablation header-only vs receiver-only vs fused; F1 vs latency plot; warm-start vs scratch |
| Task 6 — Feasibility | header + receivers only, no bytecode scan → high deployability |
| Paper fidelity | RBF-SVM (γ=0.1, C=10) on `[H‖R]`; deployed MLP(H) as header reference |
| Ensemble (later) | export calibrated score; offline-learned weight |

### Comparison targets

| Config | Expected role |
|--------|---------------|
| Header only (MLP(H)) | deployed structural baseline (measured temporal number) |
| Receivers only | weaker alone (#12: ~71–79%); lifts sensitivity |
| **H + R fusion (this model)** | best structural+behavioral balance per parameter |
| vs Pattern B (H + 4381-d BoW) | ~50–100× smaller manifest branch, similar manifest value |

---

## 6. Per-phase checklists

### P2 feature parity checklist (Python internal)
- [ ] Dex header = bytes 8–111 `/255`; magic checked; bad dex logged
- [ ] Multidex sum-pool then corpus min–max **last** (M1); `multidex_mode` recorded
- [ ] Corpus min–max fit on train only; `normalization_header.json` written (M3)
- [ ] Receiver actions from **static `<receiver>` ∩ system-action allow-list** (M4)
- [ ] Set semantics within APK (dup tags → one bit)
- [ ] `A` built on train only; `R ≈ 30–80`

### P8 / A4 parity checklist (train vs device)
- [ ] Same `classes*.dex` set read; same dex `mins/maxs` file
- [ ] Same `A` order, same `system_actions.json` (same order → same index)
- [ ] float32; two inputs `dex_header`/`receiver` in correct name + order
- [ ] Same sigmoid/logits handling
- [ ] Score delta `≤ 1e-4` on all parity samples

---

## 7. Estimated effort

| Phase | Effort (solo, familiar stack) |
|-------|-------------------------------|
| P0–P1 | 0.5–1 day |
| P2 (dex reuse + manifest parse + system-action list + min/max) | 2.5–4 days |
| P3–P4 (two towers + warm-start + fusion head) | 1 day |
| P5–P6 (+ SVM baseline + ablations) | 1.5–2.5 days |
| P7–P8 (two-input ONNX) | 1 day |
| A1–A4 (Java parity: dex reuse + receiver system-action list) | 2–3 days |

**Critical path:** P2 ↔ A1 dex-normalization + system-action-list consistency, and the two-input ONNX graph parity.

---

## 8. Open decisions (defaults proposed above)

1. Header branch: warm-start from deployed MLP(H) trunk vs train from scratch? (default: warm-start, fine-tune)
2. Fusion head: MLP `(128+d_R)→64→1` vs bare logistic? (default: MLP, fall back to logistic if no val gain)
3. Receiver embedding `d_R`: 32 vs 64? (default: 64, tune on val for branch balance — M7)
4. Multidex: `sum` (deployed default) vs `primary_only` (MSFDroid-faithful)? (default: sum; ablate primary_only — M11)
5. Python manifest backend: `androguard` (paper-faithful) vs faster alternative? (default: androguard)
6. Two-input ONNX graph vs single early-concat input? (default: two inputs to keep branch semantics explicit; document either way)
7. System-action allow-list source/API-level coverage? (default: union across target minSdk..targetSdk — M4)
8. `apk_root` path on the training machine?
9. Cascade slot vs other dex/manifest models (BM1, simple_1, broadcast hybrid, cascade)?

---

## 9. References
- Verified rough plan: `detailed_implementation_plans/dexheader_brdcst_rec_fusion_opus.html`
- Tutorial: `detailed_implementation_plans/dexheader_brdcst_rec_fusion_tutorial.html`
- Pipeline guide: `sendable/Source_papers/Pipeline_full_concept.html`
- Sibling hybrid plans: `broadcast_mldp_hybrid_full_impl_opus.md`, `mldp_dexheader_cascade_full_impl_opus.md`, `simple_1_brd_rec_perm.md`
- Paper #12: `sendable/Source_papers/12_Detecting Android Malwares By Mining Statically Registered Broadcast Receivers (Full paper).pdf` (tutorial: `sendable/Source_papers/broadcast_receiver_paper_tutorial.html`)
- MSFDroid: **PDF not in workspace** — Dex header grounded on `Dex_header_paper_implementation/only_base1_model/` + `vigidroid/.../models/mlp_header/`
- Deployed MLP(H): `Dex_header_paper_implementation/only_base1_model/src/models/mlp_header.py`
- Model catalog / ranks: `todo_model_ranks.html`

---

*Document version: 2026-06-08 · Plan id: `hybrid_3_dexheader_broadcast` · Receivers cross-checked against source PDF #12; Dex header cross-checked against the deployed MLP(H) implementation (MSFDroid PDF unavailable).*
