# MLDP Permissions + Dex Header Cascade — Full Implementation Plan (Opus)

**Cross-paper hybrid:**
- **#7** Ghasempour, Mohd Sani & Abari — *Permission Extraction Framework for Android Malware Detection*, IJACSA 11(11), 2020 → **MLDP permission pruning** (PDF present, fully verified)
- **MSFDroid** — *Dex header structural bytes + MLP(H)* → **PDF absent from workspace**; the Dex-header block is verified against the **deployed implementation** in `Dex_header_paper_implementation/only_base1_model/` and the shipped bundle `vigidroid/app/src/main/assets/models/mlp_header/`.

**Verified rough plan:** `detailed_implementation_plans/mldp_dexheader_cascade_opus.html`
**Pipeline alignment:** `sendable/Source_papers/Pipeline_full_concept.html` (P0–P8 offline, A1–A4 on-device)
**Thesis category:** Hybrid (with modifications) — manifest + Dex-header, two deployment modes
**Reuse sources:**
- `Dex_header_paper_implementation/only_base1_model/` — Dex header extractor (`src/features/dex_header.py`, `multidex.py`), MLP(H) (`src/models/mlp_header.py`), ONNX export (`scripts/export_onnx.py`)
- `permission_extractor/` — working MLDP (`src/mldp/prnr.py`, `support_filter.py`, `association_rules.py`, `pipeline.py`)
- `vigidroid/app/src/main/java/com/msh/vigidroid/DexHeaderFeatureExtractor.java` + `MlpHeaderOnnxRunner.java` — Android parity baseline
- `detailed_implementation_plans/broadcast_mldp_hybrid_full_impl_opus.md` — sibling hybrid plan (same conventions)

---

## 0. Up-front assignments

| Field | Value | Notes |
|-------|-------|-------|
| **`model_id`** | `mldp_dexheader_cascade` | Used in `artifacts/export/`, Android `assets/models/`, metrics JSON |
| **`domain`** | `manifest_mldp_perm_dex_header` | Distinct from `dex_header_d3` (BM1) and `manifest_mldp_perm_receiver_actions` (broadcast hybrid) |
| **On-device feasible** | **Yes** | Manifest parse + 112-byte dex-header reads; target `< 150 ms` extract, `< 30 KB` ONNX |
| **Deployment modes** | **Mode A** fused single-pass MLP · **Mode B** two-stage cascade | Build both; pick per thesis experiment |
| **Mode A head** | Tiny MLP `d→64→1`, `d≈126` | ONNX opset 14 |
| **Mode B Stage 1** | Logistic `|S|→1` on MLDP perms (offline twin: SVM/DT) | new artifact |
| **Mode B Stage 2** | **Reuse deployed `mlp_header` ONNX** (`104→128→128→1`) | **no retrain** — load shipped model |
| **Paper baseline** | RBF C-SVM + Decision Tree (#7) on MLDP block | sklearn, offline only, not exported |
| **Project folder** | `mldp_dexheader_cascade/` | Self-contained training workspace |
| **Plan id** | `hybrid_2_mldp_dexheader` | Second hybrid in thesis queue |

### Architecture summary (target)

```
Raw APK
  ├─► AndroidManifest.xml ──► MLDP permission block   x_S ∈ {0,1}^|S|   (|S|≈22, frozen set S)
  └─► classes*.dex ──► /255 per dex ──► sum-pool ──► corpus min–max ──► H ∈ [0,1]^104

Mode A (fused):   x = [x_S ‖ H] ∈ R^d  (d≈126) ──► MLP(d→64→1) ──► P(malware)
Mode B (cascade): x_S ──► s1=σ(w·x_S+b) ──► exit if confident
                                          └─ else ──► H ──► MLP(H) 104→128→128→1 ──► s2 ──► exit/escalate
```

### Cross-check corrections folded into this plan (from the opus rough plan)

| # | Correction | Where enforced |
|---|------------|----------------|
| M1 | `|S| ≈ 16–25 (≈22)`, not 20–40 | P2 vocab build, config cap (`error if |S|>30`) |
| M2 | MLDP = **Variant 1** PRNR→SPR→PMAR (deterministic); published list fallback | P2 `src/features/mldp/` |
| M3 | PRNR = `(S_M−S_B)/(S_M+S_B) ∈ [-1,1]` + skew correction; keep both extremes | P2 `prnr.py` |
| M4 | Dex normalization = **3 ordered stages**: /255 → sum-pool → corpus min–max | P2 `dex_header.py` + `normalization.py` |
| M5 | Fused dim `d = |S|+104 ≈ 126` (not 150) | P4 model, export manifest shape |
| M6 | Mode A head `d→64→1`; Mode B Stage 2 **reuses deployed MLP(H) 104→128→128→1** | P4, P7 |
| M7 | Add #7 SVM + DT offline baselines on MLDP block | P5 `svm_baseline.py` |
| M8 | Multidex sum-pool default; `primary_only` ablation | P2 config, export manifest `multidex_mode` |
| M9 | Ship `normalization_header.json` + `mldp_permission_vocab.json`; both sides load same | P7, A1 |
| M10 | Dex parse cost framed honestly (112 B/dex, no bytecode scan) | thesis text, P2 docstring |
| M11 | Step-1 exit rate is a hypothesis → calibrate `t_low,t_high` on val; report realized rate | P6 |
| M12 | Class imbalance → `pos_weight` | P5 training |

---

## 1. Dependencies and risks

### 1.1 External dependencies

- **APK corpus** on disk (`apk_root/`), year folders `2020–2023`, `benign/` + `malware/` — not in git.
- **Manifest decoder (Python):** `androguard` (paper #7 used Androguard); lock version in P0.
- **Dex header reader (Python):** reuse `only_base1_model/src/features/dex_header.py` + `multidex.py` (ZIP enumerate `classes*.dex`, magic check, bytes 8–111).
- **MLDP mining:** reuse `permission_extractor/src/mldp/*`; `mlxtend` (Apriori) for PMAR or a small hand-rolled Apriori.
- **Deployed MLP(H) ONNX:** `vigidroid/.../models/mlp_header/model.onnx` + `features/normalization_header.json` — needed for Mode B Stage 2 (copy into this bundle).
- **Training:** PyTorch 2.x, scikit-learn (SVM/DT baseline), ONNX 1.x + onnxruntime, opset **14**.
- **Android decoder:** existing `AxmlReader` + `DexHeaderFeatureExtractor` in VigiDroid.

### 1.2 Risk register

| Risk | Mitigation |
|------|------------|
| Train/serve skew (Java ≠ Python features) | Shared golden APK set; **P8 + A4** parity on `parity_samples/` |
| **Dex normalization mismatch** (recompute on device vs frozen train stats) | Ship single `normalization_header.json`; both sides load it (M4, M9) |
| MLDP set / S leakage from test years | Build `S` and dex min/max **only** from `split=train` APKs in P2 |
| MLDP non-determinism | Variant 1 (PRNR→SPR→PMAR) fixed thresholds + seed; persist `mldp_trace.json` |
| Mode B Stage-2 drift from deployed MLP(H) | Reuse the *exact* shipped ONNX + its normalization; do not retrain |
| Obfuscated manifest / packed dex / bad magic | Log to `failed_apks.log`; exclude from counts (#7 notes this) |
| Multidex aggregation mismatch | Record `multidex_mode` in export manifest; default `sum` both sides (M8) |
| Class imbalance | `pos_weight` in BCE (M12) |
| Overstating exit rate / accuracy | Calibrate thresholds on val; report realized numbers (M11) |

---

## 2. Project layout

```
mldp_dexheader_cascade/
├── config/
│   └── default.yaml
├── requirements.txt
├── scripts/
│   ├── verify_setup.py
│   ├── index_dataset.py            # P1
│   ├── run_mldp.sh                 # P2: PRNR→SPR→PMAR → freeze S
│   ├── run_preprocess.sh           # P2 vectorize wrapper (manifest + dex)
│   ├── run_train.sh                # P5
│   ├── run_evaluate.sh             # P6
│   ├── export_onnx.py              # P7 (Mode A net; copies deployed MLP(H) for Mode B)
│   └── calibrate_thresholds.py     # P6: t_low/t_high on val holdout
├── src/
│   ├── config.py
│   ├── constants.py                # DEX_MAGIC, DEX_HEADER_SIZE=0x70, FEATURE_DIM=104, label names
│   ├── indexing/
│   │   └── build_manifest.py       # P1 CSV/JSON index
│   ├── features/
│   │   ├── manifest_decode.py      # APK → declared permissions
│   │   ├── permissions.py          # raw permission extraction + normalize names
│   │   ├── dex_header.py           # bytes 8–111 / 255  (reuse from only_base1_model)
│   │   ├── multidex.py             # sum-pool across classes*.dex (reuse)
│   │   ├── normalization.py        # fit/transform corpus min–max (reuse)
│   │   ├── mldp/
│   │   │   ├── prnr.py             # PRNR  R=(S_M−S_B)/(S_M+S_B), skew-correct (M3)
│   │   │   ├── spr.py              # Support-based ranking
│   │   │   ├── pmar.py             # Apriori association-rule collapse
│   │   │   └── select.py           # orchestrate → freeze S, write mldp_trace.json
│   │   ├── vocab.py                # build/freeze S (perms)
│   │   └── vectorize.py            # x_S, H, concat → x
│   ├── preprocessing/
│   │   └── preprocess_apks.py      # P2 batch job (manifest + dex)
│   ├── data/
│   │   ├── store.py
│   │   ├── dataset.py
│   │   └── dataloaders.py
│   ├── models/
│   │   ├── fused_mlp.py            # Mode A head (d→64→1)
│   │   ├── mldp_logistic.py        # Mode B Stage-1 (|S|→1)
│   │   └── mlp_header_ref.py       # loader/wrapper for deployed MLP(H) (Mode B Stage-2)
│   └── training/
│       ├── svm_baseline.py         # #7 RBF-SVM + DT on MLDP block (M7)
│       ├── train.py                # Mode A MLP + Mode B Stage-1 + ablations
│       ├── evaluate.py             # P6 (both modes + cascade exit-rate)
│       └── parity_onnx.py          # P8
└── artifacts/
    ├── manifests/                  # P1 index
    ├── processed/                  # P2 shards + S vocab + dex min/max + mldp_trace.json
    ├── checkpoints/
    ├── metrics/
    └── export/mldp_dexheader_cascade/
```

**Android (sibling repo `vigidroid/`):**

```
app/src/main/assets/models/mldp_dexheader_cascade/
├── mode_a/
│   ├── model.onnx                  # fused MLP d→64→1
│   ├── export_manifest.json
│   └── thresholds.json             # default + tuned + t_low/t_high
├── mode_b/
│   ├── stage1_mldp.onnx            # logistic |S|→1
│   ├── stage2_mlp_header.onnx      # COPY of deployed mlp_header model
│   ├── export_manifest.json
│   └── thresholds.json             # t_low/t_high for Stage 1
├── features/
│   ├── mldp_permission_vocab.json  # frozen S (ordered)
│   ├── normalization_header.json   # dex corpus min/max (same as BM1 family)  (M4/M9)
│   └── feature_layout.json         # {"order":["mldp_perms","dex_header"],"S":..,"H":104,"d":..}
└── parity_samples/
```

---

## 3. Configuration contract (P0)

### 3.1 `config/default.yaml`

```yaml
model_id: mldp_dexheader_cascade
domain: manifest_mldp_perm_dex_header

paths:
  apk_root: /path/to/apk_corpus    # EDIT per machine
  train_years: [2020, 2021]
  test_years: [2022, 2023]
  deployed_mlp_header_bundle: ../vigidroid/app/src/main/assets/models/mlp_header  # Mode B Stage-2 reuse

splits:
  val_fraction_of_train: 0.10      # early stopping + threshold calibration only

features:
  manifest_backend: androguard
  normalize_permission_names: true # strip "android.permission." consistently
  dex:
    header_size: 112               # 0x70
    magic_len: 8
    feature_dim: 104               # 112 - 8
    multidex_mode: sum             # sum | mean | primary_only  (M8; sum = deployed default)
    dex_pattern: "^classes(\\d*)\\.dex$"

mldp:                              # M2 — Variant 1 (deterministic)
  method: prnr_spr_pmar            # prnr_spr_pmar | published_list
  prnr_drop_abs_threshold: 0.05    # |R(Pj)| <= t → dropped (M3)
  skew_correction: true            # eq. 1/3 (M3)
  spr_keep_top: 25
  pmar_min_support: 0.10
  pmar_min_confidence: 0.965
  target_size_hint: 22             # |S| sanity bound (M1); error if |S|>30
  fallback_published_list: true    # Table I (22 perms) if mining degenerate

model:
  mode_a_hidden: 64                # fused MLP d→64→1
  mode_b_stage1: logistic          # logistic | tiny_mlp
  reuse_deployed_mlp_header: true  # M6 — Mode B Stage 2

training:
  batch_size: 256
  epochs: 60
  learning_rate: 0.005
  weight_decay: 0.0001
  pos_weight: auto                 # N_neg / N_pos on train (M12)
  early_stop_patience: 6
  seed: 42

cascade:
  target_false_omission_rate: 0.02 # used to pick t_low on val (M11)
  target_false_alarm_at_thigh: 0.02

baseline:
  paper_svm: true                  # M7
  svm_C: 10.0
  svm_gamma: 0.1
  svm_kernel: rbf
  decision_tree: true

export:
  onnx_opset: 14
  parity_num_samples: 10
  parity_max_delta: 1.0e-4
```

### 3.2 P0 deliverables / exit criteria

| Deliverable | Exit criterion |
|-------------|----------------|
| `requirements.txt` | `pip install -r requirements.txt` succeeds |
| `verify_setup.py` | Imports torch/sklearn/onnx/androguard/mlxtend; loads YAML; asserts `apk_root` exists; asserts deployed `mlp_header` bundle reachable (Mode B) |
| `ensure_artifact_dirs()` | Creates `artifacts/{manifests,processed,checkpoints,metrics,export}` |
| README stub | Paper links, train years, run order, Mode A vs Mode B |

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

## Phase P2 — Feature extraction (MLDP set S, dex min/max, tensors)

This is the **critical phase** — it owns corrections M1–M4, M8, M9. Sub-stages: (2a) manifest parse, (2b) dex header read, (2c) freeze S from train, (2d) fit dex min/max from train, (2e) vectorize all.

### 2a. Manifest parsing (`src/features/manifest_decode.py` + `permissions.py`)
- Open APK ZIP → `AndroidManifest.xml`; decode binary XML (backend from config).
- Collect every `<uses-permission android:name>`; normalize (strip `android.permission.`); dedup within APK (set semantics).
- Document whether `uses-permission-sdk-23` is included (default: include, normalized).
- **Exclusions:** no `classes.dex` bytecode, no receivers/activities (this model uses permissions + header only).

### 2b. Dex header read (`src/features/dex_header.py` + `multidex.py`) — **reuse from `only_base1_model`**
Per APK, **per dex** (`classes*.dex` discovered via `^classes(\d*)\.dex$`, primary first):
1. Verify 8-byte magic `dex\n035\0`; on failure log to `failed_apks.log` and skip the dex.
2. Read bytes 8–111 → `h_b = byte_b / 255.0` → `float64[104]`.
3. **Sum-pool** across all valid dex → `H_raw ∈ [0, D_dex]^104` (`multidex_mode: sum`, M8).

> **M10 note:** only the 112-byte header per dex is parsed — no bytecode/string-pool scan. Cheap and roughly constant per dex.

### 2c. MLDP selection → freeze S (`src/features/mldp/`) — **M2, M3** (reuse `permission_extractor/src/mldp/*`)
Run **once on `split=train` APKs only**:
1. Build `M` (malware) and `B` (benign) binary permission matrices over the **full** permission vocabulary observed in train.
2. **Skew correction (eq. 1/3):** scale the larger class's supports down to match the smaller (M3).
3. **PRNR (`prnr.py`):** `R(P_j) = (S_M(P_j) − S_B(P_j)) / (S_M(P_j) + S_B(P_j)) ∈ [-1,1]`. Drop `|R| ≤ prnr_drop_abs_threshold`. **Keep both extremes** (M3). ~135 → ~95.
4. **SPR (`spr.py`):** rank survivors by overall support; keep top `spr_keep_top` (≈25).
5. **PMAR (`pmar.py`):** Apriori (`min_support=0.10`, `min_confidence=0.965`); collapse implied permissions (keep one representative). ~25 → **~22**.
6. Freeze ordered `S`; write:
   - `artifacts/processed/mldp_permission_vocab.json` → `{"tokens":[...], "size":|S|}`
   - `artifacts/processed/mldp_trace.json` → per-stage counts, `R(P_j)` table, dropped/kept lists, rules found.
7. **Guards:** `|S| > 30` → raise (violates M1). If degenerate (`|S| < 8`) and `fallback_published_list: true` → use #7 Table I (22 perms), record fallback in trace.

### 2d. Dex corpus min–max (`src/features/normalization.py`) — **M4, M9** (reuse)
1. Over `split=train` APKs, stack the sum-pooled `H_raw` vectors.
2. `fit_minmax` → per-dimension `mins`, `maxs` (constant dims → denom 1).
3. Write `artifacts/processed/normalization_header.json` → `{"feature_dim":104,"mins":[...],"maxs":[...],"multidex_mode":"sum","dex_pattern":"..."}`.
   - **Critical:** this file is the parity source of truth (M9). For Mode B Stage 2, instead use the deployed `mlp_header` bundle's own `normalization_header.json` (so Stage-2 input matches the model it was trained with).

> **Decision (document):** Mode A trains a *new* fused net, so it fits its own dex min/max on this corpus. Mode B Stage 2 reuses the deployed MLP(H), so it must use the *deployed* min/max. Keep both files clearly labelled in the bundle.

### 2e. Vectorization (`src/features/vectorize.py`)
For APK *i*:
```
x_S[i,j] = 1 if perm_j ∈ S and declared, else 0
H[i]     = minmax( sum_pool( bytes[8:112]/255 over classes*.dex ) )   ∈ [0,1]^104
x[i]     = concat(x_S[i], H[i]) ∈ R^(|S|+104)           # Mode A input
```
Store as **float32**. Keep `x_S` and `H` separately too (Mode B needs them apart).

### 2f. Batch preprocess (`src/preprocessing/preprocess_apks.py`)
1. Load `S`, dex `mins/maxs` (must exist).
2. For each APK (train + test): parse manifest + read dex → vectorize → append to shard buffers; failures → `artifacts/failed_apks.log`.
3. Write `features_train.pt`, `features_test.pt`, optional `features_val.pt` (each: `x_S`, `H`, `x`, `y`, `paths`, `sha256`).
4. Emit `preprocessing_meta.json`: `preprocessing_version` (git hash/date), `|S|`, `d`, `multidex_mode`, counts, failures.

### Exit criteria
- [ ] `|S| ≈ 16–25` (≤30) logged; `mldp_trace.json` present (M1, M2)
- [ ] PRNR table shows both malware- and benign-indicative perms retained (M3)
- [ ] Dex `H` built via /255 → sum-pool → corpus min–max (M4); spot-check 5 APKs vs deployed BM1 extractor output
- [ ] `normalization_header.json` written; `feature_dim == 104`
- [ ] No test APK influenced `S` or dex min/max
- [ ] train/test counts = P1 minus failures

---

## Phase P3 — DataLoaders

### Tasks
1. `CascadeDataset.__getitem__` → `(x_S: float32[|S|], H: float32[104], x: float32[d], y: long)`.
2. `build_dataloaders()`: `train_loader` (train split), `val_loader` (10% holdout from train), `test_loader` (2022+2023).
3. `batch_size` from config; `num_workers`/`pin_memory` per machine.

### Exit criteria
- [ ] One batch smoke test: shapes `[B,|S|]`, `[B,104]`, `[B,d]`; labels in `{0,1}`
- [ ] Per-split class balance printed (feeds `pos_weight`)

---

## Phase P4 — Model definition

### 4.1 Mode A — Fused tiny MLP (`src/models/fused_mlp.py`) — **M5, M6**
```
Linear(d → 64) → ReLU → Dropout(0.2) → Linear(64 → 1)     # d = |S| + 104 ≈ 126
# forward returns logits; sigmoid applied in export graph (document for Java)
```
`BCEWithLogitsLoss(pos_weight=...)`. Optional BatchNorm1d after layer 1 for parity with BM1 style.

### 4.2 Mode B Stage 1 — MLDP logistic (`src/models/mldp_logistic.py`)
`nn.Linear(|S|, 1)` → sigmoid. Smallest possible head; offline twin = SVM/DT (4.4).

### 4.3 Mode B Stage 2 — reuse deployed MLP(H) (`src/models/mlp_header_ref.py`) — **M6**
- **Do not retrain.** Load the deployed `mlp_header` ONNX (`104→128→128→1`, BN+ReLU+sigmoid) from `deployed_mlp_header_bundle`.
- This wrapper just runs that ONNX on `H` (normalized with the **deployed** stats) and returns `s2`.

### 4.4 Paper baseline — RBF-SVM + DT on MLDP block (`src/training/svm_baseline.py`) — **M7**
- `SVC(kernel="rbf", C=10, gamma=0.1, class_weight="balanced")` and `DecisionTreeClassifier`, fit on `x_S` (train), evaluate on val + test.
- Save `svm_rbf.joblib`, `svm_metrics.json`, `dt_metrics.json`. **Not** exported to ONNX.

### Exit criteria
- [ ] Mode A forward on dummy `[1,d]` works; param count well under 30 KB ONNX
- [ ] Stage-1 logistic forward on `[1,|S|]` works
- [ ] Stage-2 wrapper loads deployed ONNX and scores a dummy `[1,104]`
- [ ] SVM/DT baseline fits on a small subset

---

## Phase P5 — Training

### Tasks
1. **Mode A MLP (PyTorch):** AdamW (config lr/wd); `BCEWithLogitsLoss(pos_weight=auto)` (M12); per-epoch train/val loss, val F1, val AUC; early stop on val F1; save `artifacts/checkpoints/mode_a_best.pt` (`model_state`, `S`, `d`, `config_hash`).
2. **Mode B Stage 1 (PyTorch):** same loop on `x_S` only → `stage1_best.pt`.
3. **SVM/DT baseline** (parallel script): metrics for the #7 "paper-faithful" table (M7).
4. **Ablations (thesis narrative):**
   - **MLDP perms only** (`x_S`)
   - **Dex header only** (`H` — equals deployed MLP(H), report its existing number for reference)
   - **Mode A fusion** (`x`)

> Mode B Stage 2 is **not trained** here — it reuses the deployed model.

### Exit criteria
- [ ] Mode A val F1 ≥ max(perm-only, header-only) — qualitative "fusion ≥ either"
- [ ] `mode_a_best.pt` and `stage1_best.pt` reloadable; training logs saved
- [ ] SVM/DT baseline JSON exists

---

## Phase P6 — Evaluation (test split 2022 + 2023)

### Metrics
Accuracy, F1 (malware = positive), ROC-AUC, confusion matrix. For **Mode B**: also the **realized Step-1 exit rate**, the false-omission rate at `t_low`, and end-to-end cascade F1/latency at the chosen operating point.

### Threshold calibration (`scripts/calibrate_thresholds.py`) — **M11**
- On the **val holdout only**, pick `t_low` so the false-omission rate ≤ `target_false_omission_rate`, and `t_high` so the false-alarm rate ≤ `target_false_alarm_at_thigh`.
- Persist to `thresholds.json`. Then evaluate on test and **report the realized exit rate** — do not assume "70%".

### Output `artifacts/metrics/test_results.json`
```json
{
  "model_id": "mldp_dexheader_cascade",
  "split": "test",
  "train_years": [2020, 2021],
  "test_years": [2022, 2023],
  "n_samples": 0,
  "feature_dims": { "S": 0, "H": 104, "d": 0 },
  "mode_a": { "accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0, "confusion_matrix": [[0,0],[0,0]], "threshold": 0.5 },
  "mode_b": {
    "stage1_t_low": 0.0, "stage1_t_high": 0.0,
    "step1_exit_rate": 0.0, "false_omission_rate": 0.0,
    "end_to_end_f1": 0.0, "end_to_end_acc": 0.0
  },
  "ablations": {
    "mldp_perms_only": { "f1": 0.0 },
    "dex_header_only": { "f1": 0.0 },
    "mode_a_fusion": { "f1": 0.0 }
  },
  "paper_baselines": { "svm_rbf": { "f1": 0.0 }, "decision_tree": { "f1": 0.0 } }
}
```

### Exit criteria
- [ ] Eval reads only `features_*.pt` (never APKs)
- [ ] Ablation table present
- [ ] **M11:** realized Step-1 exit rate reported (not assumed); thresholds came from val only
- [ ] Honest accuracy framing in thesis text (latency/exit-rate focus)

---

## Phase P7 — ONNX export bundle

### Directory `artifacts/export/mldp_dexheader_cascade/`

### Tasks
1. **Mode A:** load `mode_a_best.pt`; trace `[1,d]`; export `mode_a/model.onnx` (opset 14), input `features` `[1,d]` float32, output `malware_prob` (sigmoid in graph — document).
2. **Mode B Stage 1:** export `mode_b/stage1_mldp.onnx`, input `[1,|S|]`.
3. **Mode B Stage 2:** **copy** the deployed `mlp_header/model.onnx` → `mode_b/stage2_mlp_header.onnx` (M6); copy its `normalization_header.json` too.
4. Copy features: `mldp_permission_vocab.json`, `normalization_header.json` (Mode A's), `feature_layout.json` → `features/`.
5. `thresholds.json`: Mode A `{default, tuned_val}`; Mode B `{t_low, t_high}` (M11).
6. `export_manifest.json` per mode:
```json
{
  "model_id": "mldp_dexheader_cascade",
  "mode": "A",
  "domain": "manifest_mldp_perm_dex_header",
  "opset": 14,
  "inputs": [{ "name": "features", "shape": [1, "D"], "dtype": "float32" }],
  "outputs": [{ "name": "malware_prob", "dtype": "float32" }],
  "preprocessing_version": "<date-or-git>",
  "multidex_mode": "sum",
  "feature_extraction": {
    "apk_parts": ["AndroidManifest.xml", "classes*.dex(header)"],
    "fusion": "early_concat",
    "blocks": ["mldp_perms", "dex_header_104"],
    "mldp_size_S": 0,
    "dex_feature_dim": 104,
    "dex_normalization": "per_byte_div255 -> multidex_sum -> corpus_minmax"
  }
}
```
7. `parity_samples/` (~10 APKs): each `x_S.npy`, `H.npy`, `x.npy` + `expected_prob.json` (Mode A `ŷ`, Mode B `s1`, `s2`) from PyTorch/ONNX.

### Exit criteria
- [ ] Mode A ONNX `< 30 KB`; Stage-1 ONNX tiny
- [ ] Stage-2 ONNX is a byte-copy of the deployed `mlp_header` model
- [ ] `normalization_header.json` + `mldp_permission_vocab.json` present (M9)
- [ ] Bundle copies cleanly to `vigidroid/.../models/mldp_dexheader_cascade/`

---

## Phase P8 — Parity (PyTorch vs ONNX)

### Script `src/training/parity_onnx.py`
1. Load checkpoints + ONNX sessions (Mode A, Stage 1, Stage 2).
2. For each `parity_samples/`: run both → compare scores.
3. Write `artifacts/metrics/parity_report.json` with per-sample delta + `max_delta` per output.

### Exit criteria
- [ ] `max_delta ≤ 1e-4` on all samples and all outputs
- [ ] On failure: check float32 dtype, row-major layout, sigmoid placement, **and that the right normalization stats were applied** (Mode A's vs deployed for Stage 2)

---

## Android phases A1–A4

### A1 — Feature extractor (`MldpDexHeaderExtractor`)
Semantics **must** match Python P2:
1. Open APK ZIP → `AndroidManifest.xml`; `AxmlReader` collects permission names (normalize, dedup).
2. Build `x_S` (`float[|S|]`) from `mldp_permission_vocab.json`.
3. Read every `classes*.dex` header (reuse `DexHeaderFeatureExtractor`): bytes 8–111 `/255`, sum-pool, then `transformMinMax` with the shipped `normalization_header.json` (M4/M9).
4. **Mode A:** concat → `float[d]`. **Mode B:** keep `x_S` and `H` separate.
5. Timings: `parse_ms`, `dex_ms`, `vectorize_ms`.
- **Deliverable:** instrumented unit test on 3 APKs comparing Java `x_S`, `H`, `x` to Python dumps.

### A2 — ONNX inference
1. `ModelRegistry` entries: `mldp_dexheader_cascade_mode_a`, `..._mode_b`.
2. **Mode A:** load `mode_a/model.onnx`; input `[1,d]`; read `malware_prob`; apply `thresholds.json`.
3. **Mode B:** run `stage1_mldp.onnx` → `s1`; if `t_low < s1 < t_high`, read dex → `stage2_mlp_header.onnx` → `s2`; else early-exit (skip dex read).

### A3 — Scan orchestration
Append one `stages[]` entry per scan (Mode B may record both stages):
```json
{
  "domain": "manifest_mldp_perm_dex_header",
  "model_id": "mldp_dexheader_cascade",
  "mode": "B",
  "parse_ms": 0, "dex_ms": 0, "vectorize_ms": 0, "inference_ms": 0,
  "stage1_score": 0.0, "stage2_score": 0.0, "early_exit": true,
  "score": 0.0, "mem_delta_bytes": 0
}
```
**Suggested cascade position:** earliest cheap gate (before manifest BoW / ByteCNN). Record whether the dex read was skipped (Mode B early exit) — this is the latency win for thesis Task 1/2.

### A4 — Instrumented parity test
- Load each `parity_samples/` input → compare device scores (`ŷ` / `s1` / `s2`) to `expected_prob` within `1e-4`.
- CI gate before release build.

### A1–A4 exit criteria
- [ ] End-to-end parity-APK scan passes on device/emulator (both modes)
- [ ] Java dex `H` uses the **same** `normalization_header.json` as Python (M9)
- [ ] Mode B early-exit demonstrably skips the dex read on confident Stage-1 cases
- [ ] p50 `parse+dex+vectorize+inference` ms logged

---

## 4. Execution order and gates

```
P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → A1 → A2 → A3 → A4
            ↑ S and dex min/max frozen here (train only) — do NOT start A1 before P2 done on full train set
```

| Gate | Rule |
|------|------|
| G1 | P2 may not use any `split=test` APK to build `S` or dex min/max |
| G2 | P7 only after P5 produces `mode_a_best.pt` + `stage1_best.pt` |
| G3 | Mode B Stage 2 ONNX must be a copy of the deployed `mlp_header` model (no retrain) |
| G4 | A1 only after `features/*.json` (+ Stage-2 deployed normalization) exist in bundle |
| G5 | P8 green before copying bundle to VigiDroid |
| G6 | A4 green before reporting on-device numbers |

---

## 5. Thesis experiment hooks

| Thesis task | Contribution |
|-------------|--------------|
| Task 1 — Resource optimization | manifest + 112-B/dex header read + infer ms; ONNX `< 30 KB` |
| Task 2 — Multistep | Mode B early-exit; `t_low`/`t_high` on val holdout; realized exit rate (M11) |
| Task 5 — Tradeoffs | ablation MLDP-only vs header-only vs fused; F1 vs latency; Mode A vs Mode B |
| Task 6 — Feasibility | manifest + header only, no bytecode scan → high deployability |
| Paper fidelity | #7 RBF-SVM + DT on MLDP block; deployed MLP(H) as header reference |
| Ensemble (later) | export calibrated score; offline-learned weight |

### Comparison targets

| Config | Expected role |
|--------|---------------|
| MLDP perms only | strong solo (#7: F ≈ 94–97% on their corpus; lower on temporal split) |
| Dex header only (MLP(H)) | deployed structural baseline |
| **MLDP + H (this model)** | best cheap-static balance per parameter; fast Step-1 filter |

---

## 6. Per-phase checklists

### P2 feature parity checklist (Python internal)
- [ ] Permissions from `<uses-permission>` only; normalized consistently
- [ ] Dex header = bytes 8–111 `/255`; magic checked; bad dex logged
- [ ] Multidex sum-pool (M8); `multidex_mode` recorded
- [ ] Corpus min–max fit on train only; `normalization_header.json` written (M4)
- [ ] `S` via PRNR→SPR→PMAR on train only; `mldp_trace.json` written (M2/M3)
- [ ] `|S| ≤ 30` guard passed (M1)

### P8 / A4 parity checklist (train vs device)
- [ ] Same manifest bytes + same `classes*.dex` set read
- [ ] Same `S` order, same dex `mins/maxs` file
- [ ] Mode B Stage 2 uses the **deployed** normalization, not Mode A's
- [ ] float32; sigmoid handling consistent (Mode A: in-graph; Stage 2: in-graph)
- [ ] Score delta `≤ 1e-4` on all parity samples and outputs

---

## 7. Estimated effort

| Phase | Effort (solo, familiar stack) |
|-------|-------------------------------|
| P0–P1 | 0.5–1 day |
| P2 (manifest parse + MLDP mining + dex reuse + min/max) | 2.5–4 days |
| P3–P4 | 0.5–1 day |
| P5–P6 (+ SVM baseline + ablations + threshold calibration) | 1.5–2.5 days |
| P7–P8 (two modes, reuse Stage-2) | 1 day |
| A1–A4 (Java parity, dex reuse, cascade branch logic) | 2–3 days |

**Critical path:** P2 ↔ A1 dex-normalization + S consistency, and the Mode B Stage-2 reuse of the deployed bundle.

---

## 8. Open decisions (defaults proposed above)

1. MLDP method: **Variant 1 (PRNR→SPR→PMAR)** vs published-list? (default: Variant 1, list fallback)
2. Include `uses-permission-sdk-23` in permission space? (default: yes, normalized)
3. Ship **both** Mode A and Mode B, or pick one? (default: both; thesis compares them)
4. Mode B Stage 1: logistic vs tiny MLP? (default: logistic — smallest, interpretable)
5. Multidex: `sum` (deployed default) vs `primary_only` (MSFDroid-faithful)? (default: sum; ablate primary_only)
6. Cascade operating point: target false-omission rate for `t_low`? (default: 0.02, tune on val)
7. `apk_root` path on the training machine?
8. Cascade slot vs other manifest/header models (BM1, broadcast hybrid)?

---

## 9. References
- Verified rough plan: `detailed_implementation_plans/mldp_dexheader_cascade_opus.html`
- Tutorial: `detailed_implementation_plans/mldp_dexheader_cascade_tutorial.html`
- Pipeline guide: `sendable/Source_papers/Pipeline_full_concept.html`
- Sibling hybrid plan: `detailed_implementation_plans/broadcast_mldp_hybrid_full_impl_opus.md`
- Paper #7: `sendable/Source_papers/7_Permission Extraction Framework for Android Malware Detection.pdf`
- MSFDroid: **PDF not in workspace** — Dex header grounded on `Dex_header_paper_implementation/only_base1_model/` + `vigidroid/.../models/mlp_header/`
- Deployed MLP(H): `Dex_header_paper_implementation/only_base1_model/src/models/mlp_header.py`
- Working MLDP: `permission_extractor/src/mldp/`
- Model catalog / ranks: `todo_model_ranks.html`

---

*Document version: 2026-06-07 · Plan id: `hybrid_2_mldp_dexheader` · MLDP cross-checked against source PDF #7; Dex header cross-checked against the deployed MLP(H) implementation (MSFDroid PDF unavailable).*
