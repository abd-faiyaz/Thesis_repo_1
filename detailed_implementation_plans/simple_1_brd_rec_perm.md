# Broadcast Receiver + Permission Classifier — Detailed Implementation Plan

**Paper:** Mohsen et al., CIC 2017 — *Detecting Android Malwares By Mining Statically Registered Broadcast Receivers*  
**Rough architecture:** `rough_model_explanations/broadcast_receiver_permission_classifier.html`  
**Pipeline alignment:** `Pipeline_full_concept.html` (P0–P8 offline, A1–A4 on-device)  
**Thesis category:** Original / paper-faithful model (manifest-only, early fusion)

---

## 0. Up-front assignments

| Field | Proposed value | Notes |
|-------|----------------|-------|
| **`model_id`** | `broadcast_receiver_perm` | Used in `artifacts/export/`, Android `assets/models/`, metrics JSON |
| **`domain`** | `manifest_perm_receiver_actions` | Distinct from D1 MH1M BoW (`mh1m_2500_rp`) and from Dex-header domains |
| **On-device feasible** | **Yes** | Manifest-only parse; target &lt;100 ms extract, &lt;50 KB ONNX (per rough plan) |
| **Fusion type** | **Early fusion** | \(\mathbf{x} = [\mathbf{p} \,\|\, \mathbf{r}]\) → single classifier (not late fusion / voting) |
| **Project folder** | `broadcast_receiver_perm/` | Self-contained under training workspace (Thesis_repo_1 or local clone) |
| **Plan file id** | `simple_1` | First “simple” original-model implementation in thesis queue |

### Architecture summary (target)

```
Raw APK
  → AndroidManifest.xml (binary XML decode)
       ├─► Permission BoW p ∈ {0,1}^P     (uses-permission names)
       └─► Receiver-action BoW r ∈ {0,1}^R (<receiver>/<intent-filter>/<action>)
  → x = concat(p, r) ∈ {0,1}^(P+R)   typically P+R ≈ 200–350
  → Classifier → P(malware) ∈ [0,1]
```

### Paper vs thesis deployment split

| Aspect | Paper (faithful offline) | Thesis / VigiDroid (ONNX) |
|--------|--------------------------|---------------------------|
| Classifier | C-SVM, RBF kernel; grid-search \(C\), \(\gamma\) | **Logistic regression** (preferred) or 1-hidden-layer MLP |
| Features | Full permission vocab + receiver-action vocab from training set | Same semantics; vocabs frozen in `features/vocab.json` |
| Evaluation | Reported ~5,723+5,723 APK experiments | Train **2020–2021**, test **2022–2023** (temporal split per pipeline guide) |
| Goal | Reproduce “permissions + receivers &gt; either alone” | Ship ONNX + Java parity for on-device scan stage |

**Recommendation:** Run **both** classifiers in P5/P6 — sklearn `SVC(RBF)` as `paper_baseline`, PyTorch `LogisticRegression` or `TinyMLP` as `deployment_model`. Export ONNX only from the deployment model.

---

## 1. Dependencies and risks

### 1.1 External dependencies

- **APK corpus** on disk (`apk_root/`), year folders 2020–2023, `benign/` + `malware/` — not in git
- **Manifest decoder (Python):** `androguard`, `axmlparserpy`, or subprocess `aapt2 dump xmltree` — pick one and lock version in P0
- **Manifest decoder (Android):** existing `AxmlReader` in VigiDroid (align tag/attribute traversal with Python)
- **Training:** PyTorch 2.x, scikit-learn (SVM baseline), ONNX 1.x, opset **14**

### 1.2 Risk register

| Risk | Mitigation |
|------|------------|
| Train/serve skew (Java ≠ Python manifest parse) | Shared golden APK set; P8 + A4 parity on `parity_samples/` |
| Vocabulary leakage from test years | Build \(\mathcal{P}\), \(\mathcal{A}\) **only** from `split=train` APKs in P2 |
| Overlap with deployed D1 MH1M | Document as complementary domain; compare F1/latency in thesis ablation |
| Obfuscated / broken manifests | Log to `failed_apks.log`; exclude from training counts |
| Dynamic receivers | Paper uses **static** `<receiver>` in manifest only — do not mine bytecode |
| Sparse 0-vector APKs | Keep in dataset; model should handle all-zero rows |
| Class imbalance | `pos_weight` in BCE / `class_weight` in SVM; report F1 + AUC |

---

## 2. Project layout

```
broadcast_receiver_perm/
├── config/
│   └── default.yaml
├── requirements.txt
├── scripts/
│   ├── verify_setup.py
│   ├── index_dataset.py          # P1
│   ├── run_preprocess.sh         # P2 wrapper
│   ├── run_train.sh              # P5
│   ├── run_evaluate.sh           # P6
│   └── export_onnx.py            # P7
├── src/
│   ├── config.py
│   ├── constants.py              # label names, manifest tag constants
│   ├── indexing/
│   │   └── build_manifest.py     # P1: CSV/JSON manifest
│   ├── features/
│   │   ├── manifest_decode.py    # APK → parsed manifest tree / dict
│   │   ├── permissions.py        # extract + vectorize p
│   │   ├── receivers.py          # extract + vectorize r
│   │   ├── vocab.py              # build/freeze P, A
│   │   └── vectorize.py          # concat → x
│   ├── preprocessing/
│   │   └── preprocess_apks.py    # P2 batch job
│   ├── data/
│   │   ├── store.py
│   │   ├── dataset.py
│   │   └── dataloaders.py
│   ├── models/
│   │   ├── logistic_head.py      # deployment
│   │   └── tiny_mlp.py           # optional alternative
│   └── training/
│       ├── svm_baseline.py       # paper-faithful (sklearn)
│       ├── train.py
│       ├── evaluate.py
│       └── parity_onnx.py        # P8
└── artifacts/
    ├── manifests/                # P1 index
    ├── processed/                # P2 shards + vocab
    ├── checkpoints/
    ├── metrics/
    └── export/broadcast_receiver_perm/
```

**Android (sibling repo `vigidroid/`):**

```
app/src/main/assets/models/broadcast_receiver_perm/
├── model.onnx
├── export_manifest.json
├── thresholds.json
├── features/
│   ├── permission_vocab.json
│   ├── receiver_action_vocab.json
│   └── feature_layout.json       # [p_indices | r_indices], dim P+R
└── parity_samples/
```

---

## 3. Configuration contract (P0)

### 3.1 `config/default.yaml` (minimum)

```yaml
model_id: broadcast_receiver_perm
domain: manifest_perm_receiver_actions

paths:
  apk_root: /path/to/apk_corpus    # EDIT per machine
  train_years: [2020, 2021]
  test_years: [2022, 2023]

splits:
  val_fraction_of_train: 0.10      # early stopping / threshold tuning only

features:
  manifest_backend: axmlparserpy   # or androguard | aapt2
  permission_min_doc_freq: 1       # TUNE: drop rare perms on train only
  receiver_action_min_doc_freq: 1
  max_permission_vocab: null       # optional cap, e.g. 300
  max_receiver_vocab: null         # optional cap, e.g. 100
  normalize_permission_names: true # strip android.permission. prefix
  include_protection_level: false  # paper uses name only
  receiver_scope: static_manifest_only
  unknown_token_policy: ignore     # ignore | bucket_other (single extra dim per block)

classifier:
  deployment: logistic             # logistic | tiny_mlp
  paper_baseline_svm: true
  tiny_mlp_hidden: 64

training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.01
  weight_decay: 0.0001
  pos_weight: auto                 # n_neg / n_pos on train
  early_stop_patience: 5
  seed: 42

export:
  onnx_opset: 14
  parity_num_samples: 10
  parity_max_delta: 1.0e-4
```

### 3.2 P0 deliverables and exit criteria

| Deliverable | Exit criterion |
|-------------|----------------|
| `requirements.txt` | `pip install -r requirements.txt` succeeds |
| `verify_setup.py` | Imports torch/sklearn/onnx; loads YAML; prints resolved `apk_root` exists |
| `ensure_artifact_dirs()` | Creates `artifacts/{processed,checkpoints,metrics,export}` |
| README stub | One paragraph: paper link, train years, how to run `run_preprocess.sh` |

**Do not start P2 until P0 passes on the training machine.**

---

## Phase P1 — Dataset indexing

### Goal

Build a machine-readable manifest of all APKs with **label**, **year**, **split**, and integrity fields — without copying the 500 GB corpus.

### Tasks

1. Walk `apk_root/{year}/{benign|malware}/**/*.apk`.
2. For each file:
   - Compute **SHA-256**; skip unreadable ZIPs with reason logged.
   - Record: `apk_path`, `sha256`, `label` (0 benign / 1 malware), `year`, `split`.
   - Assign `split`: `train` if `year ∈ {2020,2021}`, `test` if `year ∈ {2022,2023}`.
   - Optional: `apk_size_bytes`, `num_dex_files` (zip listing only).
3. **Deduplicate** by `sha256` — keep first path, drop duplicates with log entry.
4. Write `artifacts/manifests/apk_index.csv` (and optional `apk_index.json`).

### Output schema

| Column | Type | Description |
|--------|------|-------------|
| `apk_path` | str | Absolute or corpus-relative path |
| `sha256` | str | Hex digest |
| `label` | int | 0 / 1 |
| `year` | int | 2020–2023 |
| `split` | str | `train` \| `test` |
| `apk_size_bytes` | int | Optional |

### Script

- `scripts/index_dataset.py --config config/default.yaml`

### Exit criteria

- [ ] Row counts per year/label/split printed
- [ ] No APK from 2022/2023 in train shard used for vocab (split column correct)
- [ ] Duplicate hash report generated
- [ ] `failed_index.log` for corrupt APKs

---

## Phase P2 — Feature extraction (manifest → tensors + vocab)

### Goal

Parse **only** `AndroidManifest.xml` per APK; build frozen vocabularies from **train split**; write sharded feature tensors so P5 never touches APKs.

### 2.1 Manifest parsing (`src/features/manifest_decode.py`)

**Input:** APK path  
**Output:** structured lists:

- `permissions: List[str]` — from `<uses-permission android:name="...">` (and `uses-permission-sdk-23` if present in corpus — document choice)
- `receiver_actions: List[str]` — union of all `<action android:name="...">` under any `<receiver>…<intent-filter>…`

**Parser steps (per APK):**

1. Open APK as ZIP; read `AndroidManifest.xml` bytes.
2. Decode binary XML → element tree (backend from config).
3. Traverse XML:
   - **Permissions:** every `uses-permission` / optional `uses-permission-sdk-23` → collect `android:name`.
   - **Receivers:** every `receiver` element → for each child `intent-filter` → each `action` → collect `android:name`.
4. Normalize names if configured (e.g. `android.permission.SEND_SMS` → consistent full name).
5. Deduplicate within APK (set semantics): multiple identical actions still → single bit 1.

**Explicit exclusions (paper alignment):**

- No `classes.dex`, no API calls, no activities/services/providers unless needed for debugging only.
- No dynamically registered receivers (bytecode) — static manifest only.
- Optional: ignore `android:permission` on `<receiver>` for feature bits (paper focuses on actions).

### 2.2 Vocabulary construction (`src/features/vocab.py`)

**Run once on train-split APKs only:**

1. Stream all `split=train` rows from `apk_index.csv`.
2. Count document frequency for each permission string → \(\mathcal{P}\).
3. Count document frequency for each receiver action string → \(\mathcal{A}\).
4. Filter by `permission_min_doc_freq`, `receiver_action_min_doc_freq`.
5. Sort vocabularies lexicographically (stable index mapping).
6. Save:
   - `artifacts/processed/permission_vocab.json` → `{ "tokens": [...], "size": P }`
   - `artifacts/processed/receiver_action_vocab.json` → `{ "tokens": [...], "size": R }`
   - `artifacts/processed/feature_layout.json` → `{ "P": P, "R": R, "order": ["permissions", "receivers"] }`

### 2.3 Vectorization (`src/features/vectorize.py`)

For APK \(i\):

\[
p_{i,j} = \begin{cases} 1 & \text{if } \text{perm}_j \in \text{Manifest}(i) \\ 0 & \text{otherwise} \end{cases}, \quad
r_{i,k} = \begin{cases} 1 & \text{if } \text{action}_k \in \text{Receivers}(i) \\ 0 & \text{otherwise} \end{cases}
\]

\[
\mathbf{x}_i = [\mathbf{p}_i \,\|\, \mathbf{r}_i] \in \{0,1\}^{P+R}
\]

Store as **`float32`** in PyTorch/ONNX (0.0/1.0) for runtime compatibility.

### 2.4 Batch preprocess (`src/preprocessing/preprocess_apks.py`)

1. Load vocabs (must exist).
2. For each APK in index (train + test):
   - Extract → vectorize → append to shard buffers.
   - On parse failure → `artifacts/failed_apks.log` (path, reason).
3. Write sharded outputs under `artifacts/processed/`:

| File | Contents |
|------|----------|
| `features_train.pt` | `x` `[N_train, P+R]`, `y`, `paths`, `sha256` |
| `features_test.pt` | Same for test split |
| `features_val.pt` | Optional: 10% holdout from train for early stopping |
| Or shard pattern | `train_shard_000.pt`, … for memory |

4. Emit preprocessing metadata: `preprocessing_version` (date/git hash), vocab sizes, counts failed.

### Exit criteria

- [ ] \(P\) and \(R\) documented in log (expect ~150–300 and ~50–100 respectively on full corpus — rough plan ranges)
- [ ] Spot-check 5 APKs manually: permission/action sets match `aapt2` or apktool dump
- [ ] Train APK count + test APK count match P1 minus failures
- [ ] No test APK influenced vocab token list

---

## Phase P3 — DataLoaders

### Goal

PyTorch `Dataset` + `DataLoader` reading **only** `artifacts/processed/*.pt`.

### Tasks

1. `ManifestBoWDataset`: `__getitem__` returns `(x.float32, y.long)` shape `[P+R]`, scalar label.
2. `build_dataloaders()`:
   - `train_loader` → train (+ optionally merge val for final train — document policy)
   - `val_loader` → 10% holdout from train years
   - `test_loader` → 2022+2023 only
3. `batch_size` from config; `num_workers` per machine; `pin_memory` if GPU.

### Exit criteria

- [ ] One batch smoke test: shapes `[B, P+R]`, labels in `{0,1}`
- [ ] Class balance stats printed per split

---

## Phase P4 — Model definition

### Goal

Define deployment architecture (ONNX-exportable) and optional paper baseline hook.

### 4.1 Deployment: logistic regression (recommended)

\[
\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b), \quad \sigma(z) = \frac{1}{1+e^{-z}}
\]

**PyTorch module:** `nn.Linear(P+R, 1)` + `BCEWithLogitsLoss` (or sigmoid in forward for prob export).

**Why:** Matches sparse linear separability of BoW; smallest ONNX; aligns with “ONNX-friendly” note in rough plan.

### 4.2 Deployment alternative: tiny MLP

```
Linear(P+R → 64) → ReLU → Dropout(0.2) → Linear(64 → 1) → Sigmoid
```

Use only if logistic underperforms SVM baseline by &gt;2% F1 on val.

### 4.3 Paper baseline: sklearn SVM-RBF (offline only)

\[
f(\mathbf{x}) = \mathrm{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right), \quad
K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x}-\mathbf{x}'\|_2^2)
\]

- `GridSearchCV` or manual grid on **train holdout** for `C`, `gamma`.
- Save `artifacts/checkpoints/svm_rbf.joblib` + `svm_metrics.json`.
- **Not exported to ONNX** unless you add sklearn-onnx (out of scope unless requested).

### Exit criteria

- [ ] Forward pass on dummy `x` shape `[1, P+R]` works
- [ ] Parameter count logged (expect &lt; few × (P+R) weights — well under 50 KB ONNX)

---

## Phase P5 — Training

### Goal

Train deployment model on **train** split; monitor **val** holdout; save best checkpoint.

### Tasks

1. **Logistic / MLP (PyTorch):**
   - Optimizer: AdamW (config lr, weight_decay).
   - Loss: `BCEWithLogitsLoss(pos_weight=…)`.
   - Metrics per epoch: train/val loss, val F1, val AUC.
   - Early stopping on val F1 (patience from config).
   - Save `artifacts/checkpoints/best.pt` with `model_state`, `P`, `R`, `config_hash`.

2. **SVM baseline (parallel script `train_svm_baseline.py`):**
   - Fit on full train features (numpy); evaluate on val + test.
   - Store metrics for thesis “paper faithful” table.

3. **Ablations (required for paper narrative):**
   - Train/eval **permissions-only** \(\mathbf{p}\) (first \(P\) dims).
   - Train/eval **receivers-only** \(\mathbf{r}\) (last \(R\) dims).
   - Train/eval **full** \(\mathbf{x}\).

### Exit criteria

- [ ] Full model val F1 ≥ max(perm-only, receiver-only) — qualitative match to paper Table narrative
- [ ] `best.pt` reloadable; training log saved
- [ ] SVM baseline metrics JSON exists if `paper_baseline_svm: true`

---

## Phase P6 — Evaluation (test split: 2022 + 2023)

### Goal

Report **final** thesis metrics on temporal test set only.

### Metrics

- Accuracy, F1 (malware positive class), ROC-AUC
- Confusion matrix TN, FP, FN, TP
- Threshold: default 0.5; optional tuned threshold on **val only** → save to `thresholds.json`

### Outputs

`artifacts/metrics/test_results.json`:

```json
{
  "model_id": "broadcast_receiver_perm",
  "split": "test",
  "train_years": [2020, 2021],
  "test_years": [2022, 2023],
  "n_samples": 0,
  "feature_dims": { "P": 0, "R": 0, "total": 0 },
  "metrics": { "accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0 },
  "confusion_matrix": [[0, 0], [0, 0]],
  "threshold": 0.5,
  "ablations": {
    "permissions_only": { "f1": 0.0 },
    "receivers_only": { "f1": 0.0 },
    "full_fusion": { "f1": 0.0 }
  },
  "paper_svm_baseline": { "f1": 0.0, "note": "optional" }
}
```

### Exit criteria

- [ ] Evaluation script never reads APKs — only `features_test.pt`
- [ ] Ablation table included
- [ ] Confusion matrix plot saved optional (`artifacts/metrics/confusion.png`)

---

## Phase P7 — ONNX export bundle

### Goal

Produce Android-ready bundle per `Pipeline_full_concept.html`.

### Directory

`artifacts/export/broadcast_receiver_perm/`

### Tasks

1. Load `best.pt`; trace `model.eval()` with example input `[1, P+R]`.
2. Export `model.onnx`, opset 14, input name `features`, output `malware_prob` (sigmoid applied in graph **or** document logits + Java sigmoid — pick one, document in manifest).
3. Copy vocabs → `features/permission_vocab.json`, `features/receiver_action_vocab.json`, `features/feature_layout.json`.
4. Write `thresholds.json`: `{ "default": 0.5, "tuned_val": <float> }`.
5. Write `export_manifest.json`:

```json
{
  "model_id": "broadcast_receiver_perm",
  "domain": "manifest_perm_receiver_actions",
  "opset": 14,
  "inputs": [{ "name": "features", "shape": [1, "P_PLUS_R"], "dtype": "float32" }],
  "outputs": [{ "name": "malware_prob", "dtype": "float32" }],
  "preprocessing_version": "<date-or-git>",
  "multidex_mode": "n/a",
  "feature_extraction": {
    "apk_part": "AndroidManifest.xml",
    "fusion": "early_concat",
    "permission_dims": "P",
    "receiver_action_dims": "R"
  }
}
```

6. Generate `parity_samples/` (~10 APKs):
   - For each: raw `x.npy` or JSON list, `expected_prob.json` from PyTorch reference.

### Exit criteria

- [ ] ONNX model size &lt; 50 KB (logistic) or justified if MLP
- [ ] Bundle copies cleanly to `vigidroid/app/src/main/assets/models/broadcast_receiver_perm/`
- [ ] `export_manifest.json` validates on target minSdk device (smoke test)

---

## Phase P8 — Parity (Python PyTorch vs ONNX)

### Goal

Max absolute difference ≤ `1e-4` on all `parity_samples/`.

### Script

`src/training/parity_onnx.py`:

1. Load PyTorch checkpoint and ONNX session.
2. For each sample in `parity_samples/`:
   - Run both → compare `malware_prob`.
3. Write `artifacts/metrics/parity_report.json` with per-sample delta and `max_delta`.

### Exit criteria

- [ ] All samples pass threshold
- [ ] If fail: fix dtype (float32), row-major layout, or sigmoid placement before A1

---

## Android phases A1–A4

### A1 — Feature extractor (Java)

**Class:** `BroadcastReceiverPermExtractor` (name may follow VigiDroid conventions)

**Semantics must match Python P2:**

1. Open APK ZIP → `AndroidManifest.xml`.
2. Use `AxmlReader` (or equivalent) to collect permission names and receiver actions.
3. Load `permission_vocab.json` + `receiver_action_vocab.json` from assets.
4. Build `float[]` length `P+R` (0.0/1.0).
5. Split timings: `parse_ms`, `vectorize_ms`.

**Deliverable:** Unit test on 3 APKs comparing Java vector to Python dump (pre-parity).

### A2 — ONNX inference

1. `ModelRegistry` entry for `broadcast_receiver_perm`.
2. Load `model.onnx` + manifest; create ORT session.
3. Input tensor `[1, P+R]` float32; read `malware_prob`.
4. Apply threshold from `thresholds.json`.

### A3 — Scan orchestration integration

Append to per-scan `stages[]`:

```json
{
  "domain": "manifest_perm_receiver_actions",
  "model_id": "broadcast_receiver_perm",
  "parse_ms": 0,
  "vectorize_ms": 0,
  "inference_ms": 0,
  "score": 0.0,
  "mem_delta_bytes": 0
}
```

**Suggested cascade position (thesis):** early manifest stage — cheap, complementary to D1 MH1M and D3 Dex header (confirm with user).

### A4 — Instrumented parity test

- AndroidTest or JVM test: load each `parity_samples/` input → compare score to `expected_prob` within `1e-4`.
- CI gate before release build.

### A1–A4 exit criteria

- [ ] End-to-end scan of parity APK on device/emulator passes
- [ ] p50 `parse_ms + vectorize_ms + inference_ms` logged for thesis Task 1 / Task 5

---

## 4. Execution order and gates

```
P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → A1 → A2 → A3 → A4
         ↑ vocab frozen here — do NOT start A1 before P2 on full train set
```

| Gate | Rule |
|------|------|
| G1 | P7 only after P5 produces `best.pt` |
| G2 | A1 only after `features/*.json` exist in export bundle |
| G3 | P8 green before copying bundle to VigiDroid |
| G4 | A4 green before thesis on-device numbers |

---

## 5. Thesis experiment hooks

| Thesis task | How this model contributes |
|-------------|----------------------------|
| **Task 1** — Resource optimization | Log manifest parse + vectorize + infer ms; model &lt;50 KB |
| **Task 2** — Multistep | Use as fast manifest gate; define `t_low`, `t_high` on val |
| **Task 5** — Tradeoffs | Ablation: perm-only vs receiver-only vs fused; F1 vs latency plot |
| **Task 6** — Feasibility | Manifest-only, no Dex — high deployability score |
| **Ensemble (later)** | Export calibrated score; weight learned offline — do not block single-model E2E |

### Comparison targets (document in thesis)

| Config | Expected role |
|--------|----------------|
| Permissions only | Baseline ~paper ~76% region (your split will differ) |
| Receivers only | Weaker alone but complementary |
| **Permissions + receivers** | Best single manifest Mohsen configuration |
| vs D1 MH1M | Different feature space (2500-d XGB vs ~280-d fused BoW) |

---

## 6. Checklists (copy per phase completion)

### P2 feature parity checklist (Python internal)

- [ ] Same permission tags as paper (`uses-permission`)
- [ ] Receiver actions from static `<receiver>` only
- [ ] Set semantics within APK (duplicate tags → one bit)
- [ ] Vocab from train years only
- [ ] Unknown tokens ignored (or bucketed consistently)

### P8 / A4 parity checklist (train vs device)

- [ ] Same manifest bytes read
- [ ] Same vocab order → same index for each token
- [ ] float32 0.0/1.0 features
- [ ] Same sigmoid / logits handling
- [ ] Score delta ≤ 1e-4 on all parity samples

---

## 7. Estimated effort (planning aid)

| Phase | Effort (solo, familiar stack) |
|-------|--------------------------------|
| P0–P1 | 0.5–1 day |
| P2 | 2–4 days (parser edge cases) |
| P3–P4 | 0.5 day |
| P5–P6 | 1–2 days (+ GPU queue for full corpus) |
| P7–P8 | 0.5–1 day |
| A1–A4 | 2–3 days (Java parity often dominates) |

**Critical path:** manifest parser consistency (P2 ↔ A1).

---

## 8. Open decisions (see queries to user)

The following items are **not** finalized in this plan; defaults are proposed above. Resolve before P2/P5:

1. **Classifier for deployment:** logistic only vs tiny MLP if logistic fails validation?
2. **SVM paper baseline:** required for thesis table or skip to save time?
3. **Python manifest backend:** `axmlparserpy` vs `androguard` vs `aapt2` subprocess?
4. **Vocab filtering:** min document frequency and hard caps on \(P\), \(R\)?
5. **Permission name normalization** rules (short vs full name)?
6. **Training repo root:** new folder in [Thesis_repo_1](https://github.com/abd-faiyaz/Thesis_repo_1) vs this workspace?
7. **`apk_root` path** on the 500 GB training machine?
8. **Relationship to D1 MH1M:** replace, run in parallel, or ablation-only?
9. **Multistep policy:** slot in cascade (before/after ByteCNN / Dex header)?
10. **`uses-permission-sdk-23`:** include in permission vocabulary or not?

---

## 9. References

- Rough architecture: `rough_model_explanations/broadcast_receiver_permission_classifier.html`
- Pipeline guide: `Pipeline_full_concept.html`
- Thesis pipeline overview: `PIPELINE_IMPLEMENTATION_PLAN.md`
- Model catalog: `sendable/Source_papers/models_info.md` (listed as **Original**)
- VigiDroid app: https://github.com/Sakhawat238/vigidroid
- Training repo: https://github.com/abd-faiyaz/Thesis_repo_1

---

*Document version: 2026-06-04 · Plan id: `simple_1_brd_rec_perm`*
