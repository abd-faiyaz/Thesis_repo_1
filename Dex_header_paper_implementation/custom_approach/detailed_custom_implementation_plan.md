# Detailed Custom Implementation Plan: Lightweight Header–Manifest Malware Detector (MSFDroid-Inspired)

This document refines the approach outlined in `gemini_proposed_architecture.md` using the original paper *A Lightweight Multi-Source Fast Android Malware Detection Model* (MSFDroid, Peng et al., *Appl. Sci.* 2022, [doi:10.3390/app12115394](https://doi.org/10.3390/app12115394)). The goal remains a practical **accuracy vs. on-device cost** trade-off for a Vigidroid-style static scanner: **drop the structural-entropy / MEM-PSD path** (Base Model 2, `MLP(M)`) to avoid 256-byte blocking, Shannon entropy over the full Dex stream, and Burg/Yule–Walker PSD—while preserving as much of the paper’s **proven** components as possible elsewhere.

**No implementation work is started here; this is a specification and rationale only, pending your confirmation.**

---

## 1. Rationale and Expected Trade-offs (Grounded in the Paper)

### 1.1 What MSFDroid Actually Combines

The paper uses **four** base models and **adaptive soft voting** (AdaSV):

| Component | Paper label | Role |
|-----------|-------------|------|
| Dex header only | `MLP(H)` | Fast structural meta-features from `DexHeader` (sizes/offsets, normalized). |
| Dex entropy → PSD | `MLP(M)` | 256-byte blocks → Shannon entropy sequence → max-entropy PSD → length-128 vector → MLP. |
| Manifest BoW | `ASCNN(I)` | Permissions + intent keywords → sparse BoW → 3-layer ASCNN → MLP. |
| Header + manifest concat | `ASCNN(C)` | Concatenate header features and BoW → ASCNN → MLP (paper Fig. 3, §3.2). |

**Important:** In §3.2, Base Model 4 explicitly **concatenates Dex header features with permission/intent features** and feeds that to ASCNN + MLP—not “header + entropy.” The abbreviation table’s definition of `C` as including entropy PSD is inconsistent with §3.2 and Fig. 3; **implementation should follow §3.2 / Fig. 3** (header + manifest BoW into `ASCNN(C)`).

### 1.2 Why Removing `MLP(M)` Hurts—and How the Plan Mitigates It

From the paper’s Table 6 (indicative):

- `MLP(H)` alone: **~83.5% ACC** (weak in isolation).
- `MLP(M)` alone: **~89% ACC** (stronger single stream than header-only).
- `ASCNN(I)` alone: **~89% ACC**.
- Combining streams recovers a high ceiling; the **full four-model AdaSV** reaches **~97.3% ACC / ~99.5% AUC** on their setup.

**Conclusion:** Dropping MEM-PSD removes a high-value signal. The custom model must **lean on manifest semantics + header structure + careful fusion and training**, and you should **budget ablation studies** (header-only vs manifest-only vs fused) to measure the gap on *your* datasets and APK pipeline.

**Mitigations in this plan:**

1. Treat **manifest + header fusion** as first-class (paper’s `ASCNN(C)` idea), not an afterthought.
2. Keep the paper’s **ASCNN** (adaptive shrinkage convolution) for high-dimensional sparse BoW—its role is dimensionality reduction and denoising (§3.5–3.6, Table 5).
3. Optionally add a **lightweight risk check** (not full MEM-PSD): e.g. **file size / header sanity / multi-Dex flags** as extra scalar inputs to the header branch—cheap and aligned with “structural” signal without scanning the whole Dex as entropy (optional ablation).
4. Use **quantization** and measured **on-device latency** as first-class metrics (paper compares to Ruitao et al.’s mobile-oriented work; Fig. 11, Tables 2–3).

---

## 2. Target Architecture: “MSFDroid-Lite” (Two-Stream + Single Head)

Two viable fusion patterns—**prefer Pattern A** unless ablations show Pattern B is better.

### Pattern A (Closest to Paper Base Model 4) — **Recommended**

**Single tower:** concatenate normalized **Dex header feature vector** and **BoW vector** (dimension `N+1` with UNK), then:

`[H || BoW]` → **3× Adaptive Shrinkage Convolution (ASCNN)** as in Fig. 7 → **MLP classifier** (FC + BN + ReLU blocks per paper style) → **sigmoid** (malware probability).

- **Fig. 7 (paper):** Input width tied to lexicon size (e.g. 1×4380 → three ASU blocks with kernel 3, strides 2, 2, 1 → AvgPool1d → **128-dim** embedding) → then MLP head.
- **ASCNN** preserves the paper’s **dynamic convolution + soft thresholding** (§3.6, Fig. 8) for noise-heavy sparse features.

This directly implements **`ASCNN(C)`** without a separate `ASCNN(I)` tower—smaller and faster than a four-branch ensemble.

### Pattern B (Gemini-Style Late Fusion) — **Ablation / Alternative**

- **Branch 1:** `MLP(H)` on header features only (FC + BN + ReLU ×2 as in Fig. 3).
- **Branch 2:** `BoW` → **ASCNN(I)** → embedding.
- **Fusion:** concatenate branch embeddings → small MLP → sigmoid.

**When to use:** If you want interpretability (“header score vs manifest score”) or staged deployment (e.g. run cheap header MLP first). **Cost:** more parameters and forward passes than Pattern A.

**Recommendation:** Start with **Pattern A**; add Pattern B as an ablation row in the thesis.

---

## 3. Feature Extraction Pipeline (Static, APK-Local)

### 3.1 Dex Header (`H`) — Align with §3.3

- **Input:** `classes.dex` (primary Dex; see §3.4 for multi-Dex).
- **Steps:**
  1. Verify **magic** first 8 bytes (`dex\n035\0` or version variants per `dalvik/libdex/DexFile.h`).
  2. Parse **`DexHeader`**: offsets/sizes for `string_ids`, `type_ids`, `proto_ids`, `field_ids`, `method_ids`, `class_defs`, `data`, `link_*`, `map_off`, etc.
  3. Encode fields (paper: hexadecimal encoding then **normalization** into a **1-D “gray” feature vector**—replicate this normalization scheme for comparability).
- **Scope:** Only read the **header region** (fixed small byte window)—**no full-file scan**—keeping extraction O(1) in Dex size for this branch.

### 3.2 AndroidManifest — BoW (`I`) — Align with §3.5

- Decode binary XML to readable manifest (standard Android reverse pipeline).
- Extract **permission names** and **intent-related strings** (actions, categories, as in the paper’s “permission and intent keywords”).
- **Lexicon construction (training):**
  - Collect all keywords across the training set.
  - Drop keywords with frequency **&lt; 2** (paper).
  - Cap or rank to **N** keywords; paper uses **N = 4380** for dictionary size.
  - **UNK** bucket for out-of-vocabulary → **(N+1)-dimensional** BoW (binary or count—match the paper’s encoding; paper describes occurrence as independent probabilities → typically **multi-hot or frequency**; implement exactly one scheme and document it).
- **Inference:** fixed vocabulary from training; UNK for unknowns.

### 3.3 Omit MEM-PSD (`M`) — Explicit Non-Goals

Do **not** implement Algorithm 1 (256-byte blocks, Shannon entropy along full Dex) or Burg / Yule–Walker PSD for the **default** product path.

**Optional thesis-only appendix:** subsampled entropy (e.g. first K blocks only) as a **weak** cheap feature—only if ablations show benefit; not part of the default lightweight pipeline.

### 3.4 Multi-Dex and Edge Cases

- **Multiple `classes.dex`:** Define policy explicitly: e.g. use **primary** `classes.dex` only (matches common static tools), or **max-pool / merge** predictions across Dex files—document and keep consistent for train/test.
- **Missing or corrupt Dex/manifest:** define rejection vs benign-default vs abstain (important for real deployment metrics).

---

## 4. Network and Training Details (Paper-Aligned Defaults)

Use the paper’s setup as **starting hyperparameters**; tune on validation splits.

| Item | Paper (§4.2) | Notes for custom plan |
|------|----------------|------------------------|
| Framework | PyTorch 1.9 (historical) | Use current PyTorch LTS; document version. |
| Optimizer | SGD, momentum **0.9** | AdamW is a reasonable modern alternative—run one ablation. |
| Learning rate | **0.005** initial, decay ×0.5 | Cosine or StepLR; tune. |
| Batch size | **16** | Adjust if GPU memory allows; may affect BN. |
| Metrics | **AUC**, **ACC**, **F1** | Report all three; add **inference ms** and **model size**. |
| AdaSV | Full model uses **μ** in loss (§3.7) | **Pattern A (single ASCNN(C)):** AdaSV over **multiple heads** is **not** applicable as in the paper; use **standard BCE** on final logits, or **learned temperature / class weights** if imbalance. **Pattern B:** can implement **two-branch AdaSV** (two base probabilities) with simplified Eq. (7)–(10). |

**Regularization:** Dataset merging (CICMalDroid2020, CIC-InvesAndMal2019, Drebin) as in §4.1 risks duplication and domain shift—follow recent best practices on **deduplication** and **temporal / family-aware splits** if you need stronger claims (outside the paper but important for thesis rigor).

---

## 5. Deployment: Mobile-Lightweight Inference

### 5.1 Quantization

- Apply **post-training static INT8** (Gemini suggestion), with **calibration** on a representative APK feature set.
- Validate **AUC/ACC/F1** drop vs FP32; compare **model size (KB)** and **latency** (paper Fig. 11, Table 3 style).

### 5.2 Runtime Export

- **PyTorch Mobile** (TorchScript) **or** **ONNX → TensorFlow Lite** / ExecuTorch—choose based on your Android stack; plan should include **one** primary path and a fallback.
- **Extraction** stays in Java/Kotlin (ZIP, manifest parse, Dex header read); **inference** on the fused vector.

### 5.3 Profiling Budget

Mirror the paper’s breakdown where possible:

- **Extraction time:** unzip + manifest parse + header parse (no full Dex entropy).
- **Prediction time:** neural forward pass only.

Report **total** time per APK on representative devices (the paper uses several ARM devices; Table 3).

---

## 6. Evaluation Plan (Thesis-Grade)

1. **Offline:** Same metrics as paper (AUC, ACC, F1) on held-out test; confusion matrix; per-family error if labels exist.
2. **Ablations:**
   - `MLP(H)` only vs `ASCNN(I)` only vs **Pattern A** vs **Pattern B**.
   - With vs without optional lightweight extras (§1.2).
   - FP32 vs INT8.
3. **On-device:** Median/p95 latency; memory peak; APKs from a realistic size distribution.
4. **Comparison:** MSFDroid numbers from the paper are **not directly comparable** unless datasets and preprocessing match—phrase as **reference**, not as guaranteed reproduction.

---

## 7. Risks and Limitations (Explicit)

- **Static analysis** remains blind to reflection, dynamic loading, and server-side payloads—acknowledge as in §2.2 of the paper.
- **Obfuscation** of manifest or Dex may shift distributions; robustness testing is a separate work package.
- Removing `MLP(M)` may cap accuracy vs full MSFDroid; the thesis should report **your** measured gap honestly.

---

## 8. Suggested Implementation Phases (After You Confirm)

1. **Data layer:** APK → Dex header vector + manifest BoW builder; fixed vocabulary; train/val/test splits.
2. **Model:** Pattern A ASCNN(C) per §3.2 and Fig. 7–8 (reimplemented faithfully from the paper’s descriptions).
3. **Training:** BCE + threshold tuning; class imbalance handling if needed.
4. **Compression:** INT8 PTQ + accuracy check.
5. **Android:** Feature extraction in app; quantized model inference; end-to-end timing.

---

## 9. Summary

The **refined** plan keeps the paper’s **Dex header parsing**, **manifest BoW + ASCNN**, and **deployment-minded quantization**, but **removes MEM-PSD** for battery/CPU cost. It **prioritizes Pattern A (`ASCNN(C)`-style single tower)** over Gemini’s late-fusion sketch, documents **Pattern B** as an alternative, and adds **dataset, multi-Dex, evaluation, and risk** sections so the thesis can defend the sweet spot with measurable evidence.

**Awaiting your confirmation before any coding.**
