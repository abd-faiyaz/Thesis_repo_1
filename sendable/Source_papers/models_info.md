Here is a clean split of the **11 suggested models** (plus your **already-implemented** ones for context).

---

## 1) Original models (single paper, features + classifier aligned with the paper)

| Model | Paper file |
|--------|------------|
| **Broadcast Receiver + Permission** | `12_Detecting Android Malwares By Mining Statically Registered Broadcast Receivers (Full paper).pdf` |
| **LinRegDroid** | `55_LinRegDroid_Detection_of_Android_Malware_Using_Mul.pdf` |

*Faithful to the paper means: same feature domains and same family of classifier (e.g. SVM / linear regression + threshold). Swapping only the runtime format (e.g. ONNX export of a linear model) is still “original.”*

**Already implemented (original):**

| Model | Source |
|--------|--------|
| **MLP(H)** — Dex header only | MSFDroid (PDF not in folder; see `already_implemented_models/`) |
| **1DCNN (ByteCNN)** — last 1024 APK bytes | `56_One-dimensional convolutional neural networks for Android malware detection.pdf` |

---

## 2) Custom models (single paper, but changed for lightweight / ONNX / thesis)

| Model | Paper file | Main modifications |
|--------|------------|---------------------|
| **MLDP-Pruned Permission** | `7_Permission Extraction Framework for Android Malware Detection.pdf` | Same MLDP feature selection; classifier → tiny MLP / linear ONNX instead of SVM/trees |
| **ANASTASIA-lite** | `fereidooni2016.pdf` | 560-d → top 64–128 features; XGBoost / 6-layer DNN → small MLP |
| **FexDroid Category-Pruned Sparse** | `36_Effective and Explainable Detection of Android Malware Based on Machine Learning Algorithms.pdf` | Drebin/FexDroid categories capped (~3K–5K); LibLinear SVM → ASCNN / MLP |
| **ERBE Permission + Intent MLP** | `19_An extrinsic random-based ensemble approach for android malware detection.pdf` | Full 15-iteration ERBE dropped; single MLP on perm+intent vector only |

**Already implemented (custom):**

| Model | Source | Main modifications |
|--------|--------|---------------------|
| **Pattern A — ASCNN(C)** | MSFDroid | Drop `MLP(M)` / MEM-PSD; single fused header+BoW tower |
| **Pattern B — late fusion** | MSFDroid | Drop `MLP(M)`; dual-branch `MLP(H)` + `ASCNN(I)` |

---

## 3) Hybrid models (combine 2+ papers)

### 3a) Hybrid — **with modifications** (typical thesis path)

| Model | Paper files | What is combined | Modifications |
|--------|-------------|------------------|---------------|
| **Broadcast + MLDP Hybrid** | `12_Detecting...Broadcast Receivers (Full paper).pdf` + `7_Permission Extraction Framework...pdf` | MLDP-pruned permissions ∥ receiver actions | Fused ~80–120-d vector; tiny MLP (not in either paper) |
| **Dex Header + Broadcast Receiver Fusion** | MSFDroid (no PDF in folder) + `12_Detecting...Broadcast Receivers (Full paper).pdf` | 104-d Dex header ∥ receiver actions | Late fusion (Pattern B–style); not MSFDroid AdaSV / full ensemble |
| **ANASTASIA + Manifest BoW Fusion** | `fereidooni2016.pdf` + MSFDroid (no PDF in folder) | ANASTASIA-lite 128-d ∥ manifest BoW ~4381-d | Early concat + ASCNN(C); behavioral side already reduced |
| **FexDroid + Raw APK Tail Dual-Branch** | `36_Effective and Explainable...pdf` + `56_One-dimensional convolutional neural networks...pdf` | FexDroid-lite sparse ∥ 1024 tail bytes | FexDroid branch capped; late fusion head; two ONNX models |
| **MLDP + Dex Header Cascade** | `7_Permission Extraction Framework...pdf` + MSFDroid (no PDF in folder) | MLDP ~30-d ∥ Dex header 104-d | Cascade / tiny fused MLP; not paper’s standalone classifiers |

### 3b) Hybrid — **without modifications** (faithful features from each paper, simple fusion only)

| Model | Paper files | What is combined | Fusion (no architecture change) |
|--------|-------------|------------------|-----------------------------------|
| *(none in current suggested set)* | — | — | — |

You could treat these as **unmodified hybrids** if you build them explicitly:

| Possible build | Paper files | Idea |
|----------------|-------------|------|
| **Mohsen + LinRegDroid ensemble** | `12_Detecting...` + `55_LinRegDroid...` | Two manifest scores; average or vote (each model as in its paper) |
| **MLDP permissions + Mohsen receivers (paper classifiers)** | `7_...` + `12_...` | Each branch trained with paper’s SVM; fuse scores only |
| **Dex header MLP(H) + 1DCNN (sequential, no fusion change)** | MSFDroid + `56_One-dimensional...` | Run both paper models as-is; combine scores in ensemble (VigiDroid plan) |

Among the **11 suggested names**, every hybrid is **modified** because you intentionally change classifiers, dimensions, or fusion style for mobile deployment.

---

## Quick summary

| Category | Count (suggested 11) | Names |
|----------|----------------------|--------|
| **Original** | 2 | Broadcast+Permission (faithful), LinRegDroid |
| **Custom (single-paper)** | 4 | MLDP-pruned, ANASTASIA-lite, FexDroid-lite, ERBE MLP |
| **Hybrid + modified** | 5 | Broadcast+MLDP, Header+Broadcast, ANASTASIA+BoW, FexDroid+Tail, MLDP+Header |
| **Hybrid + unmodified** | 0 | (not in current list; optional if you fuse paper-faithful models only) |

**Deployed but not in the 11:** `mh1m_2500_rp_XGBoost` (D1) — separate external/MH1M-style pipeline, not from the eight PDFs above.