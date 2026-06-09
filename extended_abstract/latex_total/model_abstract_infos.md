# Model names for extended abstract

Canonical source: `Shared_pipeline_Files/data/model_plot_registry.json` (10 models).  
Paper mapping: `todo_model_ranks.html` **(DONE!)** rows + already-implemented legacy models noted at the bottom of that file.

**Naming rules (confirmed):**
- All **10** plot-registry models included.
- **MLDP+DEX cascade** = one display name (Modes A/B not split in abstract prose).
- Short table labels like `temp.tex`; use **MLP(H)** (not MLP(H)/BM1).
- Former Pattern A/B → **Early-Fusion Dex+Manifest** / **Dual-Branch Dex+Manifest** (`model_id` below).

---

## Master table

| # | Short label (tables) | Display name (prose) | `model_id` | Feature domain (3 elements) | Paper reference(s) | PDF / source in `todo_model_ranks.html` |
|---|----------------------|----------------------|------------|----------------------------|--------------------|----------------------------------------|
| 1 | 1-D CNN | **ByteCNN** | `bytecnn` | Last 1024 APK tail bytes · raw byte values · 1-D CNN | Hasegawa & Iyatomi~\cite{bytecnn} | *Already implemented* — `56_One-dimensional convolutional neural networks for Android malware detection.pdf` |
| 2 | LinRegDroid | **LinRegDroid** | `linregdroid_permission` | Full manifest permission vocabulary · per-permission binary vector · multiple linear regression + threshold | Narayanan et al.~\cite{linregdroid} | **(DONE!)** — `55_LinRegDroid_Detection_of_Android_Malware_Using_Mul.pdf` |
| 3 | MLDP-pruned | **MLDP-pruned** | `mldp_pruned_permission` | PRNR-ranked permissions · support-filtered set · PMAR-selected ~20–40 binary bits · tiny MLP | Wang et al.~\cite{mldp} | **(DONE!)** — `7_Permission Extraction Framework for Android Malware Detection.pdf` |
| 4 | Broadcast+MLDP | **Broadcast+MLDP hybrid** | `broadcast_mldp_hybrid` | MLDP-pruned permission bits · broadcast-receiver intent actions · fused tiny MLP (~80–120-d) | Wang et al.~\cite{mldp}; Mohsen & Ismail~\cite{broadcast} | **(DONE!)** — `7_…MLDP….pdf` + `12_Detecting Android Malwares By Mining Statically Registered Broadcast Receivers (Full paper).pdf` |
| 5 | MLP(H) | **MLP(H)** | `mlp_header` | `classes*.dex` header bytes 8–111, sum-pooled · 104-d normalized · MLP header classifier | Li et al.~\cite{msfdroid} | *Already implemented* — MSFDroid (no PDF in workspace; see implementation docs) |
| 6 | Early-Fusion Dex+Manifest | **Early-Fusion Dex+Manifest** | `early_fusion_dex_manifest` | Dex header 104-d + manifest BoW (~4381-d) · early concat · single ASCNN(C) tower | Li et al.~\cite{msfdroid} | *Already implemented* — MSFDroid early fusion (ASCNN C) |
| 7 | Dual-Branch Dex+Manifest | **Dual-Branch Dex+Manifest** | `dual_branch_dex_manifest` | Dex header 104-d + manifest BoW · separate branch embeddings · late-fusion MLP | Li et al.~\cite{msfdroid} | *Already implemented* — MSFDroid late fusion |
| 8 | MLDP+DEX cascade | **MLDP+DEX cascade** | `mldp_dexheader_cascade` | MLDP-selected permissions (~30-d) · normalized Dex header 104-d · cascade gate or fused tiny MLP | Wang et al.~\cite{mldp}; Li et al.~\cite{msfdroid} | **(DONE!)** — `7_…MLDP….pdf` + MSFDroid (header branch) |
| 9 | XGBoost | **XGBoost** | `manifest_xgb` | Manifest + all-DEX token scan (perms, intents, APIs) · 2500-d sparse OR vector · gradient-boosted trees | Drebin-style static features~\cite{drebin} *(legacy pre-thesis baseline)* | *Already implemented* — legacy D1; not in suggested-11 table |
| 10 | Dex+Broadcast | **DEX+Broadcast fusion** | `dexheader_broadcast_fusion` | Dex header 104-d · broadcast-receiver intent actions (~30–80-d) · late-fusion dual towers | Li et al.~\cite{msfdroid}; Mohsen & Ismail~\cite{broadcast} | **(DONE!)** — MSFDroid (header) + `12_…Broadcast Receivers….pdf` |

---

## Cascade tier mapping (on-device, for methodology text)

From `how_vigidroid_works.html` — tiers describe **runtime cost**, not the old LIGHT/MID/HEAVY training groups.

| Cascade tier | Models in live cascade | Ablation-only (same family) |
|--------------|------------------------|----------------------------|
| 1 — manifest gates | MLDP-pruned, Broadcast+MLDP hybrid | LinRegDroid |
| 2 — Dex header | MLDP+DEX cascade | MLP(H) fallback inside Mode B |
| 3 — heavy structural | Early-Fusion Dex+Manifest, XGBoost | Dual-Branch Dex+Manifest |
| 4 — byte tail | ByteCNN (fused with tier-3 pool) | — |
| — | — | DEX+Broadcast fusion |

---

## Bib keys used above (match `temp.tex`)

| Key | Citation |
|-----|----------|
| `bytecnn` | C. Hasegawa and H. Iyatomi, ICCE 2020 |
| `linregdroid` | A. Narayanan et al., *Computers & Security*, 2018 |
| `mldp` | S. Wang et al., *IEEE Access*, 2019 |
| `broadcast` | M. Mohsen and A. Ismail, *IJACSA*, 2017 |
| `msfdroid` | Y. Li et al., *IEEE TIFS*, 2022 |
| `drebin` | D. Arp et al., NDSS 2014 |

---

## Confirmed naming summary (for `temp_mod.tex`)

1. **All 10** plot-registry models — yes.  
2. **MLDP+DEX cascade** — single name (no Mode A/B in abstract prose).  
3. **Short labels** — yes; **MLP(H)** without “/BM1”.  
4. **Early-Fusion Dex+Manifest** (`early_fusion_dex_manifest`); **Dual-Branch Dex+Manifest** (`dual_branch_dex_manifest`).

---

## Your 8 questions (please answer before `temp_mod.tex`)

1. **Page-1 scope:** Only **Proposed Methodology** on page 1, or **Abstract + Introduction + Background + Methodology** all on page 1?

2. **On-device architecture narrative:** Follow **current HTML** (four-tier cascade, `ScanOrchestrator`, JSONL), keep **Quick/Balanced/Full** from `temp.tex`, or **blend both**?

3. **Model detail level:** Name **all 10** in methodology text, or **summarize by cascade tier** with representative citations?

4. **P0–P8 pipeline:** **One paragraph** mention, or **short enumerated list** (P0 config → … → P8 parity ≤10⁻⁴)?

5. **Flowcharts in `temp_mod.tex`:** **Remove**, **comment out**, or **one prose sentence** (“figures omitted”)?

6. **Introduction placeholder** (`<mention experimental best result…>`): **Leave**, **fill with BM1 numbers** (Acc 0.966 / F1 0.944), or **remove** the sentence?

7. **`temp_mod.tex` scope:** **Full document** (only four sections rewritten) or **snippet only**?

8. **Length target:** Any **word/page limit** for the IEEE extended abstract?
