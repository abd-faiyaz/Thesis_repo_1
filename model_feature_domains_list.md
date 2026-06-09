## The 3 components (short)

In this codebase, a model’s **feature domain** is described as **three parts** (see `todo_model_ranks.html` column *“Architecture (feature domain — 3 elements)”*):

1. **Signal source** — which APK data is read (manifest, Dex header, raw bytes, etc.)
2. **Encoding** — how it becomes a fixed vector (BoW, binary flags, normalization, pruning)
3. **Classifier / fusion** — the model head (MLP, LR, CNN, XGBoost, cascade, late fusion)

At the APK level, signals also fall into **three broad families**: **manifest**, **structural Dex**, and **raw bytes** — hybrids combine two of these.

---

## All implemented models — feature domains

From `model_plot_registry.json`, `plotting_based_codebase_fix.md`, and on-device runners in `vigidroid/`.

| Model | Method | Domain ID | Feature domain (3 elements) | Tier |
|-------|--------|-----------|----------------------------|------|
| **manifest_xgb** | XGBoost | `manifest_xgb` | Manifest + Dex token scan (perms, intents, APIs) · 2500-d sparse BoW · XGBoost | Legacy / heavy |
| **bytecnn** | 1D-CNN | `bytecnn` | Last 1024 APK tail bytes · raw bytes 0–255 · 1D-CNN | LIGHT |
| **linregdroid_permission** | LinRegDroid | `manifest_permissions` | Full manifest permission vocab · per-permission binary vector · MLR + threshold | LIGHT |
| **mldp_pruned_permission** | MLDP-pruned | `manifest_permissions_mldp` | PRNR → SPR → PMAR permission selection · ~20–40 binary permission bits · tiny MLP | LIGHT |
| **broadcast_mldp_hybrid** | Broadcast+MLDP | `manifest_mldp_perm_receiver_actions` | MLDP-pruned permissions · broadcast receiver intent actions · fused tiny MLP (~80–120-d) | LIGHT |
| **mlp_header** | MLP(H) | `dex_header_d3` | `classes*.dex` header bytes 8–111, sum-pooled · 104-d min–max normalized · MLP header | MID |
| **early_fusion_dex_manifest** | Early-Fusion Dex+Manifest | `dex_header_manifest` | Dex header 104-d + manifest BoW (~4381-d) · early concat · ASCNN(C) single tower | MID |
| **dual_branch_dex_manifest** | Dual-Branch Dex+Manifest | `dex_header_manifest_dual` | Dex header 104-d + manifest BoW · dual-branch embeddings · late-fusion MLP | MID |
| **mldp_dexheader_cascade** | MLDP+Dex Cascade | `manifest_mldp_perm_dex_header` | MLDP ~30-d permissions · Dex header 104-d · cascade or fused tiny MLP (Mode A/B) | MID |
| **dexheader_broadcast_fusion** | Dex+Broadcast Fusion | `dexheader_broadcast_fusion`* | Dex header 104-d · broadcast receiver actions (~30–80-d) · late-fusion dual towers | MID |

\*On Android the runner domain is `dex_header_receiver_actions`; the registry/plotting ID is `dexheader_broadcast_fusion`.

---

**Count:** **10 model families** in the registry (11 on-device stage IDs if you count cascade Mode A and Mode B separately). **Not in the 11 suggested set** but deployed: legacy **XGBoost (D1)** and **ByteCNN (D2)**. **Not implemented** from the suggested 11: Broadcast+Permission (faithful), ANASTASIA-lite, ERBE MLP, FexDroid variants, ANASTASIA+BoW fusion.