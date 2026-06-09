# Model feasibility rankings (high / medium / low)

**Method:** Composite score over the 10 registry models using offline test metrics (Accuracy, F1, ROC-AUC), Scan A on-device cost (median `stage_total_ms` + memory), and **cascade tier** from `cascade_tier_spec.json`.

```
composite = 0.45 × quality_norm + 0.30 × tier_weight + 0.25 × (1 − cost_norm)
quality_norm = min–max( (acc + f1 + roc_auc) / 3 )
cost_norm    = min–max( stage_total_ms + 10 × mem_mb )
```

**Tier weights (deployed cascade):** tier 1 = 1.00, tier 2 = 0.85, tier 3 = 0.65, tier 4 = 0.50. `mlp_header` = 0.80 (tier-2 fallback). Models **not** in the cascade policy = 0.25 (ablation-only).

**Buckets:** ranks 1–3 → **high**, 4–7 → **medium**, 8–10 → **low**.

---

## Rankings (POCO Scan A n=1453 + offline test n=764)

| Rank | Model | `model_id` | Tier | Composite | Feasibility | Reasoning |
|------|-------|------------|------|-----------|-------------|-----------|
| 1 | **MLP-H** | `mlp_header` | 2 fb | 0.903 | **high** | Top accuracy (0.97/0.95), fastest stage (~0.4 ms), tier-2 cascade fallback |
| 2 | **MLDP+Dex Cascade** | `mldp_dexheader_cascade` | 2 | 0.803 | **high** | Strong F1/AUC (0.93/0.95), sub-ms latency, core tier-2 deployed model |
| 3 | **Dex+Broadcast Fusion** | `dexheader_broadcast_fusion` | — | 0.774 | **high** | Highest offline scores (0.97/0.95/0.98) and ~0.5 ms stage time; not cascaded but excellent ablation profile |
| 4 | **Dex+Manifest Dual** | `dual_branch_dex_manifest` | — | 0.750 | **medium** | Near-top accuracy (0.97/0.95/0.98); ~8 ms/stage, ablation-only |
| 5 | **Broadcast+MLDP** | `broadcast_mldp_hybrid` | 1 | 0.686 | **medium** | Tier-1 gate with receiver context; good speed, moderate AUC (0.94) |
| 6 | **Dex+Manifest ASCNN** | `early_fusion_dex_manifest` | 3 | 0.679 | **medium** | Deployed at tier 3; solid AUC (0.97) but ~8 ms and lower F1 than dual-branch |
| 7 | **MLDP-pruned** | `mldp_pruned_permission` | 1 | 0.576 | **medium** | Cheapest tier-1 filter; weakest AUC (0.90) among permission models |
| 8 | **XGBoost** | `manifest_xgb` | 3 | 0.451 | **low** | Best AUC (0.99) but **~100 ms** stage time — dominates Scan A battery (~0.48%) |
| 9 | **1D-CNN** | `bytecnn` | 4 | 0.397 | **low** | Tier-4 backstop; lowest F1 (0.85) despite fast inference |
| 10 | **LinRegDroid** | `linregdroid_permission` | — | 0.324 | **low** | Lightweight permission baseline; lowest ROC-AUC (0.93), not cascaded |

---

## Cascade tier reference (`cascade_tier_spec.json`)

| Tier | Models | Role |
|------|--------|------|
| 1 | `mldp_pruned_permission`, `broadcast_mldp_hybrid` | Cheap permission filters; OR-aggregation for conservative malware call |
| 2 | `mldp_dexheader_cascade_mode_b` (+ `mlp_header` fallback) | Structural mid-tier before heavy manifest/byte |
| 3 | `early_fusion_dex_manifest`, `manifest_xgb` | High-recall manifest + MSFDroid fused header+BoW |
| 4 | `bytecnn` | Raw-byte final tier |

**Not in cascade:** `dual_branch_dex_manifest`, `linregdroid_permission`, `dexheader_broadcast_fusion` — Scan A ablation only.

---

## Battery note

Per-APK `capacity_pct_delta` was 0 on POCO (integer % stuck at 99). Session drain (~0.57% total) is derived from `charge_counter_uah_used` and allocated to each model by **aggregate stage-time share** across the full Scan A run. XGBoost receives the largest share (~0.48%) because it dominates total stage milliseconds; tier-1 MLPs receive &lt;0.01% each.
