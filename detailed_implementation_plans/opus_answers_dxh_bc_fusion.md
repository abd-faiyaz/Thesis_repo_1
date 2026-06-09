Both facts check out: the shared `system_actions.json` has **172 actions (API 21–36)** and is already on-device, and the deployed `mlp_header` uses `multidex_mode: sum`, `max_dex: 3`, `feature_dim: 104`, `hidden_dim: 128`, opset 14. Here are my answers.

---

## Q5 — Dex header min–max normalization → **B (reuse shipped BM1 `normalization_header.json`)**

The key fact that drives this: the deployed `mlp_header` (`dex_header_d3`) was itself fit on the **same 2020–2021 train split** this fusion model uses. So options A and B are *numerically almost identical anyway* — refitting on 2020–2021 reproduces (nearly) the same per-dimension min/max, modulo small differences in dedup/parse failures. Given that, **B wins on every tiebreaker**:

- **Best warm-start synergy:** the trunk you're loading was trained on BM1's stats, so feeding it BM1-normalized inputs means the trunk sees exactly the distribution it learned on — no input-distribution shift to fine-tune away.
- **Exact header-only parity:** the header-only ablation row can be the *already-measured* deployed BM1 number, with no retraining (see Q11).
- **One fewer artifact to fit/version**, and guaranteed consistency across the whole header family (BM1, cascade, this fusion model).

Avoid C (from-scratch) — you'd throw away a proven, free warm start for a tiny structural branch.

> One guard to add: if you ever change the fusion corpus/years away from BM1's, switch to A (refit) so the stats aren't stale.

## Q6 — `system_actions.json` → **A (reuse the 172-action list)**

Confirmed: `broadcast_mldp_hybrid/assets/system_actions.json` = 172 actions, API 21–36, already shipped on device. Reusing it gives **cross-model parity** (the broadcast hybrid and this fusion model index the same actions the same way) and eliminates allow-list drift between two receiver models. Rebuild (B) only risks divergence for no benefit.

## Q7 — Hyperparameters

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Header warm-start from deployed `mlp_header` trunk | **Yes** | Free, proven init; maximally effective under 5B (matched norm) |
| Fusion head | **MLP `(128+d_R)→64→1`** | Recovers cross-modal conjunctions ("odd header AND boot receiver") a bare linear can't; keep logistic only as fallback if it shows **no** val-F1 gain over MLP |
| Receiver embed `d_R` | **32** (override the plan's 64); sweep {32, 64} on val | `R` is sparse, low-rank, and weak alone (~71–79%); a 32-d embedding against a 128-d structural embedding keeps the head from letting the noisy receiver branch dominate/overfit |
| Multidex aggregation | **`sum`** | Matches deployed BM1 exactly (`multidex_mode: sum`, `max_dex: 3`); keep `primary_only` as the MSFDroid-faithful ablation |
| ONNX inputs | **Two named inputs** (`dex_header[1,104]`, `receiver[1,R]`) | Keeps the two-tower semantics explicit and avoids a brittle concat-offset contract between Python and Java |
| SVM paper baseline (RBF, C=10, γ=0.1 on `[H‖R]`) | **Yes, run offline** | It's the only paper-faithful classifier for the thesis fidelity table; cheap; not exported to ONNX |

## Q11 — Refit-norm × warm-start interaction → resolved by choosing 5B

The conflict you flagged only exists under **5A + warm-start**. By choosing **5B**, the trunk is fed exactly its training distribution, so:

- **Header-only ablation = the deployed BM1 number, no retraining.** Just cite the existing `mlp_header` test metrics as the "header only" row — it's apples-to-apples because the normalization and trunk are identical.
- One subtlety to document: once you **fine-tune end-to-end**, the fusion model's internal trunk weights drift from BM1, so the *fused* model's header pathway ≠ deployed BM1. That's expected and fine — the equality you rely on is only for the **ablation reference row**, which you take from the frozen deployed BM1, not from the fine-tuned fusion trunk.

If you ever do pick 5A instead: yes, it's acceptable for thesis reporting — the fusion model uses its own norm and deployed BM1 stays the independent reference — but then you must **retrain a header-only model on the refit norm** to get a comparable ablation number (you can't reuse BM1's), and label it clearly as "trunk fine-tuned on fusion norm." 5B avoids all of that.

---

These four choices (5B, 6A, the Q7 row, 5B-resolves-Q11) all reinforce each other around **maximum reuse + exact header parity**. 
