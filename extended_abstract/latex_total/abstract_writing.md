# Abstract writing plan — `temp_mod.tex`

## Hard constraints (from user)

| # | Decision |
|---|----------|
| 1 | **Page 1:** Abstract + Introduction + Background + Proposed Methodology (Experiments → page 2) |
| 2 | **On-device narrative:** Four-tier cascade (`ScanOrchestrator`, `cascade_policy.json`, early exit), parse-once `FeatureContext`, JSONL metrics |
| 3 | **Models:** Summarize by tier; representative citations only (not all 10 by name) |
| 4 | **Offline pipeline:** Single paragraph for P0–P8 (flowcharts carry detail) |
| 5 | **Figures:** Embed TikZ from `proposed_methodology_flowchart/gemini2.tex` (Fig 1 offline, Fig 2 on-device) |
| 6 | **Intro hook:** Best model from `Illustrations_templates/On-Device ML-Experiments - Sheet1-generated.csv` |
| 7 | **Scope:** Full IEEE doc — rewrite only the four front sections; copy Experiments, Conclusion, bib from `temp.tex` |
| 8 | **Limit:** IEEE **2-page** extended abstract total |

## Best-model fact (CSV row 11)

**Dex+Broadcast Fusion** — Acc **0.9699**, F1 **0.9503**, ROC-AUC **0.9794**, mean stage time **0.5 ms**, feasibility **high**.

Use in Introduction as the peak accuracy / resource tradeoff example.

## Model naming (canonical)

- `early_fusion_dex_manifest` — Early-Fusion Dex+Manifest
- `dual_branch_dex_manifest` — Dual-Branch Dex+Manifest
- See `model_abstract_infos.md` for full table.

## Writing tasks

### Task A — Preamble
- [x] Copy `temp.tex` preamble; add `calc` + `balance` to TikZ/libs.
- [x] Keep title, authors, keywords unchanged.

### Task B — Abstract (~90–110 words)
- [x] Done in `temp_mod.tex`.

### Task C — Introduction (~100 words)
- [x] Dex+Broadcast Fusion hook: 97.0% acc, F1 0.950, 0.5 ms.

### Task D — Background (~90 words)
- [x] Done with representative citations.

### Task E — Proposed Methodology (~200 words + figures)
- [x] P0–P8 paragraph, tier model summary, cascade on-device narrative.
- [x] Figs 1–2 from `gemini2.tex` (height-limited inline, Early-Fusion / Dual-Branch labels).

### Task F — Page-1 layout tactics
- [x] Inline side-by-side figures at 1.55 in height; `\FloatBarrier` before Experiments.
- [x] **Compiled: 2 pages total** (IEEE limit met).

### Task G — Page 2 sections
- [x] Experiments: merged Table I (offline + device columns).
- [x] Conclusion shortened.
- [x] Acknowledgment dropped to save space (restore if venue allows).
- [x] Bibliography trimmed to cited entries only; `\footnotesize`.

### Space trade-offs (document in thesis notes)
- Merged two experiment tables into one `table*`.
- Removed placeholder Pareto figure (mentioned in prose).
- Removed uncited bib items (`msdroid`, `mh1m`, `onnx`).

## Compile check

```bash
cd extended_abstract/latex_total
pdflatex -interaction=nonstopmode temp_mod.tex
```

Target: methodology + figures end on page 1; Experiments begins page 2; total ≤ 2 pages.
