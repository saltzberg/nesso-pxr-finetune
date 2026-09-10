# MW controls: concise reporting design

**Status:** design only; no outcome computation. CPU-only fixed controls using saved native scores and matched saved neural predictions. No edits to primary reports, code, predictions or split manifests.

## Question and claim boundary

Does the native Nesso scalar add predictive information beyond molecular weight when mapped to cellular PXR induction pEC50, and how do these simple mappings compare with the saved head-only and LoRA systems?

Native Nesso is a binding-related model quantity, documented as log10(IC50/µM), not a cellular induction endpoint. Fitting any of these regressions to cellular pEC50 is an **induction mapping**, not independent validation of binding affinity or mechanism. Improvement beyond MW supports incremental prediction on these data; it does not establish direct binding, causal size effects, or faithful binding-to-induction transfer. Null results do not disprove binding signal.

**Visible report caveat:** “Fixed CPU controls on reused development folds; not a new untouched holdout.” These controls follow prior development results, even though their recipes are now fixed. No prospective-confirmation or automatic-promotion claim.

## Locked comparison

Let `m` be molecular weight in Da from the approved saved MW source and `s` the frozen native Nesso scalar. Record its exact column, direction and transformation; do not mix raw native and `6 - native` conventions. Response is the existing cellular pEC50 label.

| Fixed WLS model | Point prediction |
|---|---|
| MW only | b0 + b1 m |
| Native score only, fitted mapping | b0 + b1 s |
| Score + MW | b0 + b1 s + b2 m |
| Score + MW + interaction | b0 + b1 s + b2 m + b3 s m |

All models include an intercept and unrestricted coefficients. Minimize sum(w × squared residual), with inverse-SE weights `w = 1 / max(SE, 0.1)` as inherited from the comparison protocol—not inverse variance. Any constant weight normalization leaves WLS unchanged. No tuning, regularization search, model selection, clipping, new thresholds, alternate draws or seeds. If centering/scaling is needed for numerical stability, learn it on FIT only and preserve the exact model span; report rank deficiency rather than silently changing the recipe.

Reuse the exact five strict chemical folds, draw 0, and existing N100/N500 membership: N100 = 80 FIT + 20 CAL; N500 = 400 FIT + 100 CAL. Preserve inherited test membership and purge exclusions without regenerating splits. FIT labels alone estimate coefficients. CAL remains reserved for the existing residual-interval role, never point fitting, feature selection or tuning; interval reporting is optional and must retain the original procedure if included. Outer-test labels are evaluation-only.

Use saved `head_only` and `head_plus_lora` predictions unchanged, with exact fold/draw/budget/compound joins. The fitted score-only control must not be mislabeled as the unadapted native predictor. No neural rerun or fresh Nesso inference is required.

## One-page report

1. **Compact metric table:** separate N100 and N500 blocks, fixed row order of the four controls followed by saved head-only and LoRA. Columns grouped as **overall / MW <215 Da / MW >215 Da**, each showing inverse-SE weighted MAE (primary), ordinary MAE and Spearman rho. Show eligible compound counts and missingness by subset; use matched rows across methods, not silently changing denominators. MW exactly 215 belongs in overall only, with its count explicitly noted; missing MW remains a coverage issue, not an implicit high-MW assignment. Undefined rho is `NA` with reason.
2. **Paired fold display:** small multiples with columns N100/N500 and rows overall/<215/>215; plot fold-level weighted-MAE differences from saved head-only, with a zero reference and identical difference scales. Negative means lower error. Show individual folds, not confidence bars implying independent repeated observations. Keep fixed model order, quiet grayscale, sparse axes and labels outside the data field; no decorative cards or repeated legends. Put exact values in the table/CSV rather than overprinting points.
3. **Short interpretation:** distinguish (a) score+MW versus MW-only, (b) score+MW versus score-only, (c) interaction versus additive, and (d) each fixed control versus saved head-only/LoRA. Describe overall and size-stratum trade-offs and fold consistency, including null or adverse results. Do not select a winner on the best subset or budget.

Compute pooled outer-test metrics separately within each budget, preserving one prediction per compound when the inherited fold contract provides that. Weighted MAE = sum(w × |prediction − truth|) / sum(w); MAE and Spearman rho are unweighted. Also retain fold-level metrics and paired deltas; pooled rho is not mean fold rho. Never pool N100 and N500 as independent molecules or present the reused folds as independent replicates. Small-subset counts must accompany interpretation.

## Audit sidecars and publication gate

Keep proposed `metrics.csv`, `paired_deltas.csv`, `predictions.csv` and `manifest.json` under this analysis directory. Bind source paths/hashes for native score, MW, labels/SE, split/role manifests and saved neural predictions; record feature units, MW chemistry convention, weight formula, coefficients, coverage and failures. Preserve full declared-universe coverage beside any common-row comparison; unmatched comparator cells are unavailable, not a reason to silently refit or substitute.

Before a results report: reconcile every fold × budget × method × subset cell; assert exact FIT/CAL/test role parity and disjointness, saved-comparator parity, unique prediction keys, threshold counts and metric replay. Inspect the final figure for collisions and shared-scale clipping. No numerical outcome or successful verification is claimed by this design.

Protocol basis: `PXR/experiments/20260906_affinity_comparison/README.md`, especially the native endpoint definition, fixed split roles, inverse-SE floor and development-fold caveat. Exact artifact resolution and execution checks remain implementation responsibilities.
