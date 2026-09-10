# Nesso affinity comparison: outcome-blind visualization design

Design recorded: 2026-09-07T21:56:22.662969+00:00.

**Question:** Does adding upstream rank-4 LoRA improve held-out prediction beyond fully adapting the continuous heads, separately at N100 and N500?

This is a reporting specification, not a results report. Only the training protocol and visualization/experiment-management guidance were read; no predictions, metrics, or model states were inspected. No fitting, inference, GPU access, calibration fitting, or bootstrap computation was performed. No illustrative numerical outcomes are supplied.

## 1. Frozen comparison and interpretation

Source: `PXR/experiments/20260906_affinity_comparison/train_protocol.json`.
SHA-256: `3b57c3775bf3f7a38b5f4a0b93ce45d8fcef629d0b9b1a3ede276f732c087fb8`.

- Fixed matrix: two arms × two budgets × five folds = 20 fitted prediction systems. Both native affinity members are part of each system, not independent replicates. One acquisition draw (`draw=0`) and training seed 42; no seed-variability estimate.
- `head_only`: fully train both `affinity_heads.to_affinity_pred_value` continuous heads; everything else remains frozen.
- `head_plus_lora`: the same fully trainable heads plus LoRA A/B at `esm_proj.1`, `esm_proj.3`, and `pairformer_stack.layers.0.transition_z.fc1`; rank 4, alpha 4. Do not label this whole-model fine-tuning or adaptation of the ESM encoder itself.
- Same fixed 40-epoch recipe, inverse-SE-weighted Huber training, no tuning, early stopping, or clipping. Evaluation MAE is not the Huber training loss.
- N is the total acquired target-label budget, **not** the test-set size: N100 = 80 FIT + 20 CAL; N500 = 400 FIT + 100 CAL. Outer held-out counts come from saved role manifests, not these numbers. Calibration labels count against N even if they only determine uncertainty intervals.
- Five reused chemically separated **development** folds, not a fresh untouched holdout. The protocol specifies `strict_chemical` and purge threshold 0.35; do not invent the fingerprint or the threshold's distance/similarity semantics. Validate those from the bound split manifest before describing them. The protocol's 3,344 development records are not an assumed scored denominator.
- Verify matched FIT/CAL/test identities across arms and test identities across budgets. Describe acquired-set nesting only if the actual manifests establish it; nesting does not imply fixed FIT/CAL roles.
- Native output conversion is `6 - arithmetic mean of both continuous native member values`. This IC50-equivalent conversion does not establish native cellular pEC50 endpoint support. Use the verified assay endpoint and units on axes; do not silently relabel native training biology.
- The contrast tests this fixed adapter recipe against this fixed head-only recipe. Existing CPU controls can contextualize it, but do not turn the comparison into an equivalence test against historically tuned heads or a claim about upstream pretraining generally.

## 2. Metric contract: pooled first, fold macro second

For held-out identity i, define `w_i = 1 / max(assay_SE_i, 0.1)` and absolute error `e_mi = abs(y_i - pred_mi)`. This is inverse **SE**, not inverse variance; weights derive from assay uncertainty, never model uncertainty or residual magnitude. Invalid SE must fail validation as specified, not silently receive a weight or disappear.

For each budget separately, the primary pooled score is:

`weighted_MAE_m = sum_i(w_i * e_mi) / sum_i(w_i)`.

Use a common identity cohort and the same fixed weights across compared methods. When every identity occurs once in outer test predictions, this is ordinary pooled OOF weighted MAE. If saved predictions repeat an identity within the same budget/method, first average that identity's absolute errors over its matched occurrences, then weight identities. Do not average predictions first: that would evaluate an ensemble not specified here. Preserve repeat counts and explain any departure from unique OOF coverage.

Also report:

- `rawMAE = mean_i(e_mi)` on the same cohort, using identity-averaged loss if repeats occur.
- `Spearman = corr(rank(y), rank(pred))`, unweighted, with average ranks for ties, computed on saved round-trip numeric predictions. Unique-identity pooled OOF Spearman is the default. If an identity repeats, do not silently average predictions or invent a pooled rank estimand: mark pooled rho pending an explicit repeated-prediction rule and retain per-fold rho.
- Per-fold weighted MAE, rawMAE and Spearman.
- `macro_weighted_MAE = mean_f(weighted_MAE_f)`: equal weight to each of the five folds. It is **not** the primary pooled score. With one test occurrence per identity, pooled weighted MAE averages fold scores in proportion to each fold's total assay weight; equal fold sizes alone do not make them equivalent. Arithmetic mean of fold Spearman is descriptive and is not pooled Spearman.
- `n_unique`, `n_groups`, scored occurrences, total assay weight, and `n_eff = (sum_i w_i)^2 / sum_i(w_i^2)`. The last is a weight-concentration diagnostic, not a count of independent chemical observations. Put fold-level denominators in the download.

All paired differences are **LoRA minus head-only**: negative weighted/raw MAE differences favor LoRA; positive Spearman differences favor LoRA. Keep this sign convention in plots, tables and exports. Never combine budgets into an overall mean or count the 20 fitted systems as independent evidence.

## 3. Minimal HTML: two figures and one compact table

Use a plain page, roughly 1,000 px maximum width: question/title; one sentence identifying reused development folds and the two FIT/CAL budgets; Figure 1; Table 1; Figure 2; a collapsed diagnostic/provenance section with downloads. No score cards, leaderboard ranking, bars, violins, learning curves, or fabricated placeholders. Missing evidence reads `not available` with a reason, never zero.

### Figure 1 — The paired adaptation effect (primary)

Two side-by-side small multiples, N100 and N500, sharing one horizontal scale:

- x: `Δ assay-SE-weighted MAE (LoRA − head-only)` in the verified target's log units.
- y: fixed saved fold order 0–4, followed by a separated `Pooled` lane and `Fold macro` lane.
- Five small dark circles show actual paired fold deltas; no confidence whiskers on those fold points.
- A filled diamond in the pooled lane shows the primary pooled paired difference, with a thin horizontal 95% paired identity-group bootstrap interval.
- An open diamond in the macro lane shows the equal-fold mean delta, without an uncertainty bar. It must be visually secondary to pooled.
- A light vertical zero reference. Caption: “Negative favors LoRA. Five folds share a reused development design; dots are not independent trials. The interval conditions on saved fits and held-out groups.”
- Derive a common symmetric x-domain including zero, every fold mark and both interval endpoints, with modest padding; do not crop unfavorable folds or label fold win-counts as significance.

This figure answers both magnitude and fold consistency without inferring the difference from overlap of separate model intervals. RawMAE and rho deltas belong in Table 1, not two more redundant charts.

### Table 1 — Absolute performance, paired differences, and controls

Two vertically stacked budget sections with the same columns:

`System / contrast | n identities / groups | pooled weighted MAE [95% CI] | fold-macro weighted MAE | rawMAE | Spearman rho`.

Fixed row order within each budget:

1. Native unchanged reference.
2. Head-only.
3. Heads + rank-4 LoRA.
4. **LoRA − head-only**, with paired 95% intervals for weighted MAE, rawMAE and rho wherever well-defined. Show a signed macro difference but no invented macro interval.
5. All eligible existing CPU controls, in stable provenance-defined order, separated by a light rule; exact artifact method names, not a newly selected “best CPU” row.

Absolute weighted-MAE intervals are marginal, not evidence of a paired difference. Other absolute metrics remain point estimates to keep the table compact; full uncertainty data can be downloaded. Use consistent precision (three decimals for these metric displays), retaining full precision in CSV. No automatic winner coloring or significance stars. For the delta row, repeat the same cohort denominator, not a number of fits.

Controls must have the same endpoint, native conversion where applicable, held-out identities, fold membership and compatible label budget/roles. Verify features and historical tuning budget. Unmatched controls go in a separate collapsed context table with their actual cohort/budget and no direct delta. Do not replace the primary cohort by the intersection of every historical control. A secondary matched intersection, if needed, must re-score both reference and comparator and disclose lost coverage.

Repeat the native reference on each matched budget cohort for ease of comparison; if identical, say once that the row is duplicated context, not a second native experiment. For constant CPU references, report rho as undefined; different fold constants can produce a pooled correlation that does not establish within-fold molecular ranking.

### Figure 2 — Point calibration and compression (diagnostic)

A compact 2 × 3 square-panel grid: rows N100/N500, columns Native / Head-only / Heads + LoRA. About 260 px square per panel on desktop; wrap on narrow screens without changing data or scales.

- x: observed target; y: held-out point prediction. Shared x/y range over every displayed observation and prediction, equal aspect ratio, faint identity diagonal.
- Small single-color translucent points; no fitted line, smooth trend, residual-dependent weighting, potency cutoff, clipping, or confidence envelope. Draw real outliers and let ranges expand.
- Hover/focus details: literal identity, fold, grouping ID, observed/predicted value, absolute error, assay SE and weight. Do not encode assay-weight magnitude as dot size: downweighted compounds must remain visible.
- Same cohort as the main contrast. A duplicated native panel is explicitly contextual if test identities are identical across budgets.
- Caption distinguishes accuracy from calibration: correlation can remain high despite offset, wrong scale, or compressed predictions. The diagonal is a reference, not an estimated correction.

A collapsed diagnostic table may carry mean signed error (`pred − observed`), observed/predicted SD and IQR, their ratios, and observed/predicted ranges. Include all saved potency-stratified diagnostics; if no strata were frozen, prefer downloadable residuals against observed potency over inventing outcome-selected bins. Zero denominators produce `undefined` ratios.

## 4. Calibration: do not conflate three meanings

1. **Budget calibration subset:** the protocol reserves 20% of acquired labels. Confirm its actual saved use. The JSON alone does not establish an affine point correction, residual quantile formula, or finite-sample interval procedure. Do not invent these or fit them during reporting.
2. **Point calibration diagnosis:** Figure 2, bias and spread summarize already-saved predictions. Do not fit a post-hoc regression to test labels. If an OOF calibration slope/intercept has already been saved, it can appear with its exact convention and provenance; otherwise omit it under the no-fitting restriction.
3. **Prediction intervals:** the protocol specifies nominal 80% and 90% coverages. If saved bounds and calibration provenance are available, put empirical held-out coverage and mean/median width at each nominal level in a small collapsed table, by system and budget. Use matched rows; disclose whether coverage is unweighted and retain any official weighted version separately. Reconstruct the original interval predicate, not an approximately equivalent rounded endpoint test. Do not substitute the bootstrap interval on a score for a per-compound prediction interval.

Calibration sets are small, especially CAL20. Chemical separation can violate simple exchangeability assumptions. Empirical development-fold coverage is a diagnostic, not a prospective coverage guarantee. Nothing in this design authorizes adjusting predictions, refitting calibration or choosing a better interval after looking at errors.

## 5. Retrospective uncertainty specification

These intervals are **retrospective**, even though this visualization design precedes reading results: they were not specified in the supplied training protocol. They condition on completed training systems, acquired subsets and reused development folds. They do not quantify retraining, new acquisition draws, initialization variation, protocol-selection bias or prospective chemical-shift uncertainty.

Proposed primary reporting algorithm, to be implemented later without refitting:

1. Verify a stable identity-to-resampling-group mapping from the saved manifests. Use at least whole identity groups; if the split manifest supplies broader chemical groups, preserve those together and name the exact grouping. Identity groups are not automatically chemical series. Check each resampling group belongs to one outer test fold; if not, do not split it to satisfy a bootstrap implementation—resolve the mapping before issuing an interval.
2. Form paired per-identity losses for both arms within each budget. Collapse repeated absolute losses within identity before resampling, using exactly matched occurrences. Keep all members of a selected group together.
3. For each replicate, sample groups with replacement **within each fixed fold**, drawing that fold's original number of groups. Apply each group's multiplicity to all its identities. Use the same draws for both arms, every eligible comparator, both budgets on the same cohort, and all metrics. Unequal group sizes contribute their actual identities and assay weights; do not equal-average group means and thereby change the primary estimand.
4. Recompute pooled weighted MAE as the ratio of resampled weighted-loss sum to resampled weight sum; recompute rawMAE and well-defined Spearman on the same sampled identity records, retaining multiplicities. Subtract paired scores within each replicate, rather than subtracting marginal confidence bounds.
5. Proposed reproducibility settings: 10,000 draws, NumPy PCG64 seed 20260907, two-sided percentile bounds at 2.5% and 97.5%. Save the realized group-draw inventory (or multiplicities), exact quantile convention (`linear`), code/version, grouping provenance, and successful/undefined replicate counts. These settings are a reporting proposal, not a retroactive claim of preregistration.
6. Constant observed or prediction vectors make rho undefined. Do not replace undefined bootstrap rho with zero; withhold its interval if the full planned replicate statistic is not defined and disclose the count rather than quietly conditioning on valid draws.

Do not perform an IID bootstrap over prediction rows or 20 fit records. The five fold deltas are paired development summaries, not five independent retrainings. No standard-error-over-20, paired t-test, or binomial fold-win test.

The fixed-fold stratification above intentionally does **not** resample training folds. If a reviewer requests a whole-fold or hierarchical fold-then-group resampling sensitivity, label it separately as a sensitivity, reuse paired draws, and disclose that only five correlated development folds exist and training sets overlap. Such a resample still does not reproduce model refitting and must not replace the primary conditional group interval.

A primary paired interval entirely below zero supports a conditional development-set reduction in weighted MAE; it is not proof of a generalizable LoRA win. Report rawMAE, rho and adverse-fold behavior alongside it. An interval spanning zero is uncertainty, not equivalence. No minimum useful gain or no-harm threshold is present in the supplied protocol; do not introduce one after seeing outcomes.

## 6. Data contract and verification before rendering real results

Required fields or explicit mappings: experiment/task ID; arm; budget; fold; draw/seed; literal identity; resampling-group ID; FIT/CAL/test role; observed endpoint/unit; assay SE; frozen evaluation weight; saved point prediction; optional saved interval bounds; source artifact/hash and comparator feature/tuning provenance.

Acceptance checks for the future reporter:

- Reconcile all 20 expected task identities and their validation/replay records, not merely file presence. Keep partial results labelled partial; never silently average only successful cells into a completed comparison.
- Preserve identifiers as strings and round-trip floating-point values. Reconcile both members' native conversion separately from fitted-score verification.
- Validate exact FIT/CAL counts, paired roles, test disjointness, per-fold scored counts and overlap/grouping audits. Invalid SE, nonfinite prediction or unmatched identity is a visible validation failure, not scientific evidence.
- Recompute scores and table values from saved predictions on CPU; independently check weighted numerator/denominator, pooled versus macro distinction, sign orientation, repeat handling, bootstrap pairing and missing rho handling.
- Verify every mark and row against exported plotting data; record unique-identity, group and occurrence counts instead of assuming protocol development counts equal evaluation counts.
- Keep main exports small: `metrics.csv`, `paired_fold_deltas.csv`, `bootstrap_intervals.csv`, `plot_points.csv`, and a manifest linking the original saved evidence and group draws. These are future proposed artifacts, not created by this review.
- Include protocol/input hashes, creation/edit timestamps, interval caveat and source links in collapsed provenance. Keep the source narrative canonical rather than allowing HTML and Markdown conclusions to diverge.
- Verify real rendered desktop/mobile HTML and exported SVG for clipping, shared-scale correctness, legibility and collisions. Put captions and legends outside plotting areas, keep only thin axes/reference lines, avoid decorative cards, and ensure values are accessible without hover. Numerical tests alone do not establish visual quality.

**Review boundary:** the metric/uncertainty semantics and display layout are specified. Actual calibration implementation, identity/group mapping, CPU-control compatibility, completed-task coverage, numerical results and rendered visual quality remain to be verified by the parent against real artifacts. This review intentionally does not inspect outcomes or assert any method improved.
