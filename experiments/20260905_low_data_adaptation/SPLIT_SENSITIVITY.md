# Structure-only split sensitivity

Development compounds: 3344; Morgan radius 2, 2048 bits; five balanced folds, seed 20260905. Existing grouping and fold functions reused unchanged. No outcomes inspected or models fitted.

| Similarity cutoff | Groups | Singleton compounds (%) | Largest group | Fold sizes | Cross-fold NN q05 / median / q95 | NN ≥ cutoff (%) |
|---|---:|---:|---:|---|---|---:|
| 0.50 | 3030 | 84.33 | 10 | 668, 669, 669, 669, 669 | 0.264 / 0.361 / 0.480 | 1.94 |
| 0.45 | 2844 | 75.66 | 13 | 669, 668, 669, 669, 669 | 0.263 / 0.354 / 0.449 | 4.99 |
| 0.40 | 2483 | 60.97 | 23 | 669, 669, 669, 668, 669 | 0.263 / 0.345 / 0.471 | 11.72 |
| 0.35 | 1975 | 43.60 | 36 | 669, 669, 669, 668, 669 | 0.262 / 0.333 / 0.500 | 32.51 |

Matched existing random identity-held-out split: fold sizes [669, 669, 669, 669, 668]; actual cross-fold NN q05/median/q95 = 0.265/0.370/0.577.

## Interpretation and boundaries

- Lower cutoffs broaden centroid groups; balanced folds alone do not establish series holdout. Butina does not enforce a maximum cross-fold similarity; NN ≥ cutoff explicitly measures this limitation.
- NN is each held-out compound’s maximum Tanimoto to the full complementary outer training pool. Low-N fit-only novelty must be recalculated after acquisition; these are not fit-subset NN values.
- The historical **83.76% singleton** caveat remains attached to its earlier grouping/cohort and is not erased or reinterpreted by this sensitivity. That historical number is supplied context, not recalculated here. The current .50 baseline is independently replayed above.
- Conditional .30: not evaluated because at least one requested cutoff reduced singleton percentage by ≥10 percentage points. This audit-only substantial-improvement rule was fixed before evaluating cutoffs; it is not a performance-based choice.
- No cutoff is adopted here. Parent must approve an explicit protocol amendment and freeze new prepared artifacts **before outcome fits**. The existing protocol and entire prepared folder are byte-hash unchanged.
- Baseline .50 group IDs and chemical folds, plus random folds, exactly replay the existing assignments. Stored Morgan bits round-trip exactly. Every chemical group stays wholly within one fold.

## Artifacts

- `artifacts/experiments/low_data_20260905/split_sensitivity/sensitivity.json`: full pooled/per-fold NN quantiles, group-size histograms, hashes, audit settings.
- `structure_assignments.npz`: row-indexed groups, folds, and actual NN values for replay; no outcomes.
- `audit.py`: executable structure-only audit.

## Source discrepancy

Supplied context said .50 had 3031 groups, largest 14, and NN median about .315; exact replay of the current prepared assignments instead gives 3030 groups, largest 10, and actual NN median .361111. Singleton percentage agrees (84.3301%). This audit reports the current on-disk structures and assignments, not the conflicting contextual counts.

At .35 fewer than half the compounds are singletons, but cross-fold NN q95 rises to .500 and the maximum remains .772. Broader centroid groups therefore do not imply a strict similarity-separated series holdout.
