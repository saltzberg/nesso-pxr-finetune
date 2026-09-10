# Independent affinity comparison audit

Created 2026-09-07. Independently recomputed from immutable prediction artifacts and protocols; no other analyst's conclusions used. CPU NumPy/pandas/SciPy only; no fits, model inference, GPU use or external APIs.

## Main result

The tested upstream LoRA recipe modestly improves the matched head-only recipe, but does not replace the stronger N500 CPU controls. These are **conditional results on reused development folds**, not an untouched holdout or prospective confirmation. One subset draw, one initialization and one fixed training recipe do not establish the limits of representation adaptation.

Each row below pools the five disjoint test folds: 3,344 unique compounds, once per method/budget. Weighted MAE is sum(w*absolute error)/sum(w), w=1/max(reported assay SE,0.1), not inverse variance. All SE values and predictions are finite and all saved weights match this definition. Macro fold scores are separately preserved in JSON.

| Method | Budget | Weighted MAE | MAE | Spearman |
|---|---:|---:|---:|---:|
| Native, unchanged point | — | 0.627225 | 0.914124 | 0.386872 |
| Head only | 100 | 0.589420 | 0.790666 | 0.443624 |
| Head + upstream LoRA | 100 | 0.583105 | 0.763509 | 0.494274 |
| Fresh vector ridge | 100 | 0.618610 | 0.766250 | 0.460478 |
| Historical descriptor LightGBM | 100 | 0.610266 | 0.745001 | 0.500596 |
| Head only | 500 | 0.577256 | 0.793221 | 0.472631 |
| Head + upstream LoRA | 500 | 0.561016 | 0.756535 | 0.529291 |
| Fresh vector ridge | 500 | 0.534468 | 0.666603 | 0.608439 |
| Historical descriptor LightGBM | 500 | 0.500588 | 0.599636 | 0.656863 |

- LoRA minus head-only weighted MAE: **−0.006315 at N100 (1.07%)** and **−0.016240 at N500 (2.81%)**. Improvements occur in 4/5 and 5/5 folds respectively. Raw MAE and Spearman improve in all ten matched folds. N100 fold 4 is the weighted-MAE exception (+0.003699).
- LoRA beats the unchanged native point in weighted MAE, raw MAE and rank in all ten cells.
- At N100, LoRA's better weighted MAE than descriptors is **not an unqualified win**: descriptor raw MAE and pooled rank are better. At N500, both CPU controls beat LoRA on all three metrics in every fold.
- Fresh ridge uses newly captured 384+384 member vectors, fit-only weighted scaling and grouped three-fold selection among four alphas. The descriptor comparator is the exact historical RDKit2D/Mordred2D/Morgan LightGBM recipe with its original tuning, not an unavailable earlier feature universe. These contextual systems are not optimizer-matched architectural ablations.

## Compression, not constant prediction collapse

No neural system is constant. Pooled head-only predicted/observed SD ratios are 0.429 (N100) and 0.361 (N500); LoRA increases them to 0.517 and 0.453. Head-only N500 has 3,343 distinct predictions; other neural systems have 3,344. LoRA partially restores spread and ranking but leaves substantial regression toward the middle.

Both N500 neural systems predict every test compound below pEC50 6 (maxima 5.897 and 5.913). All 56 observed >=6 compounds are underpredicted; their MAE is 1.173 for head-only versus 1.112 for LoRA. On the 1,036 compounds below 4, N500 LoRA lowers MAE from 1.661 to 1.531 but still overpredicts by mean 1.496. On the 2,308 compounds >=4, MAE slightly worsens from 0.404 to 0.409. N100 likewise improves the <4 tail while slightly worsening >=4 MAE (0.426 to 0.441). Thus the pooled gains conceal a modest potency-stratum trade-off. Assay reliability weighting does not imply that low-potency observations are all noise or censored.

Increasing labels from N100 to N500 does not automatically restore neural dynamic range: both neural recipes become more compressed. Descriptive spread, IQR, bias, calibration slope/intercept and strata are in `results.json`; those diagnostic regressions were not used to alter predictions.

## Intervals

Intervals were independently reconstructed from the within-budget calibration absolute residuals using order ceil((m+1)*coverage), then the original `abs(y-pred)<=radius` predicate. Saved radii, endpoints and coverage all reconcile. These are adapted prediction intervals, not assay confidence intervals.

| Method | N | 80% coverage | 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|
| Head only | 100 | 77.75% | 89.14% | 3.948 |
| LoRA | 100 | 78.89% | 89.74% | 3.833 |
| Head only | 500 | 76.17% | 89.06% | 3.814 |
| LoRA | 500 | 75.96% | 89.35% | 3.553 |

Pooled 90% coverage near nominal hides fold variation: LoRA N100 ranges 82.49–96.26%; N500 85.20–93.87%. LoRA N500 80% coverage ranges 68.61–82.06%. Reduced interval width does not establish chemical-shift coverage guarantees; N500 80% intervals under-cover overall. All neural intervals are finite. Controls' complete fold and pooled coverage/width are also preserved.

## Target, label roles and chemical separation

The fitted target is **cellular PXR pEC50** from the original prepared labels. Released affinity scalar conversion `6 - arithmetic mean(log10(IC50/µM))` is a pIC50-equivalent anchor; it is not native support for cellular activation pEC50. Both trained neural arms are adapted prediction systems; the frozen structure trunk is not trained.

- Per N100 task: 80 FIT + 20 calibration labels; per N500 task: 400 FIT + 100 calibration labels. No early stopping or training hyperparameter search in either neural arm. Calibration labels affect intervals, not point fitting.
- All role identities and disjointness reconcile to the frozen subsets and test assignments. Every one of the 20 N100 calibration compounds per fold becomes FIT at N500: acquired prefixes are nested, role membership is not fixed.
- Across the five folds, N100 uses 472 unique acquired compounds (384 distinct FIT, 96 distinct calibration; some change roles across folds); N500 uses 1,866 (1,575 FIT, 468 calibration). These counts must not be added as independent labels or evaluations. Each method/budget evaluates the same 3,344 test identities; repeated budgets/methods are not additional test molecules.
- Frozen assignments contain 1,975 chemical groups, 1,458 singleton groups, largest group 36. Singleton compounds are 43.60% of the evaluated cohort. The protocol removes acquired candidates at Morgan radius-2/2048 Tanimoto >=0.35 to any test member. This audit verifies the saved role allocations, not a new fingerprint/purge reconstruction.

## Integrity and mismatches

**1,785 checks passed, zero failed.** Verified exact 20-task neural inventory plus 30 completed control cells; all task output hashes and final epoch-40 neural state hashes; label/SE/weight identity; unique test IDs; calibration roles; saved raw/scored predictions; all primary metrics; and interval reconstruction. All ten reused descriptor payloads are byte-identical to their original historical task artifacts. The audit records 599 source/artifact/code SHA256 entries.

Two qualifications matter:

1. Remote versus local CPU native baselines are not fully byte/numerically identical: 687 unique test compounds differ, maximum absolute difference **9.5367431640625e-7**. Fold 0 is identical. Differences repeat across budgets, and do not change the displayed headline metrics. Both references are separately scored and retained; they were not silently conflated. This is a numerical source-path discrepancy, not a metric or row-integrity failure.
2. `CONTROLS.md` still says fresh production controls await capture completion. Saved artifacts contain all **30 completed cells**. Its unexecuted-status sentence is stale; no protocol/report was edited by this audit.

Hash agreement establishes artifact consistency, not model correctness or independent reviewer authentication. Saved frozen-before/after digests agree, but this audit did not execute the models or independently reproduce training gradients or model-state predictions. No confidence interval over refitted experiments or significance claim is made. Development-fold reuse and overlap of label/test roles across different folds prevent interpreting fold counts as independent experimental replications.

## Reproduce and outputs

From `/home/dan/projects/nesso-finetune`:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python PXR/artifacts/analysis/affinity_comparison_20260907/independent_audit/audit.py
```

- `audit.py`: independent calculation and validation code, no original metric helper imports.
- `results.json`: pooled/fold metrics, paired differences, label roles, checks and scientific limits.
- `pooled_metrics.csv`, `fold_metrics.csv`: scalar metric exports; native CPU and remote variants deliberately separate.
- `input_hashes.json`: hashes of consumed protocols, allocations, predictions, models and audit code.

Only this audit directory was written. Saved training and control artifacts remain unchanged.
