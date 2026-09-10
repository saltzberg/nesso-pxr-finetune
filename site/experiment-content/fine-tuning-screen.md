# Fine-tuning screen

[← Overview](../../index.html#methodology)

Fine-tuning improved direct transfer to PXR induction, but the tested neural recipe did not beat ligand descriptors at N500. This page brings together the completed matched affinity-adaptation screen and its Vast.ai continuation; it is a synthesis of the existing experiment, not an additional independent study.

## What we trained

We first tested fitting the continuous regression heads, then compared that arm with heads plus upstream low-rank adaptation. Both arms update `affinity_heads.to_affinity_pred_value` in **both** `affinity_module` and `affinity_module2`. The LoRA arm additionally trains rank-4, alpha-4 adapters in each member at:

- `esm_proj.1`
- `esm_proj.3`
- `pairformer_stack.layers.0.transition_z.fc1`

These are local affinity projections and the first affinity Pairformer block's transition, not the external ESM model. The original weights under the adapters, binary heads, main trunk and all other parameters and buffers remain frozen. See [architecture analysis and selected layers](../../architecture/index.html#fine-tuning) for the highlighted diagram and exact layer audit.

The completed matrix contains **20 scientific fits: five reused development folds × two label budgets × two neural arms**. N100 uses 80 FIT and 20 calibration compounds per fold; N500 uses 400 FIT and 100 calibration compounds. Each method/budget pools one held-out prediction for each of 3,344 compounds. There is one subset draw and seed (42), not five independent replications.

Both arms use the same fixed 40-epoch AdamW recipe: learning rate 0.0003, weight decay 0.0001, Huber beta 0.5, and full-FIT inverse-SE-weighted gradient accumulation with one optimizer update per epoch. There is no early stopping or checkpoint selection from test outcomes. Calibration labels are reserved for intervals, not gradient fitting.

## What ran on Vast.ai

The scientific run began locally and finished through Vast.ai instance **50146714**. The preserved matrix contains **3 historical local fits and 17 remote continuation fits**, not 20 newly trained cloud fits. The three completed local tasks were copied byte-identically and not refitted; a resumed task retained its saved epoch and optimizer state.

Before the scientific matrix, separate engineering and eight-compound pilot runs checked differentiable/native parity, seed isolation, pause/resume and saved-state replay. Those smoke tests establish that the training machinery works; they are not held-out performance results and are not included in the 20 scientific fits.

The continuation used immutable cached packets for both affinity members, source/checkpoint bindings and first-access SHA-256 checks. Cross-hardware historical replay used an explicitly approved absolute tolerance of 0.0001; new-fit same-machine saved-state replay remained exact. The final execution record confirms all 20 fits, full-cache verification and all-fit replay completed. Artifacts were preserved locally before the cloud instance was destroyed.

## Results

Pooled held-out metrics from the saved [exact metrics CSV](../../evidence/reports/architecture/results/pooled_metrics.csv), rounded to four decimals below. Weighted MAE and raw MAE are lower-is-better; Spearman ranking is higher-is-better. These are pooled scores, not equal-fold means.

| Budget (FIT / calibration) | System | Weighted MAE | Raw MAE | Spearman |
|---|---|---:|---:|---:|
| N100 (80 / 20) | Native unchanged | 0.6272 | 0.9141 | 0.3869 |
| N100 (80 / 20) | Head-only | 0.5894 | 0.7907 | 0.4436 |
| N100 (80 / 20) | Head + LoRA | 0.5831 | 0.7635 | 0.4943 |
| N100 (80 / 20) | Ligand-descriptor LightGBM | 0.6103 | 0.7450 | 0.5006 |
| N500 (400 / 100) | Native unchanged | 0.6272 | 0.9141 | 0.3869 |
| N500 (400 / 100) | Head-only | 0.5773 | 0.7932 | 0.4726 |
| N500 (400 / 100) | Head + LoRA | 0.5610 | 0.7565 | 0.5293 |
| N500 (400 / 100) | Ligand-descriptor LightGBM | 0.5006 | 0.5996 | 0.6569 |

Weighted MAE = Σ wᵢ |predictionᵢ − observedᵢ| / Σ wᵢ, where wᵢ = 1/max(reported pEC50 SEᵢ, 0.1). Raw MAE divides the unweighted absolute-error sum by 3,344. Native uses the unchanged affinity-equivalent conversion, 6 minus the mean of the two continuous member values; it is not a native cellular pEC50 output. Native point predictions are identical at both budgets. The historical descriptor fits were reused and verified, not rerun on Vast.ai; their tuning procedure differs from the neural recipe, so this is practical context rather than an optimizer-matched architecture ablation.

At N100, LoRA has the lowest weighted MAE of these systems, but descriptors have lower raw MAE and better ranking. At N500, LoRA improves on head-only in all five folds on weighted MAE, yet descriptors remain better on all three pooled metrics. Much of the LoRA gain comes from reducing low-potency overprediction; it is not a uniform improvement across the activity range. Frozen-feature ridge also beats both neural recipes at N500, so this result does not mean that Nesso's representations lack useful information.

## Interpretation and limits

This is a retrospective comparison on reused development folds, not a fresh blind-challenge result or a broad hyperparameter search. The folds share training pools, and the single draw/seed does not measure retraining variability. Chemical purging excludes acquired compounds with Morgan radius-2/2,048-bit Tanimoto at least 0.35 to any outer-test compound. Calibration roles can change between the nested N100 and N500 budgets. No outer-test labels entered the fits.

The tested adaptation works as a correction but does not justify replacing descriptors at N500. A further optimization screen needs an inner selection rule, additional draws/seeds and untouched evaluation rather than choosing another recipe on these outcomes.

## Supporting methods and evidence

- [Upstream comparison experiment](../upstream-comparison/index.html) and [transfer results and scatterplots](../../chapters/transfer/index.html)
- [Head-only methods](../head-only/index.html) and [chemical splits](../identity-standardization/index.html)
- [Exact completed training protocol](../../evidence/experiments/20260906_affinity_comparison/train_protocol.json)
- [Scientific results](../../evidence/experiments/20260906_affinity_comparison/RESULTS.md) and [execution record](../../evidence/experiments/20260906_affinity_comparison/EXECUTION.md)
- [Vast.ai continuation binding and historical-task list](../../evidence/artifacts/cloud/50146714/continuation.json)
- [Fit/control provenance table](../../evidence/reports/architecture/results/sources.csv) and [full method report](../../evidence/reports/architecture/results/RESULTS.md)
- [Editable page Markdown](../../experiment-content/fine-tuning-screen.md)
