# Nesso PXR: completed low-data study

This study tests how well target-fitted readouts on frozen Nesso-1 vectors predict cellular PXR pEC50 from small, counted label budgets.

Created: 2026-09-06T13:53:57.493848+00:00. Last edited: 2026-09-06T14:05:03.071937+00:00. Retrospective development-set assessment; no model fits were run for this closeout.

## What was learned

**The first study is complete: 7,200/7,200 task systems, 600 fully paired cells. Scientific completion does not establish a deployment-ready winner.** Twin continuous heads are the provisional Nesso route within the tested recipes. More labels help; released head initialization does not show a consistent advantage over the matched random head. Extra binary branches add complexity without a compelling observed gain, and rank-4 final-head LoRA trades accuracy for a modest fitting-cost reduction.

At N500, pretrained-head mean weighted MAE is 0.5399 / 0.5402 (random / cluster), versus native 0.6281 / 0.6282. At N25, adaptation can worsen the cluster score. The scaled generic MLP and ridge have demonstrated extrapolation vulnerabilities; these are not evidence that all MLPs, ridge models, or frozen representations fail.

The comparator is **Morgan-only LightGBM**, not the historical descriptor-rich system. Full-budget Morgan ridge is close to the continuous heads, and random-full raw MAE favors Morgan ridge (0.6431 versus 0.6463). This study cannot establish superiority over the absent richer comparator. Weighted median and native linear calibration were absent too.

## What was tested and where labels entered

```text
Canonical identity + prepared ligand state
→ released Nesso-1 v1.0.0 cached affinity vectors (2 × 384D, frozen)
  OR deterministic radius-2 / 2,048-bit Morgan fingerprints
→ method-specific preprocessing and readout [fit labels only]
→ arithmetic mean of final neural seeds 42 / 43 / 44
→ residual split-conformal intervals [reserved calibration labels inside N]
→ fixed outer-test cellular pEC50 evaluation [never fitting or selection]
```

The matrix is two policies × five folds × ten paired draws × six budgets × twelve methods. N=25/50/100/250/500 means 20/40/80/200/400 fit-and-selection labels plus 5/10/20/50/100 calibration labels. Full uses 2,675–2,676 acquired labels, 2,140 for fitting and 535–536 for calibration—not all 3,344 development labels. Acquired prefixes nest, but fit-only sets need not nest as calibration suffixes move. Repeated native point predictions are an unchanged anchor. Nonneural models fit once per task and repeat identical predictions across seed slots, not three independent fits.

| Method | Exact fitted flow and boundary |
| --- | --- |
| weighted_mean | Inverse-SE-weighted fit-label mean; not the absolute-loss-optimal weighted median. |
| native_continuous | Unchanged released two-member mean after `6 − raw_affinity`; pIC50-equivalent affinity, not native cellular pEC50. No point-prediction fitting; intervals still use calibration labels. |
| scalar_ridge | Nine reconstructed member/ensemble continuous values, binary logits and probabilities → weighted fit-only StandardScaler → weighted ridge. |
| repr_ridge | Concatenated 768D vectors → weighted fit-only StandardScaler → weighted ridge. |
| repr_mlp | Same scaled 768D inputs → 768–128–1 ReLU MLP, direct pEC50 output, no target centering; 98,561 fitted parameters per seed. |
| head_random / head_pretrained | Unscaled vectors → matched twin 384–384–384–1 ReLU MLPs → `6 − raw` per member → mean; 592,130 fitted parameters per seed. Only head initialization differs; upstream encoder is pretrained and frozen in both. |
| dual_head | Both continuous and binary-score MLPs, binary-logit maps and fusion gates train; raw member output = `continuous + 1.5*tanh(scale*binary_logit + offset)` before conversion. Gates start at zero; no extra binary assay labels; 1,184,268 fitted parameters per seed. |
| head_lora | Rank-4 A/B updates in all three Linear layers of both released continuous MLPs; alpha/rank=1, zero-B initialization. 15,368 fitted and 607,498 stored neural parameters per seed; base heads and vectors frozen. Not upstream adaptation. |
| morgan_ridge | Unscaled binary Morgan bits → weighted ridge; no descriptors or PCA. |
| morgan_lightgbm | Morgan-only absolute-loss trees; seven leaves, learning rate .05, min_child_samples 5, L2=1, 50/150 trees selected inside fit labels. |
| morgan_knn | Weighted label mean among 1/5 nearest Tanimoto neighbors; retains fit bits and labels. |

Selection uses three-fold **GroupKFold for both policies in the actual runner**: chemical IDs for cluster tasks, unique record IDs for random tasks. Every inner scaler is fitted anew inside its inner-training partition; final scalers use only final fit rows. Ridge alpha is 1/100/10,000. Neural LR .0001/.001 and epochs 1–40 are chosen by pooled inner-OOF weighted MAE with seed 42; all final seeds refit. AdamW has zero weight decay, batch size 64, gradient norm clipping 1, weighted smooth-L1 beta .5 and ensemble/member loss weights .50/.25/.25. Training weights are globally normalized. Ridge fits weighted squared loss; trees fit weighted absolute loss. Label curves also change optimizer update counts, not just label count at fixed computation. Generic MLP versus twin head changes scaling, architecture, initialization and output parameterization together: it is not a controlled one-factor comparison.

## Results: task means, not independent replicates

Headline weighted MAE is `sum(w * abs(pred − true)) / sum(w)`, with **w = 1 / max(assay SE, 0.10)**, in pEC50 units. It is inverse SE, not inverse variance or model-uncertainty weighting; curation weights are excluded. The floor is a fixed pre-outcome engineering cap, not independently measured assay noise. Weights are fixed across methods and budgets on the same test identities.

| Method | Random 25 | 100 | 500 | Full | Cluster 25 | 100 | 500 | Full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weighted_mean | 0.6686 | 0.6590 | 0.6567 | 0.6559 | 0.6959 | 0.6653 | 0.6568 | 0.6567 |
| native_continuous | 0.6281 | 0.6281 | 0.6281 | 0.6281 | 0.6282 | 0.6282 | 0.6282 | 0.6282 |
| scalar_ridge | 0.6622 | 0.6223 | 0.5870 | 0.5814 | 0.6803 | 0.6241 | 0.5872 | 0.5827 |
| repr_ridge | 0.7489 | 0.6282 | 0.5905 | 0.5197 | 0.6913 | 0.6303 | 0.5888 | 0.5207 |
| repr_mlp | 2.6565 | 0.9533 | 0.6719 | 0.5449 | 1.9236 | 1.0326 | 0.6735 | 0.5340 |
| head_random | 0.6202 | 0.6054 | 0.5404 | 0.5043 | 0.6251 | 0.5978 | 0.5390 | 0.5005 |
| head_pretrained | 0.6168 | 0.5915 | 0.5399 | 0.5053 | 0.6343 | 0.5938 | 0.5402 | 0.5026 |
| dual_head | 0.6199 | 0.5930 | 0.5396 | 0.5051 | 0.6371 | 0.5961 | 0.5404 | 0.5028 |
| head_lora | 0.6212 | 0.6034 | 0.5622 | 0.5086 | 0.6382 | 0.6078 | 0.5633 | 0.5076 |
| morgan_ridge | 0.6756 | 0.6564 | 0.5751 | 0.5080 | 0.7070 | 0.6571 | 0.5777 | 0.5169 |
| morgan_lightgbm | 0.7749 | 0.6579 | 0.5867 | 0.5392 | 0.8320 | 0.6710 | 0.5885 | 0.5428 |
| morgan_knn | 0.6895 | 0.6413 | 0.5973 | 0.5620 | 0.7349 | 0.6446 | 0.6020 | 0.5838 |

Each entry averages 50 paired fold/draw task scores; all original extremes are retained. N50 and N250, medians/ranges, raw MAE, bias, interval diagnostics, cost and floor .05/.20 score-only sensitivities are in [recomputed aggregates](../../site/low-data-assessment/aggregates.csv). [Original pooled metrics](../../site/low-data-assessment/sources/metrics.csv) instead divide total weighted error by total weight. Small differences from task means are expected and verified; neither aggregation grants independence to repeated rows. Score sensitivities do not retrain or select a new method.

### Released head initialization: mixed descriptive effects

| Split | Labels | Pretrained − random | Draw-averaged fold range |
| --- | --- | --- | --- |
| random | 25 | -0.0034 | -0.0086 to +0.0063 |
| random | 100 | -0.0140 | -0.0342 to -0.0019 |
| random | 500 | -0.0005 | -0.0032 to +0.0022 |
| random | full | +0.0010 | -0.0010 to +0.0031 |
| chemical_cluster | 25 | +0.0092 | +0.0062 to +0.0157 |
| chemical_cluster | 100 | -0.0040 | -0.0189 to +0.0127 |
| chemical_cluster | 500 | +0.0012 | -0.0036 to +0.0096 |
| chemical_cluster | full | +0.0022 | +0.0008 to +0.0028 |

Negative means lower error for pretrained heads. Fold ranges are observed variation, **not confidence intervals**. [All six paired contrasts](../../site/low-data-assessment/paired.csv) retain task median/range and fold-average signs. There is no accepted practical improvement margin and no dependence-aware inferential claim of superiority or equivalence. A random head here is not a random encoder.

### Bias, potency and uncertainty remain limitations

| Split / full budget | Weighted MAE | Raw MAE | Bias (pred − true) | 90% coverage | 90% width |
| --- | --- | --- | --- | --- | --- |
| random | 0.5053 | 0.6463 | 0.2967 | 0.9016 | 3.0121 |
| chemical_cluster | 0.5026 | 0.6523 | 0.3300 | 0.8988 | 3.0597 |

Full-budget pretrained-head task means are above. [Potency-stratified error and bias](../../site/low-data-assessment/potency.csv) report all saved observations in explicitly left-closed bins [3.5,4.5), [4.5,5.5), with tails &lt;3.5 and ≥5.5. [Prediction-tail diagnostics](../../site/low-data-assessment/prediction_tails.csv) preserve every task's range and quantiles. These pooled diagnostic observation counts include repeated compounds; they are not independent sample sizes. Low potency or a numeric SE alone does not establish noise, inactivity or censoring. pEC50 is potency, not Emax/efficacy or clinical liability.

Ordinary residual split-conformal intervals are not SE-weighted coverage guarantees. N25 has only five calibration labels: all nominal 90% intervals are unbounded, so 100% observed coverage is not useful precision. Larger-budget marginal coverage does not imply conditional or chemical-shift coverage. Seed SD is a separate uncalibrated diagnostic. The original revised reporter preserves the runner's residual predicate; 25 task/coverage-level floating-boundary differences were independently reconciled without widening intervals ([receipt](../../site/low-data-assessment/sources/interval_replay.json)).

## What failed: reproducible does not mean safe

In saved-state probes from the [independent assessment](../../site/low-data-assessment/sources/independent_assessment.md), random fold2/draw09 N50 representation ridge has a fit-only scale 5.77×10⁻⁶, a held-out standardized value around 232,721 and a single-feature contribution of magnitude 1,706.817. Prediction −1,704.6863 for measured pEC50 2.69 replays exactly; task weighted MAE is 5.3879. This is a fitted-system scaling/extrapolation vulnerability, not CSV corruption or evidence of valid deployment behavior. Removing even its worst row descriptively does not repair the task.

The worst N25 MLP probe has a standardized held-out value around 7,966 and prediction −354.51 despite decreasing training loss. Its selected epoch is 25, so a universally too-short 40-epoch schedule is not demonstrated. N25 MLP task-median errors (1.8104 / 1.6242) show the weakness is broader than one row. These probes establish poor out-of-sample behavior, not that scaling alone explains every failure. Stable scaling, regularization and target centering are **new controlled hypotheses**. No clipping, extreme-row removal or post-hoc replacement changes the official scores here.

## Generalization: what remains unproven

Only 3,344 development identities enter this matrix. The historically opened 790- and 513-compound challenge sets are excluded from fitting, tuning and evaluation here; that exclusion does not make this a prospective blinded study. Neither upstream pretraining identity/endpoint overlap nor every historical development choice is ruled out by these artifact checks. Follow-up recipes informed by these outcomes remain retrospective and hypothesis-generating, including any repartition of these same compounds.

The chemical split holds Morgan Butina .35 **centroid group IDs** apart: 1,975 groups, 1,458 singleton compounds (43.60%), largest group 36. It does not enforce strict pairwise chemical separation. The actual complementary-pool nearest-neighbor median is 0.333, q95 0.500, maximum 0.772; 32.51% have an outer-pool neighbor at or above .35. Shared scaffolds remain. These similarities use the full outer pool, not the smaller fitted subset. The structure-only .50→.35 amendment and both histories are preserved. Similar random/cluster scores do not prove robustness to chemical shift; no temporal, prospective, cross-target or strict-novelty validation is established.

There are 4,815,360 prediction observations but 3,344 unique compounds. Test identities repeat across draws/budgets, outer training pools overlap, and seeds share labels. Fifty task scores are not fifty independent experiments. Unique-compound effective weighted N is 2,782.34; all assay SEs are finite and positive and 1,107 are at/below .10, as independently verified. No independent-row bootstrap or invented uncertainty/margin is used.

Final-head LoRA is not the separate upstream feasibility pilot. That pilot updated 12,288 rank-4 parameters in first-member `esm_proj.1`, `esm_proj.3` and `pairformer_stack.layers.0.transition_z.fc1` on one real input using two SGD steps and a synthetic detached-native-plus-one target, **without an assay label**. Gradient/frozen-weight/reload checks establish an engineering path, not PXR learning or both-member adaptation. No upstream fits or new models were run for this closeout.

## Completion and verification

This closeout freshly reconciles the final task table against the original protocol Cartesian product and task manifest, re-scores all 4,815,360 original prediction rows, and checks all 43,200 task-output SHA-256 values. It also rechecks 22 bound source files and 9 bound inputs. Maximum prediction metric discrepancy: 0; original aggregate and paired tables match within declared floating-point tolerances ([verification](../../site/low-data-assessment/verification.json)). The independent reviewer previously replayed four representative fitted systems exactly; this reporting pass hashes those model artifacts but does not deserialize or replay them again.

Summed factory fitting time is 3.803 hours. This includes inner selection, final fitting, prediction and in-memory replay, but excludes extraction, preparation, queue interruptions and reporting; it is not elapsed wall time or deployment cost. Actual factory numerical threads are capped at two. Historical reports/models/code and full-matrix plots are untouched. Historical partial-status wording is superseded by the completed artifacts, not silently edited. The assessment copies below are byte-identical snapshots, not rewritten independent conclusions.

## Evidence and reproduction

- [Machine-readable verification](../../site/low-data-assessment/verification.json), [source-copy hashes](../../site/low-data-assessment/source_manifest.json), [generated-file hashes](../../site/low-data-assessment/artifact_manifest.json)
- [Recomputed aggregates](../../site/low-data-assessment/aggregates.csv), [paired effects](../../site/low-data-assessment/paired.csv), [potency diagnostics](../../site/low-data-assessment/potency.csv), [all task prediction tails](../../site/low-data-assessment/prediction_tails.csv)
- [Report Markdown](../../site/low-data-assessment/report.md); [report generator](../../site/low-data-assessment/closeout.py)

- [independent_assessment.md](../../site/low-data-assessment/sources/independent_assessment.md)
- [independent_assessment_aggregates.csv](../../site/low-data-assessment/sources/independent_assessment_aggregates.csv)
- [independent_assessment_diagnostics.csv](../../site/low-data-assessment/sources/independent_assessment_diagnostics.csv)
- [independent_assessment_paired.csv](../../site/low-data-assessment/sources/independent_assessment_paired.csv)
- [independent_assessment_verification.json](../../site/low-data-assessment/sources/independent_assessment_verification.json)
- [task_metrics.csv](../../site/low-data-assessment/sources/task_metrics.csv)
- [metrics.csv](../../site/low-data-assessment/sources/metrics.csv)
- [summary.json](../../site/low-data-assessment/sources/summary.json)
- [interval_replay.json](../../site/low-data-assessment/sources/interval_replay.json)
- [protocol.json](../../site/low-data-assessment/sources/protocol.json)
- [run_bindings.json](../../site/low-data-assessment/sources/run_bindings.json)
- [task_manifest.json](../../site/low-data-assessment/sources/task_manifest.json)
- [split_audit.json](../../site/low-data-assessment/sources/split_audit.json)
- [outer_nearest_similarity.csv](../../site/low-data-assessment/sources/outer_nearest_similarity.csv)
- [preparation_manifest.json](../../site/low-data-assessment/sources/preparation_manifest.json)
- [README.md](../../site/low-data-assessment/sources/README.md)
- [MODEL_NOTES.md](../../site/low-data-assessment/sources/MODEL_NOTES.md)
- [SPLIT_SENSITIVITY.md](../../site/low-data-assessment/sources/SPLIT_SENSITIVITY.md)
- [ADAPTER_FEASIBILITY.md](../../site/low-data-assessment/sources/ADAPTER_FEASIBILITY.md)
- [REPORTING_NOTES.md](../../site/low-data-assessment/sources/REPORTING_NOTES.md)
- [IMPLEMENTATION_PLAN.md](../../site/low-data-assessment/sources/IMPLEMENTATION_PLAN.md)

Run from the PXR repository root using the existing runtime (reporting only):

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src:. \
  python \
  experiments/20260906_low_data_followup/code/closeout.py build
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src:. \
  python -m pytest \
  tests/test_followup_closeout.py -q -o addopts=''
python \
  experiments/20260906_low_data_followup/code/closeout.py verify \
  --url http://192.168.6.154:8890/low-data-assessment/
```

The LAN page is served by the existing site service on port 8890. No follow-up status links are published until actual report targets exist; completion here refers only to the first study, not the new lanes.
