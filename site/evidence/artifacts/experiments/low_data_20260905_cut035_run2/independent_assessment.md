# Independent assessment: what the Nesso-1 PXR low-data study tested and learned

## Bottom line

The full cached-feature experiment is complete and reproducible. It provides a useful provisional answer about **training a PXR readout on frozen Nesso features**, not about how to train or fine-tune Nesso-1 upstream. The twin continuous readouts are the strongest practical Nesso candidates in this tested matrix. More labels improve their predictions; released head initialization does not confer a consistent advantage over the matched random head. Adding the binary branches is not a convincing improvement, and final-head LoRA exchanges some accuracy for fewer trainable parameters and lower fitting cost. None of these is an upstream adaptation result.

The weak generic representation MLP and erratic representation ridge should not be interpreted as proof that those model families or the representation lack useful signal. Saved-model inspection identifies severe small-subset scaling/extrapolation in representative failures. Optimization choices are also restrictive, but their causal contribution is not established. The existing scores must remain unchanged; fixing these weaknesses would be a new comparison.

This assessment is retrospective, independent of the existing report's narrative, and descriptive rather than a declaration of a statistically established winner. No training, extraction, network call, or modification of existing run/report artifacts was performed.

## 1. Intent and scope

The declared question is: “How should pretrained Nesso-1 be adapted with 25–500 PXR cellular-activation labels?” Operationally, the study asks which readout, fitted and selected inside a small counted target-label budget, best predicts held-out development-set pEC50. It is a passive, paired learning-curve experiment, not active acquisition, hit enrichment, mechanistic inference, or an evaluation of clinical PXR liability. Predicting potency in this retrospective assay cohort does not by itself establish performance on all new molecules, nonresponders, efficacy/Emax, or clinically relevant liability.

The source is the 3,344 development compounds in `data/published/modeling_manifest.csv`, with immutable cached Nesso features and the released v1.0.0 checkpoint. The already opened 790- and 513-compound challenge sets are not fitting or tuning sets in this matrix. That exclusion does not turn a historically developed, single-target study into a prospective blinded test. This assessment verifies the current bound files and label roles, not the absence of endpoint/identity exposure in upstream pretraining or every historical development decision.

The primary scientific intervention is downstream readout fitting. The separately documented first-affinity-member LoRA experiment is engineering feasibility only.

## 2. Exact matrix, model flow, and label accounting

Canonical development identity and prepared ligand state → frozen cached Nesso affinity representations (two 384-dimensional vectors), or radius-2 2,048-bit Morgan fingerprints → declared readout and fit-only preprocessing → final-seed arithmetic mean → intervals calibrated within the same acquired budget → outer-test pEC50 prediction.

The matrix is 2 split policies × 5 outer folds × 10 subset draws × 6 budgets × 12 methods = **7,200 task systems**. Budgets are 25, 50, 100, 250, 500, and a separate full outer-pool reference. Each task contains the three final neural seeds 42/43/44; seeds are not three extra matrix tasks. Nonneural states/predictions are fitted once and copied across those seed slots. Pretrained heads share the released initialization; their seed variation comes from training order/numerics, not different pretrained checkpoints. The random-head control changes head initialization, not the frozen upstream encoder.

Every split/fold/draw/budget cell has all 12 methods on the same test identities and the same fit/calibration identities. Acquired sets are nested seeded prefixes; their fit-only portions need not be nested because the calibration suffix moves. The budget counts every unique target label used for fitting, inner tuning/epoch selection, and interval calibration:

| Acquired labels | Fit + inner selection | Reserved calibration |
|---:|---:|---:|
| 25 | 20 | 5 |
| 50 | 40 | 10 |
| 100 | 80 | 20 |
| 250 | 200 | 50 |
| 500 | 400 | 100 |
| Full: 2,675–2,676 | 2,140 | 535–536 |

“Full” is not a fit on all 3,344 labels. Full-reference draws still change fit/calibration membership and row order. The native point predictor consumes no PXR fit labels, although its intervals use reserved calibration labels; its repeated point scores are a repeated anchor, not new training evidence.

### Readouts actually implemented

| Method | Fitted flow; trainable parameters per final seed |
|---|---|
| weighted_mean | Inverse-SE weighted fit-label mean; 1. A mean, not the weighted-median optimum for absolute loss. |
| native_continuous | Unchanged released two-member mean after `6 - raw_affinity`; 0. pIC50-equivalent affinity anchor, not a native cellular-pEC50 model. |
| scalar_ridge | Weighted fit-only scaler → ridge on nine reconstructed member/ensemble continuous values, binary logits, and probabilities; 10 coefficients/intercept. |
| repr_ridge | Weighted fit-only scaler → ridge on concatenated 768D vectors; 769. |
| repr_mlp | Weighted fit-only scaler → 768–128–1 ReLU MLP directly predicting pEC50; 98,561. |
| head_random / head_pretrained | Identical twin 384–384–384–1 ReLU MLPs, one per frozen member vector; convert each by `6 - raw`, average; 592,130. Only initialization differs between these controls. No input StandardScaler. |
| dual_head | Released continuous and binary-score MLPs plus binary-logit maps and fitted fusion gates; 1,184,268. Per member, raw output is `continuous + 1.5*tanh(scale*binary_logit + offset)`; scale/offset start at zero. All branches train against pEC50 through this fusion, not extra binary assay labels. |
| head_lora | Rank-4 A/B updates in all three Linear layers of each released continuous MLP, with alpha/rank=1; 15,368 trainable, 607,498 stored neural parameters. Base heads and cached vectors stay frozen. Zero-B start preserves the native head. |
| morgan_ridge | Unscaled binary Morgan bits → weighted ridge; 2,049. |
| morgan_lightgbm | Morgan-only weighted absolute-loss LightGBM, 7 leaves, learning rate .05, min_child_samples 5, L2=1; choose 50 or 150 trees. No descriptor branch. |
| morgan_knn | Assay-weighted target mean among 1 or 5 Tanimoto neighbors; no optimized parameters, but stores all fit fingerprints and labels. |

Counts are not complete comparable capacity/storage measures: scaler statistics, tree structure and KNN storage are excluded from simple parameter counts. The comparison of generic MLP versus twin head changes architecture, scaling, output parameterization, and initialization together; it does not isolate any one of those factors.

Selection is three-fold inner CV inside fit rows. The actual runner supplies groups for both policies, so both use GroupKFold: chemical group IDs for cluster tasks, unique record IDs for random tasks (the factory's shuffled-KFold fallback is not this run's path). Each inner scaler is trained afresh on inner-training rows. Ridge alpha grid is 1/100/10,000; neural learning rates are .0001/.001, with pooled inner-OOF weighted MAE selecting learning rate and epoch 1–40 using seed 42. All final seeds refit for that selected epoch count. Neural training uses AdamW, zero weight decay, gradient clipping at 1, minibatches of 64, and weighted smooth-L1 beta .5, with ensemble/member loss weights .50/.25/.25. Weights are globally normalized for optimization. Ridge uses weighted squared loss; trees use weighted absolute loss. These are not identical training objectives or equal effective tuning capacity. More fit rows also mean more minibatch updates per epoch; learning curves describe this fixed training recipe, not label count isolated from computation.

## 3. Metric semantics and the completed results

Headline loss is `sum(w_i * abs(pred_i - y_i)) / sum(w_i)`, where `w_i = 1 / max(assay_SE_i, 0.10)`. Units are pEC50. This is inverse SE, **not inverse variance**, and is not model-uncertainty weighting. The 0.10 floor is a pre-outcome engineering cap from an existing grid, not an independently established assay noise floor. Curation weights are not multiplied into the primary objective. All 3,344 SEs are finite and positive; 1,107 are at/below the floor. The unique-compound effective weighted sample size is 2,782.34, not millions.

The following table is the arithmetic mean of the **50 fully paired fold/draw task weighted MAEs** per method/budget/split, not a standard-error estimate. It preserves outliers. The final site's `metrics.csv` instead pools weighted error/weight sums across these observations, so small last-digit differences are expected. Both aggregations are independently verified and retained in the supporting CSV. Neither treats repeated observations as independent evidence. N50/N250 are included in the supporting aggregate file.

| Method | Random N25 | N100 | N500 | Full | Cluster N25 | N100 | N500 | Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
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

### What the numerical comparisons say

1. **Frozen Nesso features support useful target-specific readout learning.** Pretrained continuous-head error moves from .6168/.6343 at N25 to .5399/.5402 at N500 and .5053/.5026 at full, for random/cluster respectively. Its paired improvement over the unchanged native point prediction is .0882/.0880 at N500 and .1229/.1256 at full. At N25 there is no dependable adaptation advantage over native: random improves by .0113, cluster worsens by .0061 on average. Adding labels is more convincing here than choosing one elaborate readout over another.

2. **Released head initialization is not a consistent source of sample efficiency.** Pretrained minus matched-random head deltas at N25/N100/N500/full are random: −.0034/−.0140/−.0005/+.0010; cluster: +.0092/−.0040/+.0012/+.0022. N50 and N250 are likewise mixed. Random N100 favors pretraining in all five draw-averaged folds, but individual task deltas range −.2431 to +.0342 and the median effect is only −.0042. Cluster N100 fold-average signs are mixed. These results do not establish equivalence, nor justify a universal pretraining advantage. They say the released head is not essential to the strong frozen-feature performance under this optimizer/grid. Nothing here compares a pretrained encoder with a random or scratch-trained encoder.

3. **The more complex interventions lack a compelling gain.** Dual minus continuous-head mean differences are within about .0031 in the displayed cells despite approximately doubling trainable parameters and fitting time. Head LoRA is worse than full continuous-head tuning by .0223/.0231 at N500, and .0034/.0049 at full. At N500, its fold-averaged deltas are positive in every fold of both policies. This is a measured accuracy/cost tradeoff for rank 4 and this training grid, not proof that all low-rank adaptation is inferior.

4. **Baseline identity materially changes the story.** Continuous-head tuning improves over this Morgan-only LightGBM by .0468/.0483 at N500 and .0340/.0402 at full. All corresponding fold-average deltas favor the head. But Morgan ridge is stronger than these trees at full: .5080/.5169, close to continuous heads, especially on random holdout. Random-full raw MAE even favors Morgan ridge (.6431 versus .6463 for the pretrained head), although weighted MAE slightly favors the head. There is no direct evidence of beating the historical descriptor-rich LightGBM system: it is not present in this matrix. A restricted seven-leaf/50-or-150-tree comparator and a weighted-mean constant are useful controls, not an exhaustive strongest-baseline search. Weighted median and a restrained native affine calibration are worthwhile missing reference points.

5. **The metric measures a reliability-weighted subset emphasis, not all-cohort accuracy alone.** Full pretrained-head raw MAE is .6463/.6523, with mean signed overprediction +.2967/+.3300, versus weighted MAE .5053/.5026. Native raw MAE is about .9150 and positive bias .7130. Better headline error does not remove systematic potency bias. Score-only SE-floor sensitivities materially change absolute levels and some tiny initialization contrasts; they are not retraining experiments. For example, pooling descriptive cell means across policies at N25, pretrained/random are .6099/.6160 with floor .05 but .6969/.6828 with floor .20. Do not infer noisy/censored/inactive status solely from low potency or a numeric fitted SE; the assay observation process and representativeness remain limitations.

## 4. What failed, and what the artifacts actually explain

### Generic representation MLP

The low-data failure is not just one unlucky row: N25 task-median weighted MAE is 1.8104 random and 1.6242 cluster, versus task means 2.6565 and 1.9236. Some tasks are much worse (random fold2/draw01 N25: 16.5680). At N100, median errors remain about .915/.913. The MLP improves with more data, but its full-reference means still trail the twin readouts.

The implementation is a direct pEC50 MLP initialized near zero, without target centering, on independently standardized cached dimensions. Twin random heads instead have the `6 - raw` output parameterization and unstandardized inputs. Consequently “MLP failed, pretrained architecture succeeded” is not an isolated pretraining experiment.

Saved-state probes give direct evidence of unstable extrapolation: in the worst N25 MLP task, a training-fitted scale is .000334 and a held-out standardized feature reaches magnitude 7,966; the associated prediction is −354.51. Its three final training losses decreased from about 4.16–4.27 to .316–.361. In a more ordinary N25 task, training loss drops to .054–.087 while held-out weighted MAE is 1.6836 and a standardized feature reaches magnitude 381. The saved predictions replay exactly. These observations rule out “it never trained” or CSV corruption as a sufficient explanation for those tasks; they show fitting plus poor out-of-sample behavior.

The learning-rate grid chooses .001 in 49/50 tasks for each split at N25, and in all N100 tasks. Some selections reach epoch 40, but many do not, including the worst N25 task (25 epochs). Thus a universally too-short schedule is not demonstrated. Small-sample scaler support, lack of regularization, direct-output initialization, width, and epoch/step selection warrant controlled ablations. No claim is made that any one fix will recover competitiveness or that scaling alone explains every MLP failure.

### Representation ridge

The official mean is highly outlier-sensitive. Random N25 mean/median are .7489/.6448; random N50 .7450/.6318. The worst N50 task (random fold2/draw09) has weighted MAE 5.3879, raw MAE 10.7086, and prediction −1704.6863 for a measured 2.69 (`nesso_4f5d960ee79593ed1c2e`). That single row contributes 1.1015 to the weighted MAE; deleting it descriptively still leaves 4.2891, so this task is not repaired by removing one point.

Here the numerical mechanism can be traced more specifically: the saved fit-only scaler has scale 5.77×10^-6 at feature column 110; its held-out standardized value is about 232,721. That one linear feature contributes an absolute 1,706.817 to the prediction. Selected alpha is 100. Exact saved-model replay returns the same predictions. This is a genuine fitted-system extrapolation vulnerability in nearly constant training dimensions, not an invented solver failure or serialization discrepancy. Other outlier tasks occur at different budgets and splits. In the broader grid, ridge selects the maximum alpha 10,000 in 92/100 N25 and 98/100 N100 tasks; regularization range and stable scaling/variance treatment need examination. These are new-method hypotheses, not grounds to delete, clip, or retrospectively replace official errors. Lower median loss is not a deployment guarantee against such failures.

## 5. Generalization and uncertainty

Random holdout keeps canonical identities separate. Chemical holdout keeps Morgan Butina .35 centroid groups separate: 1,975 groups, 1,458 singleton compounds (43.60%), largest group 36. The .50→.35 amendment was documented as structure-only and pre-outcome, and both preparation histories remain preserved. This assessment confirms the operative .35 bound artifacts; it cannot independently reconstruct when every historical decision was made.

The cluster split is **not strict novelty or chemical-series exclusion**. Cross-fold nearest-neighbor median is about .333, q95 .500, maximum .772, and 32.51% of test compounds have a full-outer-pool neighbor at or above .35. Shared scaffolds remain. These are complementary-pool similarities, distinct from the fit-subset nearest similarities saved per task. Similar random/cluster results do not prove immunity to chemical shift; grouping, composition and training membership differ. No temporal, external-target, or truly similarity-separated generalization claim is established.

There are 4,815,360 saved prediction observations but only 3,344 unique development compounds. Every budget/draw repeats outer-test compounds; outer training sets overlap, acquired subsets nest, and the three seeds share labels. Fifty task scores are **not 50 independent experiments**. The paired CSV reports actual matched deltas and ranges of draw-averaged fold effects, without t-tests, row bootstrap, or invented confidence intervals. The existing plot min–max bars are observed variation, not confidence intervals. No practical improvement margin was specified (`null` in protocol); a superiority or equivalence claim needs a justified margin and dependence-aware paired uncertainty, ideally confirmed on untouched evaluation data.

Intervals are ordinary residual split-conformal intervals from calibration labels inside N, not SE-weighted conformal guarantees. N25 has five calibration observations and its nominal 90% interval is unbounded for every method; 100% coverage there is not useful precision. For the pretrained head at N500, mean observed 90% coverage is .9043/.9037 with widths 3.5562/3.5120 pEC50; at full it is .9016/.8988 with widths 3.0121/3.0597. These broad marginal intervals and mean coverage do not establish conditional calibration, chemical-shift coverage, or safety thresholds. Seed SD is a different, uncalibrated quantity. The 25 task/coverage-level floating-boundary discrepancies between endpoint comparisons and the runner's residual predicate replay exactly without widened intervals.

## 6. Training cost and actual upstream feasibility

The factory runs on CPU and caps numerical threads at two in the actual implementation, even though the protocol permits four. Summed factory `fit_seconds` is 3.803 hours over all 7,200 tasks. This includes inner selection, final fits, prediction and in-memory serialization replay; it is not total queue wall time or end-to-end deployment cost. Task completion timestamps span two dates and resumed invocations; summing factory durations omits preparation, task writing, report generation and interruptions. Cached-feature acquisition and historical upstream extraction time are excluded.

Median factory seconds across both split policies (including three final neural seeds):

| Method | N25 | N500 | Full |
|---|---:|---:|---:|
| head_pretrained | .434 | 3.754 | 18.178 |
| head_random | .446 | 3.526 | 17.134 |
| dual_head | .863 | 7.360 | 35.892 |
| head_lora | .370 | 2.652 | 12.840 |
| morgan_ridge | .018 | .316 | 5.106 |
| morgan_lightgbm | .039 | .256 | .857 |

Head LoRA reduces trainable count by about 38.5-fold relative to full continuous heads, not stored parameter count or fitting time by that factor. Its upstream representations still require the same extraction.

The separate real-input pilot adapts 12,288 rank-4 parameters in first-member `esm_proj.1`, `esm_proj.3`, and `pairformer_stack.layers.0.transition_z.fc1`. It uses one development record, two SGD steps, and a synthetic engineering target of detached native continuous output plus one, **without an assay label**. The documented native/differentiable-path, zero-update, nonzero-gradient, frozen-weight, and adapter-reload tests establish a working gradient path. This assessment read that evidence but did not rerun the GPU pilot. The second affinity member was not adapted and no PXR learning curve was scored. It is not sample-efficiency evidence, end-to-end fine-tuning evidence, or a measured speed/accuracy comparison with the cached heads.

## 7. Provisional recommendation and follow-ups that could change it

For a current PXR predictor with already available cached vectors, use the twin continuous-head family as the provisional Nesso route, retaining matched-random initialization and Morgan ridge as live comparators. Released initialization is a reasonable operational default, not an established accuracy advantage. At N25 retain the native anchor and disclose that adapting 20 fit labels can worsen performance; do not promise improved predictions or finite 90% intervals. At larger budgets, target fitting is supported descriptively. Do not add binary-head complexity without a measured benefit; choose head LoRA only if the modest fitting-cost reduction is worth the observed error penalty. Avoid deploying the current generic scaled MLP/ridge recipes without addressing their extreme extrapolation risk.

Priority follow-ups, not authorization or a new implementation plan:

1. **Establish what counts as useful predictive improvement.** Specify deployment chemistry, a practical pEC50 error margin, and assay/SE/censoring interpretation. Analyze paired differences with compound/chemical-group dependence preserved and draw variability separated; confirm any selected recipe on untouched prospective, temporal or external chemistry rather than repeatedly optimizing this completed test panel.
2. **Make the comparator question fair before increasing model depth.** Restore the historical richer comparator under the same budgets/splits, add the absolute-loss-optimal weighted-median anchor, and test a restrained scalar affine readout. Keep Morgan ridge. This can change whether Nesso offers worthwhile incremental value and justify its extraction cost.
3. **Resolve the demonstrated readout vulnerabilities.** Compare train-only stable/variance-floor or unscaled feature handling, adequate regularization grids and output centering with controlled architecture/initialization/optimization budgets. Preserve original scores; use new development tests rather than selecting fixes on known outliers. This separates representation utility from preprocessing/optimization weaknesses.
4. **Only then test genuine upstream adaptation as its own hypothesis.** Compare a both-member upstream adapter with the strongest stable frozen readout on paired labels, including all tuning/calibration labels, untouched tests, and extraction/storage/CPU/GPU costs. The completed engineering pilot removes one feasibility blocker, not the need for scientific validation.

The study therefore answers “which of these frozen-feature readouts should we provisionally use for this PXR assay?” much better than “how should Nesso-1 be trained?” It does not establish the best upstream modules, pretraining objective, multitask strategy, adapter rank, full-model fine-tuning recipe, or cross-target training policy.

## 8. Independent verification and evidence inventory

Independent Python checks reconciled the Cartesian product, `task_manifest.json`, on-disk task directories, final task table, and all 600 fully paired cells: **7,200/7,200 tasks, no missing/extra cells**. All 43,200 task-output SHA-256 checks passed (28,680,058,008 bytes). All 7,200 label-access files match the prepared fit/calibration/test identities and budgets. The 22 bound source files match both current files and immutable source snapshots; all nine bound input files match their hashes.

Beyond the requested metric spot checks, this assessment reread **all 7,200 prediction CSVs / 4,815,360 rows** with round-trip float parsing. Inverse-SE weights, raw MAE and headline weighted MAE agree; maximum task-table weighted-MAE discrepancy is exactly zero. All saved interval endpoints and residual-based coverage replay. Pooled site metric agreement is within 2.23×10^-16. Four representative trusted-local `model_state.pt` systems (including ridge and MLP failures and a cluster N500 continuous head) independently replay all saved seed predictions with maximum error zero in the specified cheminformatics Python environment. No fits were run.

Canonical sources inspected: the six requested files in `experiments/20260905_low_data_adaptation/`; `src/nesso_pxr/low_data_models.py`; the actual runner and old/revised interval verification paths; final `site/low-data-verified/{task_metrics.csv,metrics.csv,summary.json,index.html}`; and the bound run's protocol, source/input bindings, prepared identities, task outputs and fitted states. Documentation contains historical partial-run language (including the HTML title and factory handoff's unexecuted-results caveat); the final artifacts supersede those operational-status statements, not their scientific boundaries.

Supporting files beside this report:
- `independent_assessment_aggregates.csv`: all methods, both splits and all six budgets; paired-cell mean/median/range, pooled weighted MAE, raw MAE, bias, coverage and cost.
- `independent_assessment_paired.csv`: six declared pairwise contrasts at every budget/split, actual cell delta distributions and fold-mean ranges; no inferential intervals.
- `independent_assessment_diagnostics.csv`: per-task prediction spread/extremes and explicitly descriptive single-worst-row sensitivity for five Nesso readout/anchor families inspected.
- `independent_assessment_verification.json`: verification counts, source table hashes, representative state/scaler probes, selection summaries, runtime and metric semantics.
