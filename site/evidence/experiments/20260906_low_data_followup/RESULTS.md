# What the completed low-data follow-up establishes

Created: 2026-09-06. Last edited: 2026-09-06.

We tested whether better readout preprocessing, a stronger ligand-only comparator and stricter chemical separation change the practical case for adapting Nesso to PXR cellular activation.

## Model flow and scope

- Nesso methods: fixed protein/ligand preparation -> frozen two-member Nesso vectors -> fitted ridge/MLP or adapted released continuous heads -> mean neural seed prediction -> intervals calibrated within the stated label budget.
- Descriptor comparator: canonical SMILES -> fixed RDKit2D, Mordred2D and Morgan features -> training-only feature reduction and fitted LightGBM -> within-budget interval calibration.
- Strict evaluation: original chemical test folds retained; fitting and calibration pools purged so every pool/test Morgan similarity is below 0.35.

The cofolding trunk and affinity representation were not adapted in these matrices. The descriptor comparator is not an exact recreation of the larger historical feature universe. These are retrospective experiments on the same development cohort.

## Main conclusions

1. **Descriptor LightGBM has the lower overall weighted error from N250 upward.** At N500, its mean-task weighted MAE is 0.497 versus 0.542 for the adapted pretrained head. The paired difference is -0.0446, with a conditional chemical-cluster reweighting interval of approximately -0.056 to -0.032. The difference has the same direction averaged within all five outer folds. These intervals condition on the existing fitted systems and inspected cohort; they do not establish prospective superiority or quantify training-population uncertainty.

2. **That overall advantage is not uniform across potency.** At N500, pooled within-stratum weighted MAE below pEC50 4 is 0.867 for LightGBM versus 1.281 for the Nesso head. At pEC50 >=4, the ordering reverses: 0.425 versus 0.397. Thus much of LightGBM's aggregate improvement comes from reducing overprediction in the lower-potency group. This is a descriptive decomposition, not permission to change the endpoint or choose a model after seeing a test compound's true potency.

3. **Both systems still compress the high-potency tail.** Among 56 unique compounds with observed pEC50 >=6, pooled prediction bias at N500 is -1.271 for LightGBM and -1.231 for the adapted head. Repeated draws do not turn these into 560 independent compounds. Neither low average error nor the lower overall LightGBM score establishes accurate tail prediction.

4. **Small label budgets favor retaining the Nesso starting point.** At N25 the native output and adapted heads have nearly the same primary error. At N100, adapted heads have lower weighted MAE than the descriptor comparator, although the latter already has lower raw MAE. Pretrained head initialization has a modest advantage at intermediate budgets but not a consistent advantage everywhere. This does not test the value of upstream pretraining.

5. **Stable readouts are credible controls, not a solved optimization problem.** Stable scaling and target centering substantially improve the original MLP recipe; added weight decay contributes very little. Stable ridge is competitive at larger budgets. Its small N500 advantage over the pretrained head spans zero under the conditional reweighting analysis; do not present it as a reliable model promotion.

6. **Stricter separation does not make novelty irrelevant.** Removing above-threshold neighbors had little effect on the aggregate scores, but the strict N500 predictions with actual nearest-fit similarity below 0.2 still have higher error. They carry only about 1.9% of evaluation weight and their membership varies with the training draw. The evidence supports this particular retrospective separation audit, not arbitrary new-series or new-target generalization.

7. **Nominal interval coverage can conceal unhelpful intervals.** At N25, 90% intervals are unbounded because the within-budget calibration set is too small. At N500, empirical 90% coverage is about 89% for the main systems, but mean widths remain broad: approximately 2.58 pEC50 units for LightGBM and 3.39 for the pretrained head. Chemical-shift coverage is not guaranteed.

## Cost and completion

All 8,100 new tasks completed without recorded failures. This analysis additionally replays 1,200 original comparator tasks: 6,219,840 dependent prediction rows in total. It recomputes scores and interval predicates from round-trip saved predictions. Parent verification checked 27,918 source hashes, nine analysis outputs, and agreement of every new task's weighted MAE with the previous completion tables.

Saved strict N500 mean task fitting times are about 7.95 seconds for descriptor LightGBM, 4.66 seconds for the pretrained head and 0.29 seconds for stable ridge. These are runner-recorded fitting times, not whole-pipeline timings. One-time feature extraction and peak VRAM are not recoverable from these task metrics and are explicitly unmeasured here.

## Consequence for the next experiment

The next useful test is upstream affinity-representation adaptation against the exact same head-only training, with descriptor LightGBM retained as a strong context comparator. Judge the primary weighted error together with the already-declared tail, bias and interval diagnostics; do not silently replace the primary objective. First extend the existing real-input engineering pilot to both members, then measure extraction/storage/GPU cost before committing to the main matrix.

Detailed conditional comparisons, novelty/stratum coverage, compression, intervals and costs are in `artifacts/experiments/low_data_followup_20260906/final_analysis/`. The analysis performs no new model fits and makes no untouched-validation or automatic-promotion claim.
