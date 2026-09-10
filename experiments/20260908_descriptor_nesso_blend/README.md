# Descriptor + frozen-Nesso prediction: one-fold CPU pilot

Created and last edited: 2026-09-08.

This exploratory pilot combines the existing ligand-descriptor LightGBM predictor with a ridge readout of frozen Nesso representations. It tests incremental predictive value, not new neural fine-tuning.

## Result

On the preselected outer development fold 0, the blend improves weighted MAE from 0.5029 to 0.4852 (3.51%), raw MAE from 0.5619 to 0.5544, and Spearman from 0.6352 to 0.6722. All methods score the same 669 held-out compounds.

| System | Weighted MAE | Raw MAE | Spearman |
| --- | ---: | ---: | ---: |
| Descriptor LightGBM | 0.5029 | 0.5619 | 0.6352 |
| Frozen Nesso vectors + ridge | 0.5283 | 0.6181 | 0.6029 |
| Training-selected blend | 0.4852 | 0.5544 | 0.6722 |

The training-selected mixture is approximately 60.17% descriptor prediction and 39.83% frozen-Nesso ridge prediction. These are predictive combination coefficients, not biological source-reliability probabilities.

## Where the improvement occurs

Among the 476 compounds with observed pEC50 >=4, weighted MAE improves from 0.4608 to 0.4305. Among the 193 compounds below 4, it worsens by 0.0459. The respective contributions to the overall weighted-MAE change are -0.02527 and +0.00761, summing to -0.01766. The higher-potency group carries 83.42% of evaluation weight.

Thus this gain is not a repeat of the earlier low-potency correction story: this blend improves the higher-potency stratum while sacrificing lower-potency accuracy. It uses frozen representations, not saved LoRA outputs. The pilot does not establish whether its gain is specifically molecular-weight-related.

## Fitted and frozen stages

    Canonical molecular descriptors → fitted LightGBM
    Frozen two-member Nesso representations → fitted ridge
    Genuinely nested inner-OOF predictions → one fitted convex mixture weight
    Full-FIT refits + fixed mixture → held-out pEC50 predictions

The N500 allocation preserves the existing 400 FIT, 100 CAL and 669 TEST identities. Three grouped meta folds occur entirely within FIT; each meta-training subset independently selects its descriptor/ridge settings through further grouped cross-validation. All learned transformations remain inside their relevant training partitions. The mixture minimizes inverse-assay-SE weighted absolute error on the 400 inner-OOF predictions. No CAL or outer-test labels select it. Final base models refit using all 400 FIT compounds with their own grouped selection. CAL predictions are saved without labels; this is a point-prediction pilot, not a new interval analysis.

The descriptor recipe and frozen-vector ridge recipe are unchanged from the matched controls. Fresh fits reproduce their historical outer predictions. Native Nesso features were reused with source hashes and identity mappings checked; no new Nesso inference occurred.

## Interpretation and stop

This is encouraging evidence of complementary information in frozen Nesso features, despite their weaker standalone predictor. It warrants consideration of the same locked recipe on the remaining folds, not a broad architecture sweep.

It is one preselected reused development fold, one subset draw and one initialization. There is no fresh holdout, retraining-variability estimate, resolved cross-fold superiority claim or clinical activation-liability validation. Historical blending has already been explored; this is a current low-data, strictly nested follow-up, not a first-ever blend result. The approved pilot stops here. No additional folds were launched.

## Execution and verification

The fitted run recorded 19.36 seconds using two CPU threads and no GPU, excluding engineering, source discovery and reporting. Eight saved base-model systems include six meta-fold models and two full-FIT models; their internal tuning fits are additional work included in the runner time.

The subagent completed fitting, independent verification and 67 passing tests, then reached its agent time limit before writing the final narrative. Parent recovery reran `verify.py` successfully without fitting. It replayed all eight saved systems with maximum prediction discrepancy zero, audited 68 preprocessing-transform calls, checked source/output hashes and exact roles, and independently recovered the mixture weight with linear programming. The solver emits a warning that its thread setting is forwarded to HiGHS; verification passes. A separate repeated-run test refused refitting and preserved existing artifact bytes.

Run the read-only verifier from the repository root with the existing scientific environment:

    PYTHONDONTWRITEBYTECODE=1 /home/dan/swr/miniconda3/envs/cheminf/bin/python experiments/20260908_descriptor_nesso_blend/verify.py

This is a local-runtime command, not a claim of clean-environment portability.

- [Pre-fit protocol](protocol.json)
- [Runner](run.py), [independent verifier](verify.py), [tests](test_pilot.py)
- [Metrics](../../artifacts/experiments/descriptor_nesso_blend_20260908/metrics.json)
- [Test predictions](../../artifacts/experiments/descriptor_nesso_blend_20260908/test_predictions.csv)
- [Verification](../../artifacts/experiments/descriptor_nesso_blend_20260908/verification.json)
- [Blend selection](../../artifacts/experiments/descriptor_nesso_blend_20260908/blend_selection.json)
- [Descriptive potency decomposition](../../artifacts/experiments/descriptor_nesso_blend_20260908/descriptive_stratum_decomposition.json)
