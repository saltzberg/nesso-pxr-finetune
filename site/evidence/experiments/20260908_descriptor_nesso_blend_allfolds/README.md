# Locked descriptor + frozen-Nesso blend: all five development folds

Created and last edited: 2026-09-08.

The locked recipe improves inverse-assay-SE weighted MAE in all five reused development folds, including all four newly fitted folds. The improvement is modest and comes with worse errors below pEC50 4, not uniform improvement across chemistry or potency.

## Pooled result: 3344 unique compounds once per method

| Method | Weighted MAE | Raw MAE | Spearman |
| --- | ---: | ---: | ---: |
| RDKit/Mordred/Morgan descriptor LightGBM | 0.500588 | 0.599636 | 0.656863 |
| Frozen native Nesso vectors + stable ridge | 0.534468 | 0.666603 | 0.608439 |
| Independently training-selected convex blend | **0.486790** | **0.594731** | **0.686392** |

The blend reduces pooled weighted MAE by 2.7563% relative to descriptors. Its standalone Nesso ridge component is worse than descriptors, while the combination adds predictive value on these development evaluations. Mixture coefficients are prediction weights, not biological source-reliability probabilities.

Unweighted means across the five folds (macro-fold scores, not pooled scores):

| Method | Weighted MAE | Raw MAE | Spearman |
| --- | ---: | ---: | ---: |
| Descriptor | 0.500542 | 0.599642 | 0.654359 |
| Ridge | 0.534540 | 0.666606 | 0.607419 |
| Blend | 0.486779 | 0.594735 | 0.684199 |

## Fold-by-fold results

Fold 0 is immutable pilot reuse; folds 1–4 are new fits. Test counts are 669, 669, 669, 668 and 669. Every method has the exact same test IDs within each fold and the same 3344-ID union without repeats.

| Fold | Method | Weighted MAE | Raw MAE | Spearman | >=4 weighted MAE | <4 weighted MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Descriptor | 0.502895 | 0.561888 | 0.635200 | 0.460829 | 0.714537 |
| 0 | Ridge | 0.528261 | 0.618140 | 0.602878 | 0.448729 | 0.928402 |
| 0 | Blend | 0.485238 | 0.554354 | 0.672153 | 0.430541 | 0.760427 |
| 1 | Descriptor | 0.486880 | 0.597619 | 0.636914 | 0.418419 | 0.883522 |
| 1 | Ridge | 0.517456 | 0.663661 | 0.611623 | 0.422178 | 1.069465 |
| 1 | Blend | 0.474072 | 0.600039 | 0.684127 | 0.392497 | 0.946687 |
| 2 | Descriptor | 0.479875 | 0.608242 | 0.706905 | 0.408432 | 0.870121 |
| 2 | Ridge | 0.535001 | 0.694738 | 0.640258 | 0.431381 | 1.101002 |
| 2 | Blend | 0.472830 | 0.602037 | 0.723759 | 0.397756 | 0.882906 |
| 3 | Descriptor | 0.534288 | 0.620151 | 0.627310 | 0.467567 | 0.846642 |
| 3 | Ridge | 0.553797 | 0.678480 | 0.598169 | 0.448479 | 1.046850 |
| 3 | Blend | 0.515120 | 0.608891 | 0.656553 | 0.438711 | 0.872829 |
| 4 | Descriptor | 0.498774 | 0.610308 | 0.665467 | 0.415858 | 0.891986 |
| 4 | Ridge | 0.538187 | 0.678014 | 0.584164 | 0.444812 | 0.981002 |
| 4 | Blend | 0.486636 | 0.608356 | 0.684404 | 0.400160 | 0.896734 |

| Fold | Nesso weight | Blend minus descriptor weighted MAE |
| --- | ---: | ---: |
| 0, reused | 0.3982750014 | -0.0176570509 |
| 1, new | 0.3988772885 | -0.0128080682 |
| 2, new | 0.1845099507 | -0.0070451817 |
| 3, new | 0.2806483922 | -0.0191683121 |
| 4, new | 0.3125619971 | -0.0121374887 |

All five weighted-MAE differences improve; no ties, regressions or zero Nesso weights occurred for that primary comparison. Zero weights remained allowed and endpoint/identical-prediction selector tests passed. No outcomes were dropped. Raw MAE worsens in fold 1 despite the weighted-MAE improvement. The <4 weighted and raw errors worsen in every fold. Stratum rank changes are mixed; the complete numerical values remain in `metrics.json`.

## Potency trade-off, pooled

| Observed stratum | N | Method | Weighted MAE | Raw MAE | Spearman |
| --- | ---: | --- | ---: | ---: | ---: |
| >=4 | 2308 | Descriptor | 0.434197 | 0.422383 | 0.356726 |
| >=4 | 2308 | Ridge | 0.438975 | 0.429683 | 0.337265 |
| >=4 | 2308 | Blend | 0.411839 | 0.398909 | 0.402983 |
| <4 | 1036 | Descriptor | 0.840152 | 0.994518 | 0.294580 |
| <4 | 1036 | Ridge | 1.022874 | 1.194412 | 0.270413 |
| <4 | 1036 | Blend | 0.870134 | 1.030984 | 0.310710 |

This repeats the pilot's higher-potency improvement/lower-potency sacrifice. It does not establish that lower-potency observations are noise or that the model is universally preferable.

## Exact recipe and reuse

Approval was to “redo with locked recipe to see if it holds.” The derived [protocol](protocol.json), approval, full pilot byte inventory and fold-0 reuse map were persisted before fitting. The wrapper imports the original pilot functions rather than rewriting their scientific implementation; it changes the selected fold and output path, while retaining original helper/source identities in each new binding.

Each fold uses the existing strict-chemical N500 draw0 allocation: 400 FIT and 100 CAL. Three grouped meta folds inside FIT produce 400 genuinely held-out inner predictions per base method. Each meta-training subset independently selects hyperparameters using the original deeper three-group-fold procedure. The original six-candidate descriptor LightGBM recipe includes training-only reduction; the ridge grid is [1, 100, 10000, 1000000], using stable training-only scaling and fresh frozen native two-member Nesso vectors. All transforms are instrumented against actual training indices. Final base models independently rerun grouped selection on the full 400 FIT rows.

The scalar Nesso weight minimizes exact convex weighted absolute error on inner-OOF predictions over [0,1], with endpoints, all clipped residual/difference breakpoints and lowest-weight tie-breaking. There is no intercept, prediction clipping, additional model or changed grid. The pilot's learned weight or hyperparameters were not frozen across folds: meta-model selected settings vary, as saved in each audit.

CAL outcomes are unavailable to fit/selection calls and remain unused; label-free CAL predictions are retained. Outer-test outcomes are unavailable to fit/selection calls and are scored only after each fold's models and blend are fixed. Assay weights are inverse max(reported SE, 0.1). These are point predictions, not a new uncertainty-calibration study.

Fold 0's eight saved systems, original bindings and predictions remain in the original artifact directory, byte-identical. No fold-0 refit or rewritten pilot verification was performed. Four new folds contain 32 new saved systems: 24 meta-fold systems and eight full-FIT systems. Internal hyperparameter fits are additional work within those systems. All newly fitted final base predictions match the historical matched descriptor and fresh-feature ridge controls exactly. Historical model artifacts are consulted only after the new fold's fits and blend selection.

## Verification and limitations

- Eight focused tests passed before fitting: selector endpoints/invalid inputs/independent LP, group nesting, stable preprocessing, exact helper identity, role matrix, serialization and no-refit guard.
- Actual independent disk replay passed for all 40 saved systems, with maximum absolute prediction discrepancy **0**.
- 340 transform calls audited; exact row and chemical-group roles checked; blend objectives and weights independently recovered by linear programming.
- Source and output hashes verified, including immutable pilot source/artifact bytes and preserved historical bindings.
- All five existing-output guards refused refits with the fitting function replaced by an exception; saved bytes remained unchanged.
- Compound predictions reconstruct the same 3344 unique IDs once for each method. Pooled, macro, per-fold and both potency-stratum metrics are saved at full precision. CSV verification uses round-trip floats.
- Four new fold runs recorded 83.296 seconds in total, using two CPU threads, no GPU/cloud and no new native inference. Engineering and reporting time is additional.

No execution failures occurred. The existing SciPy/HiGHS warning says the `threads=2` option is forwarded verbatim; solver verification passes. The pilot's reviewed historical source drift handling remains unchanged: preserved snapshots are verified, with the previously reviewed controls zip-strictness difference retained. As in the pilot, extracted features, identity map, stage and source bindings are verified; bulk upstream tensor payloads are not reread, and their producer hash inventory is retained.

These are reused development folds and continuation was approved after observing the pilot, not a fresh prospective holdout or a preregistered superiority test. One draw and one seed do not estimate retraining variability; overlapping training sets make the fold results dependent. The existing chemical grouping is not proof of absence of close analogues: test folds contain 394–396 groups, with 291–293 singleton compounds each. No broader series-holdout or mechanistic claim follows. The result supports consistent incremental predictive value under this locked low-data recipe, with a clear potency trade-off; it does not authorize automatic deployment, a new sweep or publication.

## Files and commands

Artifact root: `artifacts/experiments/descriptor_nesso_blend_allfolds_20260908/`.

- `all_test_predictions.csv`: complete paired compound ledger, fold and reused/new provenance.
- `metrics.json`: full-precision pooled, macro-fold, per-fold, >=4/<4 weighted/raw MAE and rank; blend weights and paired primary differences.
- `fold1/`–`fold4/`: models, nested roles, audit calls, selections, predictions, baseline parity, immutable snapshots and completion hashes.
- `verification.json`, `verification_fold0.json`–`verification_fold4.json`: actual replay results written only in this new directory.
- `protocol.json`, `complete.json`, `failures.json`, `run.log`, `verify.log`: approval/bindings, outputs, failures and real execution logs.

From `/home/dan/projects/nesso-finetune/PXR`:

```bash
export PYTHONDONTWRITEBYTECODE=1
/home/dan/swr/miniconda3/envs/cheminf/bin/python -m unittest discover -s experiments/20260908_descriptor_nesso_blend_allfolds -p 'test_*.py' -v
/home/dan/swr/miniconda3/envs/cheminf/bin/python experiments/20260908_descriptor_nesso_blend_allfolds/run.py --run
/home/dan/swr/miniconda3/envs/cheminf/bin/python experiments/20260908_descriptor_nesso_blend_allfolds/run.py --verify
```

`--prepare` was run once before fits and refuses existing protocol/output paths. `--run` now checks and reuses the four completed new folds without fitting; an incomplete existing attempt requires review rather than an automatic refit. `--verify` never fits and writes only derived verification/aggregate files; it does not modify pilot artifacts. These commands use the existing local scientific interpreter, not a claim of clean-environment portability. No root README or site edits were made.
