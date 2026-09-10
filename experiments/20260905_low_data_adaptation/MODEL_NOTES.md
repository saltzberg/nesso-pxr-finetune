# Model factory handoff

Audited 2026-09-05. This finishes the existing factory rather than replacing it. Scientific matrix execution, held-out scoring, protocol changes, and upstream representation adaptation are outside this handoff.

## Prediction construction and fitted scope

Prepared immutable features → method-specific fit-only preprocessing/readout → arithmetic mean over final seeds → raw predictions and population seed SD. Calibration/interval construction remains the runner's responsibility; the factory does not apply interval offsets, clipping, or target calibration outside its declared readout.

The prepared arrays inspected contain 3,344 aligned rows: member1 and member2 each 384D, concatenated representation 768D, Morgan 2,048 binary bits, scalar features 9D, native continuous 1D. The scalar columns are the two member and ensemble pIC50-equivalent values, two member and ensemble binary logits, and two member and ensemble binary probabilities. All are reconstructed without target fitting in preparation.

| Method | Fitted parameters per seed | Stored neural parameters per seed | Trained/frozen branches |
|---|---:|---:|---|
| weighted_mean | 1 | — | Inverse-SE-weighted target mean |
| native_continuous | 0 | — | Supplied frozen published two-member mean; no target fitting |
| scalar_ridge | 10 | — | Weighted fit-only scaler and 9 coefficients + intercept |
| repr_ridge | 769 | — | Weighted fit-only scaler and 768 coefficients + intercept |
| repr_mlp | 98,561 | 98,561 | Weighted fit-only scaler; all 768–128–1 MLP parameters trained |
| head_random | 592,130 | 592,130 | Both random 384–384–384–1 continuous MLPs fully trained |
| head_pretrained | 592,130 | 592,130 | Same architecture as random, initialized from both released continuous MLPs; fully trained |
| dual_head | 1,184,268 | 1,184,268 | Both continuous MLPs, binary-score MLPs, binary-logit linear maps, and scalar fusion scale/offset parameters fully trained |
| head_lora, rank 4 | 15,368 | 607,498 | Only A/B adapters in all three Linear layers of each continuous MLP trained; 592,130 base parameters frozen |
| morgan_ridge | 2,049 | — | Unscaled binary bits; 2,048 coefficients + intercept |
| morgan_lightgbm | Fit-dependent leaf count | — | Weighted Morgan-only regression_l1 trees; no descriptor branch |
| morgan_knn | 0 optimized | — | Stores fit fingerprints, targets, normalized assay weights; weighted mean of k nearest Tanimoto neighbors |

Counts are per final seed, not multiplied by ensemble size. Classical counts exclude scaler statistics, tree split structure, and KNN storage; these are not comparable complete capacity estimates. Nonneural methods fit once and repeat the identical model/predictions across declared seeds, so their ensemble SD is zero.

The two-head raw output is converted via `6 - raw_affinity`, then averaged. This is a pIC50-equivalent native affinity anchor, not native cellular pEC50. For dual_head each member's raw prediction is `continuous + 1.5*tanh(binary_scale*binary_logit + binary_offset)` before conversion. Scale/offset begin at zero, so the binary branch first receives nonzero gradients after the gate changes; the two-step branch-update test verifies this. Head LoRA initializes B to zero, A randomly, with alpha/rank = 1. Its zero-update function agrees with the pretrained continuous head. It is not the separate genuine Pairformer adapter pilot and never changes the cached representation.

## Inner-CV isolation

Only fit rows and their labels/SEs reach fitting. Each candidate receives fresh inner-train scalers and models; weighted StandardScaler fits use only that fold's rows and normalized inverse-SE weights. Final scalers refit on the complete fit subset only. The new instrumented tests capture every StandardScaler.fit call for scalar_ridge, repr_ridge and repr_mlp, compare its matrix and weights against the exact declared inner-training indices, and check weighted means. Altering prediction features does not change selection.

Neural learning rate and epoch are jointly selected by pooled inner-OOF weighted MAE using the first declared seed. Every final seed refits from its own initialization for the selected epoch count. Smooth-L1 beta is fixed to 0.5, ensemble/member loss weights are 0.50/0.25/0.25, AdamW weight decay is zero, and gradient norm clipping is 1. Global fit-weight normalization is preserved across minibatches. Supplied groups use GroupKFold; too few groups fail explicitly, while reduced-but-feasible fold counts are disclosed. Without groups, shuffled KFold assumes caller-verified unique identities. No global PCA, validation-fitted scaler, unsupported estimator substitute, or weight-contract import fallback remains.

## Saved-state/replay contract

The API already existed; this audit bounds and validates it rather than introducing a second estimator interface:

```python
from nesso_pxr.low_data_models import load_model, predict_model
state = load_model("model_state.pkl")  # trusted local pickle only
out = predict_model(state, prediction_features, batch_size=1024)
# out: pred (rows,), ensemble_std (rows,), seed_predictions (seeds, rows)
```

A parent `pickle.dump(result['model_state'], file)` is compatible. `save_model` is optional. Artifact schema_version=1 includes method, ordered seeds, one fitted state per seed, CPU thread cap, selection, and method-required feature widths. Neural states include complete frozen and fitted weights plus scalers; sklearn states include fitted intercepts and scalers; dual fusion offsets/scales are torch parameters in the model. No original checkpoint path is needed for replay. New artifacts validate relevant widths only; unrelated cached feature arrays need not be supplied. Legacy schema-1 artifacts lacking feature_shapes remain supported. Labels/unknown feature keys are rejected.

Replay defaults to 1,024 prediction rows per batch and caps CPU threads at two, restoring torch's prior thread setting on exit. KNN distance intermediates are bounded by batch rows × stored training rows; output arrays still require seeds × prediction rows storage. Identical batch shape gives bitwise pickle replay for all three outputs. Changing batch size may change float32 GEMM rounding; tests use numerical tolerance for that case. Original unbatched predictions from older artifacts larger than 1,024 rows therefore need matching `batch_size` for an exact historical comparison. Pickles are executable trusted-local artifacts, not safe exchange files or a stable cross-version format. Preserve package/source versions in the parent manifest. This API returns raw model outputs, not externally calibrated intervals.

The actual integrated runner writes `model_state.pt` using `torch.save`, not the optional standalone pickle helper. Its verified replay path is `state = torch.load(path, map_location='cpu', weights_only=False)` followed by `predict_model(state, features)`, and is only appropriate for these trusted-local artifacts. Parent replayed all 24 real N25 model/split artifacts with maximum absolute seed-prediction error zero; the receipt is `parent_replay_verification.json` in the operative run directory. The `.pkl` example above describes the standalone factory helper, not the runner's filename.

All factory computation is CPU-only; `peak_vram_bytes=0` is now explicit, not a missing key or GPU measurement. Thread limits are process-global and this API is not designed for concurrent numerical threads in one process.

## Measured verification

Runtime: existing cheminformatics Python, `PYTHONPATH=src:.`, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`; torch 2.12.0+cu130, NumPy 2.1.3. No installation, upstream extraction, matrix run, or held-out metric comparison occurred.

- Initial factory suite: 18 tests passed.
- After changes: combined model/runner/contract suite **58 passed in 5.09 seconds**. Targeted `ruff format` and `ruff check --fix` pass for the two owned Python files.
- All 12 method tests verify deterministic fitting, complete seed/mean/SD disk replay, direct-pickle compatibility, bounded-batch tolerance, empty prediction rows, and prediction-feature isolation.
- Exact neural counts above are both runtime-enumerated and asserted. Adapter tests verify nonzero gradients/updates and exact frozen-base invariance. Dual tests verify updates in both members' continuous, binary-score, binary-logit, scale and offset branches.
- Real native parity: prepared rows 0–127, released checkpoint heads, no labels. Pretrained, dual zero-update, and rank-4 LoRA zero-update each agree with prepared native continuous values at max absolute difference **4.76837158203125e-7**, below absolute tolerance 2e-6. Building/checking these models took **0.02023 seconds**. This is prepared-reference agreement, not a fresh upstream encoder rerun; converting before versus after the member mean can differ by float32 rounding.
- Exactly one cheap real N25 head_lora fitting smoke: prepared random split, outer_fold=0, draw=0; **20 fit labels**, **5 calibration-feature-only prediction rows**, no calibration/test labels read. Only fit rows were loaded from the label CSV using row filtering. Smoke overrides were epochs=2, inner_folds=2, learning-rate grid=[0.0001], seeds=(42,), threads=2; these are engineering overrides, not the full protocol matrix. Selected learning rate 0.0001 and epoch 2; **0.72805 seconds** factory time, **15,368** trainable parameters, **0 VRAM bytes**. Predictions were finite and every output replayed bitwise from a temporary on-disk pickle. Temporary artifacts were cleaned; no scored result was published.

Smoke fit row indices: `289,2642,625,1970,1321,3323,1963,796,2856,979,3210,3168,1944,907,1895,912,323,3014,215,733`.
Prediction-only calibration row indices: `2885,1502,853,2993,1591`.

Verification command, from repository root with the runtime environment above:

```sh
python -m pytest tests/test_low_data_models.py tests/test_low_data_runner.py tests/test_low_data_contract.py -q -o addopts=''
python -m ruff check src/nesso_pxr/low_data_models.py tests/test_low_data_models.py
```

## Remaining boundaries

No full-protocol real neural fit, three-seed real ensemble, chemical-group real smoke, held-out performance, or full-matrix timing is established here. Fixture fits and replay cover every method and multiple seeds, but they are not evidence of scientific benefit. The single real smoke is an engineering check only. The factory does not prove cache provenance or row identities by itself: the preparation and parent manifests must bind them. Input feature widths are bound, not feature-content hashes. The exact historical native head source and binary-output provenance remain preparation responsibilities. Source changes require a fresh parent run/source binding before further matrix execution.
