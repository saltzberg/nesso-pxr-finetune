# Stable frozen-vector readouts

Retrospective controlled preprocessing/readout comparison on the original development-only acquired-label grid. The recipe was frozen in `protocol_readouts.json` before follow-up predictions; original-study failures were already inspected. No smoke score is used to change choices, no output clipping, no upstream adaptation, no superiority margin or independence claim.

## Exported API (strict-lane integration contract)

Load `code/readouts.py` through `importlib.import_module("experiments.20260906_low_data_followup.code.readouts")`, or an ordinary importlib file loader. No global path mutation or original-factory monkeypatch is required.

```python
METHODS = (
    "repr_ridge_unscaled",
    "repr_ridge_stable",
    "repr_mlp_unscaled",
    "repr_mlp_stable",
    "repr_mlp_stable_centered",
    "repr_mlp_stable_centered_regularized",
)

fit_predict(method, train_features, y_train, assay_se_train, test_features,
            *, config, checkpoint_path, seeds=(42, 43, 44)) -> dict
predict_model(model_state, features, *, batch_size=1024) -> dict
```

The strict panel should import **`repr_ridge_unscaled` and `repr_ridge_stable`**, not historical `repr_ridge`. The factory takes the original feature dictionary (`member1`, `member2`, optional original keys including `groups`) with 384 columns per member. Extra keys, including `descriptors` or label keys, are rejected: strip descriptor-only features before this factory. `checkpoint_path` is accepted for API compatibility but never opened. Only fit labels enter the factory. Supply fit-only chemical groups under `groups`; group-starved inner CV fails rather than changing grouping policy.

Use the frozen readout config values: `se_floor=.1`, `inner_folds=3`, `grids.ridge_alpha=[1,100,10000,1000000]`, `grids.neural_learning_rate=[.0001,.001]`, `epochs=40`, `batch_size=64`, `neural_huber_beta=.5`, seeds `[42,43,44]`. The callable respects config grids/epochs for compatibility and small synthetic tests; callers must bind their protocol, not inadvertently pass the original three-alpha grid. Internal scaling constants and regularized decay are fixed, not tuned.

Returns `pred`, `ensemble_std`, `seed_predictions`, `n_parameters`, `fit_seconds`, `selection`, `fit_audit`, and `model_state`. The model state contains all per-seed models, fitted training-only centers/scales, target offsets and selection state. It contains only plain dictionaries and canonical original torch/sklearn objects, so file-loader module aliases do not prevent serialization. It can be saved with trusted-local pickle or `torch.save`; this runner uses **`torch.save`** and reloads with `torch.load(..., map_location="cpu", weights_only=False)`. Replay needs no labels/checkpoint. Same 1024-row batching is exact; different GEMM batch shapes may have floating-point differences. Non-neural per-seed outputs are identical replicas, not independently fitted uncertainty estimates.

## Frozen model flow

All six methods concatenate the same cached two-member vectors. Unscaled means no feature centering or scaling (ridge still fits its standard intercept). Stable preprocessing uses weighted population moments, with inverse-assay-SE weights:

- `mean_j = sum(w_i*x_ij)/sum(w_i)`;
- `var_j = sum(w_i*(x_ij-mean_j)^2)/sum(w_i)`;
- `global_spread = sqrt(mean_j(var_j))`, across all 768 columns, including constants;
- `scale_j = max(sqrt(var_j), .1*global_spread, 1e-8)`.

Every inner training fold and final fit gets its own moments. Zero-spread training data produce epsilon scales and zero transformed training values; truly unseen changes remain unclipped. The floor prevents amplification of nearly constant columns when other training features have spread; it is **not** a guarantee against unseen covariate shifts in an entirely constant feature matrix.

Ridge minimizes sample-weighted squared loss plus alpha penalty and selects alpha by pooled inner-OOF weighted MAE. The four MLP variants retain the original 768–128–ReLU–1 architecture, initializations, minibatches, weighted smooth-L1 beta=.5, gradient-norm clipping at 1, LR grid and maximum 40 epochs. LR and epoch are chosen using the first seed and pooled inner-OOF weighted MAE; ties retain grid order then earliest epoch. Final refits use all three seeds. Centered variants subtract the weighted inner-training/final-fit label mean and restore it at prediction; they do not scale target variance. Only the final variant adds AdamW weight decay `.01`; others use zero decay. Comparisons to original twin heads remain multi-factor comparisons.

Evaluation/calibration arrays never determine model preprocessing or selection. Calibration labels determine only the original finite-sample split-conformal radii; N includes fit and calibration. Test weights are fixed `1/max(SE,.10)`. Raw MAE, original potency bias, score-only SE-floor sensitivities `.05/.20`, prediction quantiles/extremes, fitted scales, and transformed feature extrema are retained. No pathological rows are removed.

## Runner and supervision

Repository root: `/home/dan/projects/nesso-finetune/PXR`.
Runtime: `/home/dan/swr/miniconda3/envs/cheminf/bin/python` with `PYTHONPATH=src:.`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.

```bash
python experiments/20260906_low_data_followup/code/run_readouts.py --mode initialize
python experiments/20260906_low_data_followup/code/run_readouts.py --mode smoke
python experiments/20260906_low_data_followup/code/run_readouts.py --mode run
python experiments/20260906_low_data_followup/code/run_readouts.py --mode verify
python experiments/20260906_low_data_followup/code/run_readouts.py --mode report
```

`smoke` executes the original N25 fold0/draw0 paired cells for all six methods and both split policies. Successful exact disk seed/CSV/metric replay and historical identity-role checks publish `pilot.json`; these genuine completed fits are reused in the main grid, not duplicated. `run` requires the matching pilot gate. The full task manifest includes both splits, five folds, ten draws, all five budgets plus full, and six methods. There is no elapsed-time or max-task cap. Signals finish the current task then pause; resumed incomplete writes are recomputed, valid hash-verified tasks skipped. Failed/corrupt tasks are never silently marked complete: `--retry-failed` archives their existing directory under `failed_attempts/` before a fresh attempt.

Original source/input hashes (including prepared identities/subsets/features and released checkpoint) are verified against the original binding. New source/protocol/runtime bindings and byte-preserved source snapshots are frozen at initialization. Changes require a distinct versioned attempt; current bound files must not be edited while running. Historical `repr_ridge`, `repr_mlp`, `head_random`, `head_pretrained`, and `morgan_ridge` outputs are references by hash only after exact fit/calibration/test identity and ordered evaluation-row matching. No reference fit is fabricated.

The exclusive runner lock prevents concurrent fitting. `status.json` provides current task, PID and counts; `failure_ledger.json` and each task's `failure.json` retain technical failures, while `runner_failure.json` exposes fatal setup/source errors. `complete.json` requires all declared outputs, binding/task identity, output hashes and successful disk replay. An empty original-style marker cannot pass. Full queue shutdown invokes final expected-vs-valid verification and reporting in separate subprocesses. Verification returns nonzero for incomplete grids. Process exit/launch is not scientific completion.

Reporting runs separately with `--mode report`, producing deterministic sorted `summary_tasks.csv`, `summary_paired.csv`, `summary_aggregates.csv`, and `summary.json`. Aggregates are **means of task-level weighted MAE**, not a pooled score or independent repeated-molecule analysis. Paired differences are new minus original reference on exact matched cells; no automatic model promotion follows.

Artifact root: `/home/dan/projects/nesso-finetune/PXR/artifacts/experiments/low_data_followup_20260906/readouts/`.
Supervised service: `nesso-pxr-followup-readouts.service`; actual service, pilot and test evidence are recorded under this root. Training has no GPU access requirement and no upstream/checkpoint weight changes.

## Verification evidence

Synthetic tests are in `tests/test_followup_readouts.py`: weighted scale formula, adjacent exact/near constants, all-zero spread, no clipping, all six methods' pickle and actual torch disk serialization replay, exact inner-training preprocessing call order and weights, fit-only target offsets, invariance of fitted state to changed evaluation features, invalid label-bearing keys, group-starved CV, runner task count, corruption/foreign-binding rejection, and interrupted-task reconciliation. Completion requires seed/CSV/metric/reference replay flags and the exact saved-model hash.

Executed recovery evidence (2026-09-06): **30 tests passed in 2.21 s** (`tests.log`). All **12 real N25 pilot cells** passed exact disk seed/CSV/metric replay and original identity-role matching (`pilot.json`, `smoke.log`). A second smoke invocation attempted **zero fits**, preserving all **108 task files byte-for-byte** and every pilot completion marker (`resume_verification.json`, `resume.log`). No model recipe was changed in response to smoke outcomes. A regression-test expectation was corrected for floating-point SD roundoff of identical ridge replicas; actual per-seed predictions remain exactly equal.

The full **3600-task** queue launched under user systemd `nesso-pxr-followup-readouts.service`, PID **1843424**, at **2026-09-06 13:59:41 UTC**. Read-back confirmed `active/running`, `RuntimeMaxUSec=infinity`, `TimeoutStopUSec=infinity`, CPU quota 200%, all numerical thread environments set to 2, and CUDA visibility empty. The first independent verifier found 45 valid tasks, zero failed/invalid, and correctly exited 2 for an incomplete ongoing grid; this is launch evidence, not full scientific completion. Binding SHA256: `ea780987796c215355074ae6166707fcd8fae69ab743b00dd6a02ec4790725af`.

Live handles:

```bash
systemctl --user status nesso-pxr-followup-readouts.service
# Artifact root below contains status.json, service.log, failure_ledger.json,
# pilot.json, resume_verification.json, verification.json, and launch.json.
# Final verification (exit 2 is expected while tasks remain):
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/dan/swr/miniconda3/envs/cheminf/bin/python experiments/20260906_low_data_followup/code/run_readouts.py --mode verify
```

The service is a transient user unit (`/run/user/1000/systemd/transient/nesso-pxr-followup-readouts.service`), not boot-persistent installation. Artifacts remain resumable after a host restart. Source-bound implementation/protocol files must stay unchanged while this queue runs; this README and tests are not in the bound source set.
