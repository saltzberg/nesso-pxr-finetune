# Nesso-1 adaptation from 25–500 PXR measurements

This study compares ways to adapt pretrained Nesso-1 to cellular PXR activation using small, explicitly counted sets of target-specific labels. The upstream Nesso representations remain frozen in the primary matrix; a separate real-input experiment tests whether upstream affinity-module LoRA can be trained at all.

Created: 2026-09-05. Last edited: 2026-09-05.

## Model

Canonical compound identity and prepared ligand state
→ cached released Nesso representations, or Morgan bits [frozen]
→ selected readout and any input scaling [fitted within the acquired label budget]
→ average over initialization seeds
→ uncertainty intervals [calibrated using reserved labels inside that same budget]
→ outer-test pEC50 predictions and assay-uncertainty-weighted evaluation.

The frozen representations are two 384-dimensional affinity vectors. Representation ridge and MLP concatenate them into 768 inputs. The scalar model uses nine reconstructed continuous/binary outputs. Continuous-head tuning starts from the released twin readouts; matched random initialization tests whether pretraining helps beyond the architecture. Dual-head tuning additionally updates the binary branches and their declared fusion. Head LoRA changes only the final readouts, not the frozen vectors. Morgan ridge, Morgan-only LightGBM, and a Tanimoto nearest-neighbor model are non-Nesso comparators. Weighted mean and the unchanged native continuous output are anchors.

Exact model architecture, fitted parameter counts, loss, preprocessing and saved-model replay are in [MODEL_NOTES.md](MODEL_NOTES.md). The native continuous anchor is pIC50-equivalent affinity, not native cellular pEC50.

## Training and evaluation

The primary matrix uses 3,344 development compounds, with acquired-label budgets of 25, 50, 100, 250 and 500 plus a separate full-pool reference. It includes random identity holdout and Morgan chemical-cluster holdout, five outer folds, ten paired subset draws and three initialization seeds for neural models. The unchanged nonneural model is not three independent fits. All methods share the same acquired compounds and test fold for a given cell.

Twenty percent of each acquired label set is reserved for interval calibration; the rest supplies fitting, inner cross-validation and epoch selection. Thus N=25 means 20 fit/tuning labels and five calibration labels, not 25 fitting labels plus a hidden validation set. Acquired sets are nested; their fit-only portions need not be.

The primary score is

    weight_i = 1 / max(assay_standard_error_i, 0.10)
    weighted_MAE = sum(weight_i * abs(prediction_i - measurement_i)) / sum(weight_i)

The fixed 0.10 floor limits unusually large weights. It is an engineering choice from the existing protocol's floor grid, not independently measured assay precision or a parameter selected from the new test results. Evaluation weights do not depend on model uncertainty or errors. Supporting outputs retain unweighted MAE, low/high-potency diagnostics and score-only floor sensitivities. Training and inner model selection also use assay uncertainty; the primary study does not multiply these weights by historical curation weights.

Neural training minimizes weighted smooth-L1; weighted ridge uses squared loss; the tree comparator uses weighted absolute-error loss. Hyperparameters are selected by inner weighted MAE. These objectives are disclosed rather than described as identical optimizers.

## Current execution and outputs

The implementation provides a resumable, source/input/protocol-bound task runner and a report that recomputes scores from saved predictions. The initial local queue is capped at six hours and pauses with explicit remaining tasks if it reaches that cap. A launch is not completion; current observed counts and any failures are in the generated report and run `status.json`.

- LAN report: the `/low-data/` page on the existing project preview service.
- Operative protocol: [protocol.json](protocol.json).
- Implementation and acceptance plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
- Main artifacts: `artifacts/experiments/low_data_20260905_cut035_run2/` from repository root.
- Each complete task retains predictions, seed predictions, model state, fit/selection provenance, label-access checks, metrics and output hashes.
- Five figure families cover learning curves, paired effects, nearest-neighbor similarity versus error, prediction/interval diagnostics and accuracy versus fitting cost. Plot tables are downloadable.

The report does not declare a winner from a partial panel. Its current paired bars show observed fold/draw variation, not inferential confidence intervals. A practical improvement margin and a separate dependence-aware inferential analysis remain necessary before a recommendation is presented as established.

## Interruption recovery and reporting revision

The training queue survived the terminal/network interruption under `nesso-pxr-low-data-20260905.service`. The frozen runner, protocol, source bindings and task artifacts remain unchanged.

The original report compares measured values against serialized interval endpoints, while the runner computes coverage as `abs(y - prediction) <= radius`. Floating-point cancellation can make these mathematically equivalent expressions disagree exactly at a boundary. The reporting-only `scripts/report_low_data_v2.py` reconstructs radius from the saved width, requires exact replay of both endpoints, and recomputes coverage with the runner's original residual comparison. It does not widen intervals or relax score tolerances. `interval_replay.json` lists every differing task and compound and both coverage values.

The revised live report is `/low-data-verified/`; `/low-data/` preserves the original verifier. A separate bounded `nesso-pxr-low-data-report-v2.service` refreshes the revised report every two minutes while the queue is running, with a final refresh after the queue pauses/completes. Neither service is a promise of completion of the full matrix. On reboot these transient jobs require explicit resumption; they survive terminal disconnects.

## Uncapped completion authorization

On 2026-09-06 the user explicitly removed the six-hour operational cap and requested completion of the entire matrix. The queue resumes with `scripts/run_low_data.py --max-seconds inf --retry-failed` under `nesso-pxr-low-data-finish.service`; the service also has no runtime deadline. The original protocol/source bindings remain unchanged: this is an explicit command-line execution-limit override, not a scientific protocol change. Completed, hash-verified tasks are skipped. The reporting-only service `nesso-pxr-low-data-report-finish.service` refreshes `/low-data-verified/` every five minutes with no time cap and a final refresh when the runner ends. This supersedes the earlier operational cap and refresh cadence described above.

## Limitations and interpretation

This is a retrospective development-set study of PXR activation, not a blinded challenge evaluation or a general answer across targets. Previously opened challenge sets do not tune this matrix.

The original chemical-family split had 83.76% singleton compounds and mostly measured molecule-held-out prediction. A structure-only audit found that the initial new Butina 0.50 grouping still had 84.33% singleton compounds. Before outcome comparisons, the protocol was amended to Butina 0.35: 43.60% singleton compounds, 1,975 groups and a largest group of 36. Actual cross-group similarities can exceed the centroid cutoff, so the new split is not a strict series-extrapolation guarantee. Both the original preparation and the amendment evidence are preserved in [SPLIT_SENSITIVITY.md](SPLIT_SENSITIVITY.md).

Ordinary split-conformal intervals do not guarantee coverage under chemical shift. With five calibration observations, the nominal 90% interval is legitimately unbounded; the pipeline preserves that result rather than inventing a finite interval. Seed spread is a separate diagnostic, not automatically a calibrated prediction interval.

A real-input upstream affinity-module LoRA test passed gradient, frozen-weight, zero-update and saved-adapter replay checks. It used an engineering objective without assay labels and only the first affinity member. No upstream-adapter sample-efficiency result is established; see [ADAPTER_FEASIBILITY.md](ADAPTER_FEASIBILITY.md). Cached-head fitting costs exclude the one-time upstream extraction, whose historical timing is not reconstructed here.

## Reproduction

From the repository root, use the existing scientific Python environment with NumPy, pandas, RDKit, scikit-learn, LightGBM, torch, safetensors and matplotlib. No external API or paid compute is required.

```sh
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/prepare_low_data.py --verify-only \
  --output-dir artifacts/experiments/low_data_20260905_cut035_run2/prepared
python scripts/run_low_data.py --help
python scripts/verify_low_data.py \
  artifacts/experiments/low_data_20260905_cut035_run2
python scripts/report_low_data.py \
  --run-dir artifacts/experiments/low_data_20260905_cut035_run2 --report-dir site/low-data
```

`run_low_data.py` defaults to the operative protocol. Do not start a duplicate runner while its supervisor is active. Resume skips only hash-verified complete tasks; changed bound source, protocol or inputs require a distinct run, not silently mixed results. Pickled models are trusted-local replay artifacts, not safe exchange files.
