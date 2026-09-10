# Nesso-1 low-data adaptation — implementation plan

## Human TL;DR
Implement and execute the approved PXR-first study locally, preserving historical artifacts. Headline score is normalized inverse-assay-SE weighted MAE. Parallel implementation lanes have exclusive file ownership; parent integrates, verifies real pilots, and launches a resumable queue. No cloud spend, external publication or new-target data acquisition.

The earlier design discussion is preserved in `.hermes/plans/20260905-1954-nesso-low-data-adaptation.md`. This executable plan supersedes its approval stop for local implementation and bounded execution, following Dan's instruction to generate the plan and launch subagents.

## Frozen initial execution choices
- Dataset: only the 3,344 current development compounds from `data/published/modeling_manifest.csv`; the opened 790 and 513 challenge sets do not tune this new study.
- Primary weight: `1 / max(pEC50_standard_error, 0.10)`. The 0.10 floor comes from the existing protocol's prespecified floor grid, fixed here before new predictions, not presented as an independently measured assay noise floor. Score-only sensitivities use floors 0.05 and 0.20. Do not multiply by curation weights in this primary study.
- Reject missing/nonfinite/nonpositive assay errors explicitly; preserve the coverage audit and do not give invalid errors maximum weight.
- Budgets: 25, 50, 100, 250, 500 unique acquired labeled compounds; full outer-training-pool reference separately.
- Split policies: identity-grouped random and broader Morgan radius-2/2048-bit Tanimoto Butina clustering at similarity 0.50. Five outer folds; no outcome-based split selection. Butina grouping does not guarantee all cross-group similarities below 0.50; compute actual nearest-neighbor overlap.
- Ten paired subset draws, nested prefixes of one deterministic training-pool order per outer fold/draw. Exact-N row budgets; identity groups must be checked unique after curation or fail before execution.
- To retain honest interval calibration within N, reserve the last ceiling(20% of N) members of each acquired prefix for calibration, fit/select within the others, and record n_fit/n_calibration explicitly. Nestedness applies to the acquired label set, not necessarily its fit-only subset. No calibration label selects hyperparameters. Main x-axis says acquired labeled compounds, not fitted compounds.
- Model selection: a bounded prespecified inner-CV grid inside fit rows only. Neural final fits average seeds 42,43,44; same subset and calibration rows for all methods. Restrict torch/BLAS threads; no random model factory defaults.
- Split-conformal intervals at nominal 80% and 90%, with finite-sample quantile rank; retain infinity when too few calibration observations support nominal coverage. Intervals are ordinary empirical calibration, not weighted/conformal guarantees under chemical shift.
- Feature extraction: immutable paired 384D Nesso cached vectors; fresh deterministic Morgan bits from canonical SMILES. No use of the globally reduced historical 2D feature table. The LightGBM comparator here is Morgan-only, not the prior descriptor-rich model; label it literally.
- Primary model IDs: weighted_mean, native_continuous, scalar_ridge, repr_ridge, repr_mlp, head_random, head_pretrained, dual_head, head_lora, morgan_ridge, morgan_lightgbm, morgan_knn. Native continuous predictions retain their pIC50-equivalent/endpoint-mismatch label; scalar calibration includes reconstructed continuous and binary outputs.
- head_random and head_pretrained use identical twin 384–384–384–1 architectures and input ensemble handling. head_lora is a declared parameter-efficient final-readout intervention, not upstream representation learning.
- Separate lane audits and pilots affinity-Pairformer LoRA using real cached upstream inputs if available. If unavailable or not safe, record the exact blocker; never count head_lora as completion of deeper adaptation.
- Local runtime: existing cheminformatics Python environment for scientific code; pinned installed Nesso image for source/replay audits. No package installation without checking existing alternatives.
- Execution limits: at most four numerical CPU threads in the matrix process; one local GPU only for a separately verified feasibility pilot, no concurrent GPU experiments. Initial pilot is bounded; parent may launch the cached matrix for at most six hours after timing/smoke checks, with resumable timeout state rather than fabricated completion.

## Pre-outcome split amendment (protocol 1.1.0)

The label-free sensitivity audit in `SPLIT_SENSITIVITY.md` found that the initial Butina 0.50 setup left 84.33% of development compounds singleton. Before any outer-test performance comparison, the parent selected the audited 0.35 cutoff: 1,975 groups, 43.60% singleton compounds, largest group 36, and balanced five-fold test sets. This improves grouping breadth, but actual cross-fold similarities still exceed the centroid cutoff; do not call it a strict series or novelty holdout. The original protocol is `protocol_initial050.json`, and its immutable preparation remains under the original artifact root. The operative protocol is `protocol.json`, and the main run/preparation root is `artifacts/experiments/low_data_20260905_cut035`. All other scientific choices remain unchanged. This section supersedes the initial cutoff/run-root text above.

## Integration revision (same scientific protocol)

The first full-protocol N25 pilot completed all 24 model/split tasks. Independent weighted-score verification passed. Reporting exposed two implementation defects: default CSV float parsing altered near-ties and therefore Spearman values for KNN, and public provenance contained machine-specific paths. The reporter now uses round-trip float parsing and repository-relative public metadata. The runner also preserves exact bound source bytes, not hashes alone. The first pilot, its original source snapshot and report are retained under `artifacts/experiments/low_data_20260905_cut035`. A fresh operative run is `artifacts/experiments/low_data_20260905_cut035_run2`, with the same models, labels, weights, splits, seeds and tuning protocol; this operational path is in `protocol.json`. The pilot is repeated there before the supervised queue launches. No model setting was changed in response to performance.

## Exact stage flow
Canonical development identity/SMILES + frozen checkpoint/cache -> immutable Nesso vectors or Morgan bits -> fitted subset-only preprocessing/readout -> seed ensemble -> intervals from calibration labels within N -> held-out test predictions -> fixed assay-weighted score. No score clipping, no test-label model selection, no target-label access outside declared N by model fitting.

## Ownership and shared interfaces
All lanes can read the repository, but write only the owned new files. The repository is already dirty and contains large untracked scientific artifacts: do not stash, reset, commit, rewrite history, or move to a worktree lacking the artifacts. Avoid machine-specific absolute paths in tracked/report text; local runtime logs with absolute paths belong under artifacts/.

### Lane A — contracts, splits, preparation
Own `src/nesso_pxr/low_data_contract.py`, `tests/test_low_data_contract.py`, `scripts/prepare_low_data.py`.
Provide:
- `assay_weights(se, floor=0.10) -> np.ndarray`, reject invalid errors.
- `weighted_mae(y, pred, se, floor=0.10) -> float`.
- `score_predictions(y, pred, se, floor=0.10) -> dict` including weighted_mae, mae, spearman (None if undefined), bias, effective_n, weighted_mae_floor_005, weighted_mae_floor_020.
- `prepare_data(repo_root: Path, output_dir: Path, config: dict) -> dict` writes inputs.csv, features.npz (keys member1/member2/morgan/native_continuous/scalar_features), assignments.csv (split,outer_fold,row_index,chemical_group), subsets.csv (split,outer_fold,draw,n_train,row_index,role fit|calibration), preparation_manifest.json and split_audit.json. Here n_train is the acquired label budget for compatibility with task IDs, not n_fit.
- row_index is contiguous 0-based in inputs.csv; include record_id, feature_row, canonical_smiles, pEC50, pEC50_standard_error. Inputs in feature arrays align to row_index. assignments outer_fold denotes the assigned TEST fold, one row per policy per compound. Full reference uses n_train=-1.
- `load_prepared(output_dir) -> tuple[pd.DataFrame, dict[str,np.ndarray],pd.DataFrame,pd.DataFrame]` for inputs/features/assignments/subsets.
Use config keys from protocol.json. Prepare actual data only into artifacts/experiments/low_data_20260905/prepared; no fits.

### Lane B — model factory
Own `src/nesso_pxr/low_data_models.py`, `tests/test_low_data_models.py`.
Provide `fit_predict(method, train_features, y_train, assay_se_train, test_features, *, config, checkpoint_path, seeds=(42,43,44)) -> dict`.
Feature dict keys member1, member2, morgan, native_continuous, scalar_features; arrays aligned, sliced by caller. test_features concatenates calibration+test rows; model must not know their labels. Config contains se_floor, inner_folds, grids, epochs, threads. Return pred (1D array), ensemble_std (1D), seed_predictions (seeds x rows), n_parameters, fit_seconds, selection (JSON-safe), fit_audit (JSON-safe), optional model_state/checkpoint serialization helper. Export METHODS tuple. No matrix launching; run tiny deterministic fixture tests and one real n=25 fitting smoke if inputs exist.
Ridge and trees use assay weights, neural models weighted smooth-L1 objective. Reconstruct native binary branch scalar features in A or helper only from checkpoint (no labels). Use all final seed predictions and no test-aware tuning. For singleton-starved inner grouping, use deterministic training-only CV with explicit policy; chemical inner groups should be passed as train_features['groups'] if available; report infeasibility rather than use test rows.

### Lane C — metrics aggregation, figures and HTML
Own `src/nesso_pxr/low_data_report.py`, `scripts/report_low_data.py`, `tests/test_low_data_report.py`.
Provide `build_report(run_dir: Path, output_dir: Path) -> dict`.
Read task outputs under run_dir/tasks/{task_id}/predictions.csv + metrics.json, status.json, protocol.json, prepared/split_audit.json. Prediction CSV fields: record_id,row_index,split,outer_fold,draw,n_train,method,y_true,y_pred,assay_se,weight,ensemble_std,nearest_similarity,chemical_group,lower_80,upper_80,lower_90,upper_90. metrics.json fields include task_id,status,split,outer_fold,draw,n_train,method,n_fit,n_calibration,n_test,weighted_mae,mae,spearman,fit_seconds,n_parameters,coverage_80,width_80,coverage_90,width_90,selection,fit_audit. Handle missing/infinite intervals explicitly; do not discard failed tasks.
Generate compact index.html, metrics.csv, source plot tables and PNG/SVG figures for learning curves, paired effects, novelty/error, calibration/interval coverage, cost/error. Include exact model flow, incomplete status/coverage, raw versus effective counts, singleton caveat, retrospective status and Morgan-only baseline identity. No fabricated result placeholders or claims from fixture data. Tests may use labelled synthetic fixtures but never publish them. Group uncertainty at independent draw/fold/compound units, never treat all repeated prediction rows as independent. Parent handles serving/copy/navigation; do not modify existing site pages.

### Lane D — deeper adapter feasibility
Own `src/nesso_pxr/low_data_adapter.py`, `scripts/audit_low_data_adapter.py`, `tests/test_low_data_adapter.py`, and `experiments/20260905_low_data_adaptation/ADAPTER_FEASIBILITY.md` plus artifacts/experiments/low_data_20260905/adapter_pilot.
Inspect pinned architecture and real existing metadata caches. Implement explicit nn.Linear low-rank injection with zero-update parity, finite nonzero intended gradients, updated adapter weights, unchanged frozen weights and checkpoint reload tests. Exercise a real affinity module/backward pass from real cached upstream inputs if available; preserve crop/features protocol and state inputs. A random-tensor unit test alone must not be reported as real Nesso feasibility. No full extraction/matrix; bound numerical work to four threads, one GPU and a ten-minute pilot. If unavailable, provide exact missing input/cache boundary, measured evidence, and a usable next-step command derived from actual CLI help. Do not change installed Nesso or its base image.

### Parent — integration and execution
Own protocol.json, implementation plan, experiment README, runner/verification scripts, LAN report integration and all cross-lane fixes after reconciliation. Verify outputs rather than trusting summaries.

## Implementation sequence
1. Write this plan and explicit protocol; launch the four lanes concurrently.
2. While lanes implement, build resumable runner: frozen protocol/source/input hashes; exclusive per-task directories; atomic writes; failure ledger; no test labels passed to fitting; real nearest-neighbor similarities from fit rows; label budget audit and conformal quantiles.
3. Integrate and run all old/new tests, lint only new code, preparation integrity checks, and real paired pilots across both split policies.
4. Review adapter parity and gradient evidence; mark feasible/blocked distinctly. Time cached pilots and launch a supervised resumable main queue within the stated cap. Persist partial reports as tasks complete.
5. Serve current report as a new LAN subpage; preserve existing historical pages except a new link and explicit split/retrospective disclosure. Test HTTP routes and source-data identity.

## Risks and stop conditions
- Genuine invalid source identities/SEs, infeasible splits, invalid tensors or mutated protocol stop preparation/run. Individual model failures are recorded and do not suppress other methods.
- Source code changes after launch require an explicit new attempt/run binding, not silent mixed-code resumption.
- No external data submissions/publication or additional target experiments in this implementation.
- Never reinterpret a CPU/GPU failure as lack of scientific signal; never label partial coverage as complete.
- No winner declaration until paired uncertainty and a practically meaningful margin are prespecified/verified. Until a margin is accepted, report effect sizes/intervals, not promotion.

## Acceptance checklist
- [x] Versioned protocol/source/input manifests and explicit model flow exist.
- [x] Unit tests cover weights, invalid errors, exact-N/nestedness, disjointness, model fitting/replay, intervals and report aggregation.
- [x] Real data preparation reconciles all 3,344 development rows and both split assignments.
- [x] Real pilots produce finite saved predictions and independently recomputed scores, with labels restricted to N.
- [x] Deeper-adapter pilot is verified or a precise blocker is visible; head LoRA is not relabelled.
- [x] Main queue is supervised/resumable, live activity is verified, and every planned task has an explicit state.
- [x] Current report is reachable on LAN with correct source numbers and no fabricated completeness.
- [x] Final report distinguishes implementation complete, run in progress/complete, and unresolved science.

## Verification commands
Use the installed validated Python interpreter with PYTHONPATH=src:.; run `python -m pytest` and `python -m ruff check` on new modules/tests/scripts. Derive new runner arguments from implemented --help. Verify the exact service via systemctl and fetch the new page through both LAN and Tailscale addresses. Full experiment completion requires expected task IDs versus valid output reconciliation, not exit status alone.

## Handoff
Read this plan, protocol.json, the current run status and lane summaries. Preserve existing work. Continue from verified artifacts; do not rerun immutable upstream features or change the label/weight contract silently. User requested actual parallel implementation/execution, not a plan-only handoff.
