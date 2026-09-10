# Nesso PXR follow-up: implementation and execution of steps 1–4

## Human TL;DR
Implement and execute the approved follow-up locally: close out the completed first matrix; test stable frozen-feature readouts; restore a descriptor-rich 2D comparator under honest label budgets; evaluate a strictly similarity-separated training pool. Preserve every first-study prediction, source-bound module and protocol. Parallel lanes own disjoint files and report actual tests, artifacts, process IDs and completion checks. No arbitrary elapsed-time cap. No upstream adaptation, cloud spend, external publication or challenge submission.

## Authorization and interpretation
Dan requested an implementation plan and subagents to execute steps 1–4. This authorizes local code, tests, CPU experiments and LAN report updates within this scope. It is not plan-only and does not require another approval to launch verified local queues. Resource limits constrain numerical threads, not scientific completion. All follow-up results are retrospective/hypothesis-generating because the existing development outcomes and failure cases were examined. Repartitioning these compounds does not create untouched confirmation.

## Context gathered
- Completed first study: artifacts/experiments/low_data_20260905_cut035_run2; all 7,200 systems completed and independently verified.
- First protocol, MODEL_NOTES and README: experiments/20260905_low_data_adaptation/.
- Independent assessment and supporting CSV/JSON: first run's independent_assessment* files.
- Existing model/readout implementation: src/nesso_pxr/low_data_models.py; existing preparation/runner/report modules are read-only references.
- Historical descriptor-rich comparator discovery starts at src/nesso_pxr/comparison.py and scripts/run_publication_comparison.py; its actual fitted transformations must be traced, not inferred from its name.
- Existing runtime: validated `cheminf` scientific environment, `python` with PYTHONPATH=src:.; NumPy/RDKit/sklearn/LightGBM/torch already available. Site preview on port 8890.
- Existing dirty tree belongs to the user. No reset/stash/commit or worktree missing scientific artifacts.

## Shared scientific contract
1. Development-only 3,344 identities, same cached two-member Nesso vectors and released checkpoint. No opened challenge-set fitting/evaluation and no new external data acquisition.
2. Headline inverse-assay-SE weighted MAE, w=1/max(SE,.10), fixed evaluation weights. Preserve raw MAE, SE-floor score-only sensitivities, potency bias and prediction-tail diagnostics. No post-hoc prediction clipping or dropping extreme rows.
3. Original policies: reuse exact prepared random/chemical-cluster assignments and acquired subsets. Budgets 25/50/100/250/500/full, five folds, ten draws, final neural seeds 42/43/44; all fitting, selection and calibration labels remain inside N. Reuse baseline prediction artifacts only when all task identities/roles match exactly and retain source hashes; never fabricate duplicate completed fits.
4. Twenty percent reserved calibration, three-fold inner selection within fit labels, no calibration/test-aware hyperparameter selection. Fit learned imputation, scaling, variance selection, dimensionality reduction separately in each inner training partition and final fit partition.
5. Freeze per-lane JSON protocol, candidate grids, source/input hashes and selected methods before outcome execution. If discovery reveals a required amendment, append rationale before new fits; preserve superseded snapshots. Protocol choices after first-study outcomes are explicitly retrospective.
6. Exact model flow and preprocessing must be reported. Stable MLP versus original twin head remains a multi-factor comparison unless controlled variants isolate each factor.
7. Report paired effects descriptively; no promotion/superiority/equivalence without a user-accepted practical margin and defensible dependence-aware inference. Do not invent a margin or count repeated rows as independent observations.
8. Task systems save predictions, per-seed outputs/model replay state, metrics, fit/label-access audit, diagnostic extrema, source/input/protocol bindings and output hashes. Atomic writes, exclusive runner lock, resume hash-verified outputs only, individual failures retained, final expected-vs-valid reconciliation.

## Architecture and file ownership
Place new implementation under experiments/20260906_low_data_followup/code/ with __init__.py; each lane's test files under tests/test_followup_<lane>*.py. This avoids modifying first-study source-bound files. Artifact roots: artifacts/experiments/low_data_followup_20260906/{readouts,baselines,strict_split,closeout}. Protocol/docs are under the new experiment folder. Parent owns IMPLEMENTATION_PLAN.md, README.md, final integration/status and cross-lane reconciliation.

A and B expose the same callable signature as existing low_data_models.fit_predict:
fit_predict(method, train_features, y_train, assay_se_train, test_features, *, config, checkpoint_path, seeds=(42,43,44)) -> dict with pred, ensemble_std, seed_predictions, n_parameters, fit_seconds, selection, fit_audit, replayable model_state where applicable.
Additional descriptors live in feature dict key 'descriptors'; they must align to row identity, with immutable feature-name/order provenance. Any cached descriptor preprocessing must be purely row-wise/unfitted. Export METHODS and any descriptor-preparation entry point in lane B's module. Use importlib loading or ordinary package imports; do not alter sys.path globally or monkeypatch the original factory in a running process.

## Lane 1 — closeout and reporting
Own experiments/20260906_low_data_followup/CLOSEOUT.md, code/closeout.py, tests/test_followup_closeout.py and site/low-data-assessment/ only. Read independent_assessment* and recompute aggregate numbers from original final task table. Publish compact first-study intent/results/failure explanation/exact flow and scope boundaries, linked source tables and immutable assessment copy. Preserve historical pages and original outputs; parent adds navigation links after verification. Include completion versus scientific limitations, Morgan-only comparator identity, retrospective status, and scaling vulnerability without labelling computationally reproducible extremes as valid deployment behavior. Add new follow-up status links only when actual targets exist. Do not rewrite the old bound reporter or silently modify the full-matrix plots.

Acceptance: original metrics matched, all required sources copied with hashes, generated HTML/CSV links valid, clear what was tested/learned/unproven, tests pass, actual LAN page HTTP verified. No new model fits.

## Lane 2 — stable readouts
Own code/readouts.py, code/run_readouts.py, tests/test_followup_readouts.py, READOUTS.md, protocol_readouts.json, artifact readouts/.
Implement and preregister a compact controlled matrix rather than an optimizer sweep:
- Ridge with unscaled vectors; ridge with stable training-only scaling. Retain original scaled-ridge scores as the historical comparator.
- MLP unscaled/direct-output; stable-scaled/direct-output; stable-scaled/target-centered; stable-scaled/target-centered/regularized. Keep architecture fixed at the original 768–128–1 and matched final seeds so contrasts isolate successive choices. Target centering uses fit-only label location; inner centering uses inner-training labels only. Keep original scaled MLP scores as reference.
- Candidate numerical scaling rule: center with weighted training mean; use scale_j=max(weighted SD_j, c*weighted training global RMS feature spread, epsilon), with c=.1 fixed before new outcomes. Exact global-spread definition, epsilon and zero-spread behavior must be declared and tested; do not estimate reference scales from the outer test/unlabelled evaluation cohort. An equivalent defensible purely training-only rule may be selected and documented before fitting, not by checking old pathological molecules.
- Ridge grid [1,100,10000,1000000]; original neural LR/epoch budget initially retained. Explicit regularized variant uses a fixed positive AdamW weight decay documented before fitting (default .01); do not conflate it with the scaling-only contrasts.
Record scaled input extrema, fitted scale distributions, prediction quantiles/extremes and saved-state replay. Test nearly constant features, tiny folds, leakage instrumentation, adjacent exact/near constants and no output clipping. Preflight synthetic behavior and actual N25 paired smoke without using its score to tune recipes. Then execute the full original paired subset grid for new methods under a resumable local supervisor; existing random/pretrained head references are reused by hash only on matching cells. At most two numerical threads. No wall-clock cap. Separate reporting from fitting.

## Lane 3 — stronger 2D baselines and inexpensive controls
Own code/baselines.py, code/run_baselines.py, tests/test_followup_baselines.py, BASELINES.md, protocol_baselines.json, artifact baselines/.
Trace the actual historical descriptor-rich LightGBM system. Reconstruct its raw deterministic descriptor/fingerprint features and documented modeling recipe, but fit every learned selector/imputer/reducer inside each inner-training/final-fit subset. Do not use a globally reduced historical matrix or report a replacement feature set as the exact historical system. If a source/descriptor implementation is unavailable, report exact missing dependency and a separately named feasible comparator instead of silently claiming restoration.
Implement descriptor-rich LightGBM, assay-weighted median anchor, and restrained native continuous affine calibration. Affine transform fits within N (weighted squared regression or another explicit frozen objective; disclose exact objective), not on test labels. Keep Morgan ridge as existing reference. Freeze bounded historical/honest adaptation tuning grid before new outcomes; compare on same acquired labels/test rows as lane 2. Add source/feature identity and fit-only preprocessing tests, replay, real smoke and then supervised uncapped full paired execution. At most two numerical threads. Deterministic descriptor extraction may run label-free before model protocol finalization; numerical features and metadata are immutable.

## Lane 4 — strict chemical-separation evaluation
Own code/strict_split.py, code/run_strict.py, tests/test_followup_strict.py, STRICT_SPLIT.md, protocol_strict.json, artifact strict_split/.
First execute structure-only feasibility/preparation, without reading outcome scores. Preserve each original chemical-cluster outer TEST fold. Purge from its outer training pool any molecule with Morgan radius2/2048 Tanimoto similarity >=.35 to ANY test molecule. Require measured maximum training-pool/test similarity <.35, including calibration rows. Record every removed identity/reason, test/pool counts, exact overlap checks, similarity distributions and chemical coverage. This changes train support, not an independent untouched evaluation cohort. Do not claim temporal/prospective validation.
Use deterministic filtered original draw orders to form nested acquired subsets from eligible pools, preserving the N/calibration contract. All five folds must support the requested N500; full reference uses each eligible pool. If infeasible, publish actual feasibility and execute only explicitly supported budgets with excluded budgets enumerated; do not relax threshold silently or adjust using outcomes. Preserve exact fixed test rows so retention changes are visible.
Strict panel is fixed before new results: native_continuous, weighted_median, native_affine, head_pretrained, head_random, morgan_ridge, descriptor_lightgbm, stable ridge unscaled and stable ridge scaled. Import A/B methods when verified; do not choose entrants based on new outer-test ranking. Lane may prepare runner/protocol immediately, but launch model tasks only after A/B interfaces and provenance gates pass. Small upstream changes are excluded. At most two numerical threads. Five folds, ten draws, supported budgets plus eligible-full; no arbitrary time cap.

## Parent integration and execution tasks
- Verify lane implementation, ownership, test output and actual service handles. Reconcile source boundaries, exact method IDs and protocols before strict-panel launch.
- Check first-study source/input hashes remain intact; do not mutate prepared originals.
- Run full tests and focused lint. Exercise real CLI prepare/fit/resume/verify paths; source drift and corrupt completion markers must fail safely.
- Add project navigation to verified closeout/follow-up pages. Preserve new results separate from chapter 1 and expose status from actual artifacts.
- Supervise all approved tasks through completion with no duration cap; CPU resource bound is up to three two-thread fitting lanes, with reduced report overhead and no GPU/API spend. Long numerical jobs must be systemd supervised rather than tied to child-agent lifetimes.
- Attach completion verification that checks the generated task manifests against valid artifacts, not just service exits. Report unavailable recipes/splits as explicit blockers, not scientific losses. Do not claim completion before acceptance is met.

## Risks and stop conditions
- Changed source/identity/feature bindings: stop that run and create a versioned attempt, never mix silently.
- Historical comparator cannot be faithfully reconstructed: state the exact boundary; continue the feasible independently named comparisons, no false historical equivalence.
- Strict purge leaves insufficient training chemistry: report label-free feasibility and supported matrix; no silent threshold fallback.
- Follow-up test outcomes already known: retrospective label on all analyses, no untouched-confirmation claims.
- Extra preprocessing tuned on prior extreme rows: prohibit; test mechanism with fixtures, freeze choices and preserve original bad scores.
- No destructive edits, global environment/package changes, paid compute, upstream adaptation, or external publication.

## Human decisions still open (not implementation blockers)
The practical improvement margin and future deployment/untouched-evaluation chemistry are not specified. Report observed paired effect sizes and limitations without automatically promoting a model. Implementation, local execution and retrospective reporting proceed.

## Acceptance checklist
- [ ] First study closed out with an accurate LAN-accessible assessment and preserved original artifacts.
- [ ] Stable-readout protocols, tested implementations and full paired outputs exist.
- [ ] Descriptor-rich baseline provenance restored or precisely bounded, honest within-budget fits executed; median/affine controls verified.
- [ ] Strict-similarity preparation audited and supported fixed-panel tasks executed.
- [ ] Every named task accounted for, source/input/output hashes verified, prediction/model replay passes.
- [ ] Full tests, focused lint, CLI resume/corruption tests and live report links verified.
- [ ] Final synthesis distinguishes preprocessing findings, strongest-baseline comparison and stricter-chemistry behavior; no invented superiority claims.

## Handoff
Read this plan, lane protocols/docs, original independent assessment and artifact status files. Continue from verified local artifacts without repeating extraction or completed fits. User authorized implementation and subagent execution of steps 1–4 only. Keep first-study history immutable and finish the approved matrix without an arbitrary runtime cap.
