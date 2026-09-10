# Stable readouts, stronger baselines and stricter chemistry holdout

This follow-up tests whether the first Nesso/PXR study's conclusions survive stable preprocessing, a descriptor-rich 2D comparator and explicitly similarity-separated training chemistry. It is a retrospective extension on the same development cohort, not untouched confirmation or upstream Nesso adaptation.

Created: 2026-09-06. Last edited: 2026-09-06.

## Execution

The user authorized implementation and local execution of steps 1–4. The four lanes in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) are implemented; the closeout is published on the existing LAN site and all three numerical queues are complete:

1. Completed first-study assessment and evidence-linked LAN page.
2. Controlled stable-scaling/readout comparisons on original paired labels.
3. Honest within-budget descriptor-rich LightGBM, weighted median and native affine controls.
4. Structure-only strict similarity-purge audit, followed by a fixed-model comparison on supported label budgets.

Implementation-agent dispatch is not evidence that a training queue has started or completed. Numerical runs must pass protocol, provenance, leakage, replay and real-pilot checks before supervised launch. Task manifests and artifact status files—not this prose—are authoritative for execution counts. Approved local queues have resource limits but no arbitrary wall-clock cap.

## Scientific scope

The first run and its predictions remain immutable. New methods use the same inverse-assay-SE weighted MAE and count all fit/tuning/calibration labels inside N. Historical comparator reconstruction must disclose the actual raw features and fit-only transformations. Strict separation requires every acquired/calibration molecule to have Morgan Tanimoto <0.35 to every molecule in its fixed outer test fold; insufficient pools must be reported, not silently relaxed.

The follow-up does not include upstream adapters, extra target datasets, blinded challenge submissions, paid compute or external publication. No practical superiority margin has been accepted; report paired effects without automatic winner declarations.

## Artifact roots

- First completed study: `artifacts/experiments/low_data_20260905_cut035_run2/`.
- Follow-up: `artifacts/experiments/low_data_followup_20260906/` with `readouts/`, `baselines/`, `strict_split/` and `closeout/` lanes.
- Lane scientific protocols and notes: this directory.
- New implementations: `code/` beneath this directory; original source-bound modules are not edited.

## Verified launch and monitoring

The original implementation agents timed out after ten minutes. Their saved implementation was continued rather than restarted. These agent timeouts did not impose training timeouts.

- `nesso-pxr-followup-readouts.service`: six methods, 3,600 paired tasks.
- `nesso-pxr-followup-baselines.service`: three methods, 1,800 paired tasks.
- `nesso-pxr-followup-strict.service`: nine methods, 2,700 tasks.
- `nesso-pxr-followup-completion-monitor.service`: read-only reconciliation every five minutes, with final full output hashing after each queue exits.

The numerical services use two numerical threads each and unlimited runtime. These are transient user services: they survive this terminal/network connection, but are not a claim of boot persistence. The existing runners verify their own completed matrices and retain failures. The separate monitor writes `completion_monitor/status.json`, per-task CSVs and explicitly descriptive task-mean CSVs beneath the follow-up artifact root. It does not refit models or turn partial, unpaired summaries into comparative evidence.

The strict audit retained sufficient eligible molecules in all five folds for every requested budget (25, 50, 100, 250, 500 and eligible-full), with every measured acquired/test similarity below 0.35. This is a stricter retrospective evaluation on the existing cohort, not a newly untouched test set.

The descriptor comparator is a newly reconstructed canonical-SMILES RDKit2D + Mordred2D + Morgan pipeline with training-only reduction. It is not an exact replay of the historical descriptor-rich system; the unavailable unreduced source tables and preparation lineage are disclosed in [BASELINES.md](BASELINES.md).

One integration test remains unresolved: the frozen baseline protocol includes absolute input paths and fails the repository portability check. Its bytes are already bound into the running experiment and have deliberately not been edited. This documentation defect is separate from the passing numerical, leakage and replay tests.

## Completed matrix

Readouts completed 3,600/3,600 tasks, baselines 1,800/1,800 and strict chemistry 2,700/2,700, with no recorded failures. Final output hashing passed for all three lanes in `completion_monitor/status.json` at 2026-09-06T16:07:32Z. The strict runner replayed all 2,700 saved systems; baseline final replay covered six systems in addition to its per-task execution checks. The monitor's `active/exited` systemd-state handling was corrected separately from training code.

## Final scientific analysis

[Results and interpretation](RESULTS.md) completes the paired, potency-tail, novelty, calibration/interval and cost analysis. The source calculation replayed 6,219,840 prediction rows including original controls, with no new fits. The LAN report is at `http://192.168.6.154:8890/low-data-followup/`; source CSVs and the independent verification are linked there.

The overall descriptor advantage is real in these saved comparisons but is not uniform across potency: Nesso's adapted head retains lower weighted error in the pEC50 >=4 stratum at N500, while all main systems underpredict the >=6 tail. Conditional reweighting intervals describe the fixed fitted systems and inspected cohort, not prospective uncertainty. See RESULTS.md for the full qualifications.

See the implementation plan for per-lane acceptance and continuation instructions.
