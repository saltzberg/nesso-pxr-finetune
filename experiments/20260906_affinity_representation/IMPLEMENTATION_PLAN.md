# Both-member affinity-representation adaptation — execution plan

Created: 2026-09-06. Status: authorized local continuation; prior-run analysis must finish before GPU capture/training.

## Human TL;DR
- Finish the completed follow-up's scientific analysis and publish its actual findings first.
- Then execute a real multi-record, both-affinity-member supervised pilot on one idle local GPU only. Pre-execution operational amendment: GPU0 became occupied by an unrelated ESMFold2 run; GPU1 is now free, with its unrelated Boltz supervisor deferred until 18:00 America/New_York. Use GPU1 for this short pilot without stopping or changing other work.
- Compare the same pretrained continuous heads with versus without rank-4 upstream affinity-module adapters. The cofolding trunk and binary heads remain frozen.
- Measure actual extraction, storage, fitting and prediction cost before freezing/launching a larger N100/N500 strict-split comparison.

## Authorization and scope
Dan approved: “Yes, please finish the analysis on the previous run - then continue with these tasks”. Local implementation and the measured pilot are authorized. New protocol choices precede their predictions. No external publication/submission, paid resources, package installation, host/image modifications, second-target data acquisition or full-trunk adaptation is implied. No elapsed-time cap may silently truncate a declared scientific matrix. Measured resource cost is a review point, not a fabricated time limit.

## Scientific question and expectation
Question: does changing the affinity representation improve assay prediction beyond an otherwise identical pretrained-head update, and does any benefit justify cost against descriptor-rich LightGBM?
Expectation before pilot: the established first-member native/differentiable parity and trainability should extend to both members and multiple real inputs; zero-update/native disagreement or unintended frozen-weight changes would invalidate this implementation. The pilot is not expected to establish held-out efficacy. Upstream adaptation may add no useful predictive information on this small assay cohort; retain that possibility in the subsequent study.

## Prediction flow
Frozen authoritative protein/ligand state and released structure inference -> captured real upstream affinity inputs -> two affinity members -> released continuous heads -> arithmetic member mean -> documented 6-minus-native conversion to pEC50 prediction -> assay-weighted fitting/evaluation.

Matched pilot arms:
- Head-only: train the two continuous heads; freeze upstream affinity modules and all remaining weights.
- Upstream-plus-head: train those same continuous heads plus declared rank-4/alpha-4 adapters in each member's ESM projection and first Pairformer transition; freeze all remaining weights.

Binary-head weights stay frozen in both arms; binary predictions can still change when their shared representation changes and must not be described as invariant. No calibration, clipping, stack or extra outcome transformation is introduced into the pilot.

## Identity and information boundaries
Use eight deterministically selected development FIT-role compounds from the existing strict fold0/draw0/N100 manifest. Do not select on labels or performance. Freeze their exact identities and source rows before new capture. Labels enter only the explicitly supervised pilot. This is development engineering, not a new test set. The pilot exercises actual assay-label gradients and disk replay but does not select a recipe based on fit improvement.

Native capture is a new versioned upstream realization, not silently identical to the old cached vectors. Freeze per-record seed, process order, preprocessing inputs, crop/recycles/precision and both-member RNG states. Demonstrate parity to each packet's captured native output. Future head-only and adapted arms must consume exactly the same new packet realization; reuse older controls only if their full contracts match.

## Work ownership
- Prior-run final analysis: separate new `experiments/20260906_low_data_followup/code/analyze_final.py`, its tests and `artifacts/experiments/low_data_followup_20260906/final_analysis/`. No new fits.
- Pilot implementation: this experiment's `code/pilot.py`, `protocol_pilot.json`, README and focused tests. No original source/image patches.
- Parent: cross-component verification, first-study/follow-up report publication, this plan, GPU supervisor and live readback, cost review and next-stage decision.

## Acceptance and ordered execution
1. Reconcile all 8,100 completed follow-up tasks, compute declared saved-prediction diagnostics and dependence-aware descriptive paired effects; retain metric/interval failures distinctly.
2. Publish a scientist-facing synthesis with evidence tables and verified LAN links. All follow-up analyses are retrospective; repeated test predictions are not independent observations.
3. Validate synthetic pilot tests and frozen source/identity/runtime protocol, without any GPU work until steps 1–2 are complete.
4. Verify the recorded physical GPU availability, image identity and required processed input assets. Stage and read back the eight-record packet manifest.
5. Capture both members using one persistent process, with checkpoint/source/input hashes and durable per-record outputs/failure ledger.
6. Test real zero-update native parity, supervised gradients/updates, frozen-module hashes and saved-state prediction replay for both matched arms. Verify no crop/train-mode path change.
7. Record capture time/bytes, steps and evaluated rows, throughput and peak VRAM; extrapolate storage and the actual proposed training loop from measured operations, not original frozen-readout timings.
8. Only after costs and scientific validity are clear, freeze a matched N100/N500 comparison and decide whether the intended draw/seed/epoch grid is justified. Do not launch an enormous GPU queue on assumed throughput.

## Stop conditions
Preserve and report invalid input coverage, state/crop changes, absent native parity, missing intended gradients, frozen-weight changes and replay mismatch. An implementation failure is not scientific evidence. Never repair canonical coordinates/state or discard difficult compounds silently. Do not claim completion on process launch or a successful exit alone.

## Verification and handoff
Read the original intent and the proposed next-stage plan. Use actual CLI help and the verified historical adapter command as entry points; no unimplemented command is assumed. Fit code may be developed while prior analysis runs, but GPU capture/label training starts only after that analysis is finished and verified. Long operations run as resumable supervised jobs. Save the exact running PID/service, image, GPU mapping, source/protocol/input bindings and final expected-versus-observed accounting.
