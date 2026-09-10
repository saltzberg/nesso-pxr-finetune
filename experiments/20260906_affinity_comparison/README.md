# Does updating the affinity representation improve PXR prediction?

Created: 2026-09-06. Initial protocol decisions precede this experiment's held-out predictions. This is a new adaptation comparison on existing development folds, not a new untouched holdout.

## Completed result (2026-09-07)

All 20 neural fits and 30 matched control evaluations completed. The cloud continuation and final replay finished at 19:09 UTC; the data was checksum-preserved locally before instance destruction. Read [RESULTS.md](RESULTS.md) for the independently recomputed metrics, potency trade-offs and interpretation. The original protocol and launch observations below are preserved; resource and cross-hardware-tolerance changes are recorded in the continuation artifacts.

## Purpose and model flow

Compare the same pretrained continuous heads with and without upstream affinity-module adapters. Both arms use the same real Nesso inputs, native inference realization, labels, head initialization, optimizer, objective and training schedule. The only intended architectural difference is the upstream adapters.

Preserved prepared protein/ligand state
→ frozen released structure inference, unchanged crops/refinement/recycles
→ frozen native tensor packets for both affinity members, per-record/member RNG saved
→ either frozen affinity modules or the same modules with fitted rank-4 adapters
→ fitted released continuous heads
→ arithmetic member mean and `6 - native` in FP32
→ cellular pEC50 predictions; no point clipping or affine correction.

The native quantity is log10(IC50/µM), not a released cellular-activation endpoint. PXR-label fitting makes both trained arms adapted prediction systems. Binary-head weights and the shared output module stay frozen; their outputs may change through an adapted representation. Training remains in eval mode with gradients; the structure trunk is never trained.

## Initial comparison

- Five existing strict chemical folds, draw 0, N100 and N500, initialization seed 42.
- Two arms: `head_only` and `head_plus_lora`, giving 20 neural fits.
- Each arm starts from the same released continuous-head weights in both affinity members.
- Upstream arm: rank 4 / alpha 4 at `esm_proj.1`, `esm_proj.3` and `pairformer_stack.layers.0.transition_z.fc1` in each member, as verified by the completed engineering pilot.
- Fixed 40 full-batch gradient-accumulation epochs; AdamW learning rate 0.0003, weight decay 0.0001; inverse-assay-SE weighted smooth L1, beta 0.5, SE floor 0.1.
- No learning-rate search, early stopping, outcome-based recipe selection or extra training labels. Fixed epochs bound the optimization recipe, not elapsed runtime.
- N100 uses exactly 80 FIT and 20 calibration compounds; N500 uses 400 FIT and 100 calibration compounds. Calibration labels only determine 80%/90% absolute-residual intervals after fitting. The inherited purge excludes training/calibration compounds with Morgan Tanimoto >=0.35 to any outer-test compound.
- Preserve failures and any no-benefit result. A single recipe and draw do not establish the limits of representation adaptation or its subset/initialization variability. Further draws/seeds are a separate expansion after this initial comparison, not an unbounded hidden sweep.

## Context controls

Use the fresh native scalar and a freshly fitted stable ridge on the same newly captured 768-dimensional member vectors. Reuse only the ten descriptor-rich LightGBM cells whose exact strict fold/draw/budget/identity/label/weight and source contracts match. Their ligand descriptors do not depend on Nesso's new RNG realization. Do not reuse old Nesso vectors or ridge predictions as if they were newly matched controls.

The ridge retains its existing fit-only grouped alpha selection; the descriptor comparator retains its original fit-only tuning. These are stronger contextual systems, not optimizer-matched ablations. The head-only versus upstream-plus-head comparison is the decisive representation-effect test. There are 20 new neural fits, 10 new ridge fits, 10 native evaluation cells and 10 reused descriptor cells; these are not 50 independent training experiments.

## Metrics and interpretation

Primary: inverse-assay-SE weighted MAE, with raw MAE, rank correlation, bias and potency strata (<4, >=4 and descriptive >=6) alongside it. Retain prediction compression, chemical-novelty and interval coverage/width diagnostics. Use the original residual-radius coverage predicate and retain infinite intervals where the finite-sample quantile requires them.

Report paired fold/budget differences; do not treat repeated test predictions across N, methods or historical runs as independent molecules. Existing folds and previous outcomes informed development, so this is not prospective confirmation. There is no automatic model promotion, significance threshold or post-hoc clipping repair. A useful upstream result should improve the matched head-only system consistently enough to justify its added cost; comparison with descriptor LightGBM remains a separate question. Mixed, null and potency-tail regressions remain visible.

See `EXECUTION.md` for the launched GPU0 run and the seed-isolated `code/train_v2.py` wrapper. The wrapper preserves the original trainer, isolates initialization from packet-replay RNG, and has its own source-bound training and GPU-smoke artifacts.

## Resource authorization and cost

The user explicitly authorized physical GPU0 for this run after approving completion of the previous analysis and continuation to adaptation. Use GPU0 only. GPU1 and all unrelated cofolding queues/schedules remain unchanged. No paid resources, new model downloads, image modifications or global package installs.

The completed eight-record engineering pilot measured approximately 6.89 seconds and 73.62 MB per captured record, and 9.79 GiB peak adapter-training allocations. Extrapolation to 3,344 development records is roughly 6.4 GPU-hours and 246 GB of raw packets, not a capacity or timing guarantee. Use the separate local NVMe volume rather than the project filesystem. Exact machine-local paths and GPU identity are in runtime metadata. Larger chemistry, cold disk reads, repeated packet loads, hashing and saved-state replay can increase time or memory needs.

For the initial neural matrix the inherited FIT lists imply 192,000 compound/epoch operations in total. Keep packet memory bounded; no all-cohort tensor dictionary. Persist capture records and optimizer epochs atomically so interruption does not discard completed work. Services have no arbitrary elapsed-time cap and must reconcile expected tasks on completion.

## Implementation and acceptance

- `code/cache.py`, `cache_protocol.json`, `CACHE.md`: independently bound label-free cache, including verified byte-identical reuse of the eight completed native pilot packets. All native inputs/model sources are snapshotted; no original artifacts change.
- `code/train.py`, `train_protocol.json`, `TRAIN.md`: separately bound training queue, per-epoch optimizer resume, exact role gates, frozen-state checks, native/differentiable parity and saved-state replay before task completion.
- `code/controls.py`, `controls_protocol.json`, `CONTROLS.md`: fresh native/ridge controls and verified historical descriptor reuse.
- Parent-owned launch metadata and completion report bind the actually executed commands and component hashes without making later reporting edits invalidate the scientific queues.

Run repository-wide portability/integration tests before launch. Preserve the known machine-specific-path defect in the already frozen previous baseline protocol rather than rewriting completed provenance; distinguish it from any new failure. A real production-format capture/resume smoke and optimizer/replay smoke must pass before their long stages start. Final PASS requires all declared records/tasks, zero unresolved failures, metric replay and output hashes—not merely process exit.

The completed pilot remains unchanged at the previous experiment. New execution status and results belong in this experiment's `EXECUTION.md` and run-owned artifacts.
