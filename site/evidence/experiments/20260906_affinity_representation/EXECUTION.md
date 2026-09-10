# Both-member adapter pilot: execution results

Created: 2026-09-06. Engineering pilot completed and parent-audited; no held-out efficacy result.

The same released continuous heads were trained with and without upstream rank-4 adapters in both Nesso affinity members. Eight predetermined real FIT-role compounds contributed assay labels to two weighted SGD updates. Neither the structure trunk nor binary-head weights were updated. Native output construction, crops, FP32 affinity evaluation and per-record/member RNG replay stayed fixed.

## Result

- Eight of eight real records captured, both affinity members present.
- Both supervised arms completed. All intended tensors had nonzero gradients at the final step and changed from initialization; frozen weights and buffers were unchanged.
- The head-only and head-plus-adapter arms had identical initial continuous-head hashes.
- Native-versus-differentiable, zero-update and fresh saved-state replay maximum absolute differences were all exactly zero for the checked outputs.
- Parent rehashed 28 captured/trained payloads and checked both fresh replay proofs. No held-out predictive advantage is inferred from this pilot.
- The service completed normally: `nesso-pxr-representation-pilot-v1.service`, final `MainPID=0`, `active/exited`, exit status zero. Runtime started 2026-09-06T17:30:36.728550+00:00 and finished 2026-09-06T17:32:09.168675+00:00.

## Measured cost and scale limits

| Quantity | Observed |
|---|---:|
| Native capture per record, both members | 6.89 seconds |
| Raw captured payload per record, mean / maximum | 73.62 / 79.77 MB |
| Native capture peak allocated VRAM | 1.02 GiB |
| Head-only training throughput | 8.48 record-steps/second |
| Head-plus-adapter training throughput | 4.73 record-steps/second |
| Head-plus-adapter peak allocated VRAM | 9.79 GiB |
| Trainable parameters: head-only / head-plus-adapter | 592,130 / 616,706 |

These timings include packet verification, disk reads and transfers. They are measured on physical GPU1 while an unrelated ESMFold2 workload used GPU0, not an isolated whole-machine benchmark. The small deterministic eight-record panel does not bound memory or disk use for all chemistry. GPU allocator peaks are not total board memory use.

Straight-line extrapolation to all 3,344 development records gives approximately 6.4 GPU-hours of capture and 246 GB of raw packets, before downstream fitting and verification. This is an estimate, not completed work or a capacity guarantee. The project filesystem had only 174 GB free; the separate local NVMe filesystem had approximately 1101 GB free. Bulk capture must use the larger volume or a verified lossless storage design, not fill the project disk.

## Execution and test scope

The prior-run analysis and LAN report were finished and verified before capture or supervised training began. No image/source patches, new packages, network model calls or paid resources were used. GPU0 became occupied, so a pre-staging operational amendment used the idle GPU1. The unrelated Boltz supervisor was inspected read-only and is scheduled to resume on GPU1 at 18:00 America/New_York; it was not changed. The pilot is finished and no longer uses that GPU.

Repository-wide preflight: 281 passed, one known portability failure in the already frozen `protocol_baselines.json` from the completed older experiment. New pilot code/docs/protocol passed their portability scan. The frozen historical protocol was preserved. The focused pilot suite has 54 CPU tests; these are distinct from the successful real GPU acceptance checks above.

## Evidence

Run root: `artifacts/experiments/affinity_representation_20260906/pilot_v1/`.

- `parent_audit.json`: independent payload, gradient, initialization, invariance and replay checks.
- `stage/manifest.json` and `stage/snapshot/`: immutable protocol, code, installed model sources, checkpoint and input snapshots.
- `capture/<record_id>/manifest.json`: both-member native packets, identity, RNG, bytes and timing.
- `train/<arm>/manifest.json`, `state.pt`, `outputs.pt`: supervised state and verification evidence.
- `train/<arm>.replay.json`: separate fresh GPU saved-state replay.
- `runtime/status.json` and stage logs: exact commands, image identity, physical GPU, supervisor PID, durations and return codes.

## Next-stage gate

The engineering feasibility gate passes. The remaining question is held-out utility, not whether gradients can reach the intended modules. A new N100/N500 protocol must keep both arms matched, count tuning/calibration inside each label budget, retain native/stable-ridge/descriptor controls only under matching contracts, and report existing outer-fold reuse as development evidence. Shared-GPU scheduling and bulk-cache placement must be settled before a long queue; approximately 9.8 GiB of adapter-training allocations make overlapping the scheduled Boltz run unsafe to assume.
