# GPU0 execution

Created: 2026-09-06. Updated: 2026-09-07. Final status: all 20 scientific fits, full-cache verification and final all-fit replay completed at 19:09 UTC. All data is locally preserved; the Vast continuation instance was subsequently destroyed. See [RESULTS.md](RESULTS.md) for the completed scientific analysis.

The sections below are historical launch observations, not current status. The continuation record is `artifacts/cloud/50146714/`; its final proof and preservation receipt supersede the earlier local queue state.

The user explicitly authorized GPU0. The existing native writer is running on physical device 0; GPU1 and unrelated cofolding schedules were not modified. The parent verified live CUDA utilization, growing completed-packet counts and both continuation services. At this check, 141/3,344 packets were valid (including eight byte-identically reused pilot packets), zero invalid, and 0/20 neural fits completed. These are timestamped launch observations, not live counters.

## Automatic continuation

- `nesso-pxr-affinity-comparison-v2.service` (launch PID 2018805): await the existing native writer; independently verify all cache records; train 20 matched N100/N500 models through `code/train_v2.py`; run fresh saved-state and metric replay for the entire matrix.
- `nesso-pxr-affinity-controls-v1.service` (launch PID 2020227): await the same completed native cache; generate fresh native/ridge controls and verified historical descriptor reuse; verify all 30 control evaluation cells. CPU only, with CUDA hidden.
- Both services have `RuntimeMaxSec=infinity`. Native records and training epochs are resumable. Source/input/model/output bindings are independently checked by the scientific workers; failed gates do not launch later stages.

## Seed correction before scientific fitting

Native parity evaluation restores each packet's RNG. Parent review caught that constructing LoRA afterward could therefore use the last evaluated packet's RNG rather than the declared initialization seed. The initial trainer and its completed engineering smoke were preserved. A separately bound `train_v2.py` wrapper reseeds and isolates adapter construction with `torch.random.fork_rng`; tests establish independence from preceding packet RNG and preservation of the caller stream.

The corrected wrapper passed real GPU0 two-arm, 40-epoch engineering runs on the original eight FIT-support compounds, including an epoch-2 pause/resume and fresh all-row saved-state replay. No scientific N100/N500 fit had run before this correction. Both smoke versions are explicitly engineering evidence, not held-out performance.

The first coordinator (`nesso-pxr-affinity-comparison-v1.service`, PID 2010362) is intentionally suspended so it cannot launch the superseded trainer. Its existing Docker native writer continues normally; it was paused briefly for the corrected GPU smoke and then resumed without a restart or recapture. The v2 continuation waits for that writer to finish, retires only the exact old coordinator PID, verifies the cache and proceeds. The stopped old coordinator is not evidence of a stalled native worker.

## Verification completed before long training

- Repository-wide preflight: 357 passed; one known portability failure in the completed older `protocol_baselines.json`, preserved rather than rewriting its bound evidence.
- New full-cache staging verified all 3,344 identities and source snapshots. One exceptional recovery record used an absolute symlink back into this repository; an additional read-only mount resolved the original asset without regenerating or altering it.
- Production-format capture smoke: two new records plus all eight reused pilot records, zero invalid. The same cache continues under the same binding.
- Both initial and seed-isolated AdamW engineering smokes passed pause/resume, native/differentiable parity and fresh saved-state replay. A second completed initial-smoke invocation attempted zero fits and left all task files byte-identical.
- Parent independently ran `controls verify-history`: all ten exact descriptor cells verified with zero refits.

## Locations

Large immutable cache:
`/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1`

Run root, relative to repository:
`artifacts/experiments/affinity_comparison_20260906/`

- `runtime/gpu_queue_v2/status.json`: continuation stage, PID and heartbeat.
- `runtime/controls_queue/status.json`: CPU control continuation state.
- `runtime/gpu_queue/native_capture.log`: current native writer log owned by the original coordinator.
- `runtime/gpu_queue_v2_spec.json`, `runtime/controls_queue_spec.json`: exact argv, GPU mapping, resource policy and source hashes.
- `train_v2/`: the actual 20-fit scientific training binding and eventual outputs.
- `train_v1/`: preserved superseded preparation; no scientific fits.
- `smoke_v1/`, `smoke_v2/`: separate engineering artifacts.

The cache's `progress.json` is the live native-record counter. Completion means all required worker verifiers pass, not that a service started. Results and paired scientific interpretation remain pending.
