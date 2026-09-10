# Matched both-member affinity adaptation trainer

Created 2026-09-06. This file describes implementation, not completed scientific fits.

## Fixed prediction and fitting contract

Cached real native affinity inputs → both released continuous heads → arithmetic
mean of native member values → `6 - mean` → adapted cellular pEC50 prediction.
Both arms start with the same released heads. `head_only` updates those heads;
`head_plus_lora` also updates rank-4, alpha-4 adapters at each member's `esm_proj.1`,
`esm_proj.3`, and `pairformer_stack.layers.0.transition_z.fc1`. All other weights
and buffers are frozen. The old immutable pilot's configure/forward/hash helpers
are reused without changing its code or installed Nesso sources.

The scientific matrix is five inherited strict chemical folds × draw 0 × N100/N500
× two arms × seed 42 = 20 fits. N counts FIT plus calibration: 80/20 and 400/100.
Only FIT labels enter optimization. Each of 40 epochs accumulates the complete
normalized inverse-SE-weighted Huber gradient, beta 0.5, before one AdamW step
(lr 0.0003, weight decay 0.0001, default betas/epsilon explicitly set). Weight is
`1/max(SE,0.1)`; invalid target/SE fails. No tuning, early stopping, clipping,
mini-batch updates, dropout/train-mode policy, or fallback model is permitted.

Calibration labels enter only after final state and label-free predictions are
saved, solely to set unweighted absolute-residual split-conformal radii at 80/90%.
The finite-sample rank is `ceil((n_cal+1)*coverage)`. Test coverage uses the original
predicate `abs(y-pred) <= radius`, not rounded interval endpoints. Calibration does
not shift predictions. Saved tables include both member native values, prediction,
assay labels/SE/weight, and test interval endpoints. Metrics include weighted/raw
MAE, Spearman, bias, effective weighted N, fixed potency strata, radius, coverage,
and width. Existing pure low-data metric and interval function bodies are loaded
via a restricted AST selection to avoid importing pandas/sklearn, absent from the
pinned inference image; the complete source bytes remain bound and snapshotted.

These reused outer development folds are **not a new untouched holdout**. Purged
FIT/calibration membership is inherited unchanged, with measured nearest-test
Morgan similarity strictly below 0.35. This fixed matched optimization recipe
isolates upstream representation adaptation; it is not the old tuned readout recipe.
Parent analysis joins real historical descriptors and fresh native controls later.
This trainer supplies no invented reference values or performance claims.

## Commands

Run inside the existing pinned image with repository mounted, working directory at
the repository root, and `PYTHONPATH=src:.`. Image digest:
`sha256:e3519ab0faa11d098f14f57b628530c21df748ab7c632f5b071754548e058a24`.

```sh
python experiments/20260906_affinity_comparison/code/train.py prepare --cache /cache --output /run
python experiments/20260906_affinity_comparison/code/train.py run --cache /cache --output /run --allow-gpu
python experiments/20260906_affinity_comparison/code/train.py verify --cache /cache --output /run
python experiments/20260906_affinity_comparison/code/train.py verify --cache /cache --output /run --allow-gpu --replay
```

Expose exactly one device (parent authorizes/selects physical GPU0). The runner
refuses GPU work without `--allow-gpu` and exactly one visible CUDA device.
`prepare` never loads a model or executes GPU work. No automatic image changes,
installs, canonical cache staging, or GPU launch are part of CPU tests.

`--task-limit 1`, `--fold 0`, `--budget 100` select queue work without changing the
full run binding. `--stop-after-epoch 2` pauses operationally after an atomic epoch;
it does not change the 40-epoch recipe or mark a task complete. Resume with the
same command minus that pause flag. `verify` exits 2 if the matrix is incomplete.

### Separate real eight-record engineering smoke

```sh
python experiments/20260906_affinity_comparison/code/train.py prepare --smoke --cache /pilot --output /smoke
python experiments/20260906_affinity_comparison/code/train.py run --smoke --cache /pilot --output /smoke --allow-gpu --task-limit 1 --stop-after-epoch 2
python experiments/20260906_affinity_comparison/code/train.py run --smoke --cache /pilot --output /smoke --allow-gpu --task-limit 1
```

`/pilot` is the completed old eight-record pilot output. Smoke uses those records
in frozen order as four FIT, two calibration, two replay/test-role records, all
originally selected from old FIT support. This is exclusively an optimizer/resume/
serialization exercise, not a scientifically comparable N6 fit or held-out result.
It has a distinct binding, smoke-prefixed task IDs, two engineering tasks, and
always reports zero scientific fits. Repeat the final command to train the second
arm; completed tasks are byte-preserving skips. Never reuse smoke results as any
of the 20 scientific fits.

## Integrity and resource behavior

The separate run snapshots trainer, protocol, tests, this document, old helpers,
strict manifests/tables, labels, and the full cache lock/binding. Public source
keys are repository-relative. All source drift, foreign bindings, corrupt payloads,
noncontiguous epochs, and changed immutable packets fail closed. Source snapshots
are not an adversarial authenticity mechanism.

Each record's original per-member CPU/CUDA RNG is restored on every call by the
old pilot helper. Native kernels and differentiable evaluation must agree exactly
on **every calibration/test row**, before training, at zero update, after fitting,
and during a fresh-model saved-state replay. Native scalar baselines are durable.
Frozen hashes include buffers and are checked before/after and on restore. Intended
gradients are finite every epoch, with nonzero coverage recorded for the first two;
zero-initialized LoRA A gradients may be zero only on epoch one. Every intended
parameter must change by the final epoch.

Epoch state includes all trainable parameters and AdamW state, bound to the task,
with frozen pretrained state reconstructed from the immutable cache checkpoint.
Each epoch directory is hash-checked and atomically renamed with its manifest.
Only completed epoch directories are resumable; unfinished temporary directories
are retained/ignored. No full-model frozen snapshot is duplicated per epoch.
Completion requires CSV float roundtrip, independently reconstructed score rows,
exact all-row fresh saved-state replay, final frozen invariance, and explicit
output hashes. Durable event and failure records distinguish crashes, pauses,
completed fits and pending tasks; OOM remains a failure with no fallback.

The reader retains **only hash/stat validation metadata**, not a corpus of tensors.
On first access per task it validates the full cache packet through the cache API;
later accesses reload member tensors with inode/size/mtime guards. This assumes
trusted immutable local cache storage during a task lifetime. Each new task/process
or independent replay rehashes on first access. It avoids hashing roughly 73 MB per
record every epoch and avoids holding hundreds of packets in RAM. NVMe traffic
and largest-record VRAM remain real bottlenecks; pilot throughput is not a promise
of matrix throughput. No record is silently dropped or replaced on memory failure.

## CPU validation

```sh
PYTHONPATH=src:. python -m pytest -q tests/test_affinity_comparison_train.py
python experiments/20260906_affinity_comparison/code/train.py --help
```

CPU fixtures are explicitly synthetic and never published as scientific cache or
results. They exercise exact-N/disjoint membership and outcome-blind selection,
fixed matrix/recipe, loss conversion and normalization, inherited scoring/interval
APIs, atomic checkpoint integrity, interrupted/resumed AdamW equality, and real
serialized all-row scoring/replay with small in-memory test networks. Real native
Nesso parity and memory acceptance require the parent-run GPU smoke.
