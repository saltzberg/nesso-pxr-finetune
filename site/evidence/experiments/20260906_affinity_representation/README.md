# Both-member real-input affinity adaptation pilot

Created: 2026-09-06. **Preregistration snapshot:** no GPU capture or assay-label training had run when this document was frozen. The prior analysis is now verified and published; later execution outcomes belong in `EXECUTION.md` and run-owned artifacts, not edits to this bound document. This is eight-record development engineering, not held-out efficacy evidence.

## Prediction and fitting contract

Immutable original prepared protein/ligand state → released frozen Nesso structure inference (five recycles, unchanged crop/refinement and BF16-mixed trunk) → exact per-member affinity pre-hook tensor kwargs and RNG → two pretrained affinity modules (FP32, native autocast disabled) → each released continuous head → arithmetic native member mean → `6 - mean` → assay-weighted cellular pEC50 Huber objective.

The native quantity is log10(IC50/µM), not a native cellular pEC50 endpoint. The conversion establishes regression coordinates; assay fitting makes this an adapted system.

Both arms start from identical released `affinity_heads.to_affinity_pred_value` weights in `affinity_module` and **`affinity_module2`**:

- `head_only`: update only those two continuous MLPs.
- `head_plus_lora`: update those same MLPs plus rank-4/alpha-4 LoRA at `esm_proj.1`, `esm_proj.3`, and `pairformer_stack.layers.0.transition_z.fc1` in **each** member.

The shared `affinity_out_mlp`, binary score/logit heads, all non-target affinity parameters and the entire structure trunk remain frozen. Binary **weights**, not binary outputs, are invariant: upstream representation adaptation may change binary predictions. The code reuses the existing `nesso_pxr.low_data_adapter.inject_lora` without modifying it. Installed Nesso sources remain read-only.

All eight records contribute to each of two SGD steps, learning rate 0.001, seed 42. The objective is `sum(w * smooth_l1(6-(v1+v2)/2, pEC50, beta=0.5))/sum(w)`, with `w=1/max(assay_SE,0.1)` (inverse SE, not inverse variance; finite strictly positive SE required; zero or negative SE fails). No tuning, calibration, early stopping, efficacy comparison, clipping or held-out labels enter. The 0.1 floor is the previous engineering cap, not measured assay precision.

Models stay in **eval mode with gradients enabled**. Capture uses unchanged native kernels; replay/training uses `use_kernels=False`, preserving every other tensor/kwarg and restoring each member's own captured CPU/CUDA RNG on every call. Exact native/differentiable parity is a hard gate before and after training, including representation and binary logits. Zero-LoRA and disk reload must be exact too. Nonzero gradients are required for all intended tensors except initial zero-B LoRA A gradients; all intended tensors must change after two steps. Failure is retained as technical evidence, never a scientific result.

## Identity and cache version

`protocol_pilot.json` freezes the first eight FIT-role rows, in existing strict fold0/draw0/N100 file order. It stores their literal record IDs. Stage independently joins original row indices to record/feature identities and confirms development membership and exclusion from the inherited chemical-cluster outer test. Outcomes are not used in selection. Source labels are snapshotted, but only `train` consumes them as numerical outcomes.

This is a **new upstream realization**, deliberately distinct from historical seed42 sequential caches. Each singleton native prediction resets a SHA256-derived per-record seed; restart order cannot shift the next record's stream. The loaded full model and Trainer persist across records. Both arms consume the exact same packets. Hooks capture both actual attributes during the same native prediction, each with its own RNG state, native outputs and exact tensor kwargs. Non-tensor writer bookkeeping is explicitly listed as omitted.

Stage snapshots repository code, protocol, plan, tests, immutable reused adapter/audit source, all installed Nesso Python sources, checkpoint files, CCD, source identity/label tables, selected prepared records/structures/conformers and ESM assets. Source bytes and hashes must match on every later invocation. Machine-specific resolved source paths occur only in artifact metadata. Snapshot copies are not hardlinks.

Each captured record is committed by atomic directory rename after fsync of all tensor payloads and a hash/identity manifest. Existing records are hash/schema/binding checked before skipping. Incomplete hidden temporary directories are not completion records. Training is resumable **by complete arm**: an interrupted unfinished two-step arm restarts from released weights; a completed arm is replayed without fitting. A run lock excludes concurrent writers. Separate failure JSONs retain exceptions. `verify` checks exact expected capture coverage; `--require-train` additionally requires both states and hash-bound disk replay proofs. `--replay --allow-gpu` repeats fresh saved-model replay without fitting.

## Commands (parent-controlled)

Run from repository root. Supply `PXR_REPO` and `ADMET_PXR_ASSET_ROOT` as machine-local absolute paths. The existing historical command is retained at `artifacts/experiments/low_data_20260905/adapter_pilot/command.json`; its external-project mount remains necessary because reused assets contain links into that project. Mount it read-only at its original absolute path. No global installs or network access are needed. Finalize the README, plan, tests, protocol and code before staging: all are bound and later edits intentionally fail validation.

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_affinity_representation_pilot.py
python experiments/20260906_affinity_representation/code/pilot.py --help

: "${PXR_REPO:?absolute repository path}"
: "${ADMET_PXR_ASSET_ROOT:?absolute historical external asset project path}"
RUN_REL=artifacts/experiments/affinity_representation_20260906/pilot_v1
mkdir -p "$PXR_REPO/$RUN_REL"
IMAGE=sha256:e3519ab0faa11d098f14f57b628530c21df748ab7c632f5b071754548e058a24
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
COMMON=(--rm --network none --read-only --tmpfs /tmp:rw,size=2g
  --tmpfs /root/.cache:rw,size=1g --cpus 4 --shm-size 1g
  -v "$PXR_REPO:/repo:ro" -v "$PXR_REPO/$RUN_REL:/run:rw"
  -v "$ADMET_PXR_ASSET_ROOT:$ADMET_PXR_ASSET_ROOT:ro"
  -w /repo -e PYTHONPATH=/repo/src:/repo -e PYTHONDONTWRITEBYTECODE=1
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e MKL_NUM_THREADS=4
  --entrypoint python)
CLI=experiments/20260906_affinity_representation/code/pilot.py
# CPU-only immutable staging, including input/source snapshot size check:
docker run "${COMMON[@]}" "$IMAGE_ID" "$CLI" stage --output /run --image-id "$IMAGE_ID"

# ONLY after parent prior-analysis signoff; expose one available physical GPU:
GPU_DEVICE=1  # Operational amendment: GPU0 is occupied by unrelated ESMFold2.
docker run --gpus "device=$GPU_DEVICE" "${COMMON[@]}" "$IMAGE_ID" "$CLI" capture --output /run --allow-gpu
docker run --gpus "device=$GPU_DEVICE" "${COMMON[@]}" "$IMAGE_ID" "$CLI" train --output /run --allow-gpu
# CPU integrity verification (does not repeat GPU numerical replay):
docker run "${COMMON[@]}" "$IMAGE_ID" "$CLI" verify --output /run --require-train
# Optional independent GPU replay, no fitting:
docker run --gpus "device=$GPU_DEVICE" "${COMMON[@]}" "$IMAGE_ID" "$CLI" verify --output /run --require-train --replay --allow-gpu
```

Use a supervised foreground process/service for actual launch and preserve its exact argv, image ID, PID and GPU mapping in runtime metadata. The script does not assert physical GPU identity from CUDA's renumbered device index: the parent must expose only the physical device recorded in runtime metadata. The short pilot is assigned to GPU1 after checking that it is free and its unrelated Boltz supervisor is deferred until 18:00 America/New_York; no other job is stopped or modified. The exact image digest is protocol-bound. Source/protocol edits after stage intentionally invalidate continuation; use a new artifact directory rather than repairing a bound run.

## CPU verification (2026-09-06)

- Focused pilot suite: **54 passed** using the existing `cheminf` Python, with `CUDA_VISIBLE_DEVICES=''`; pilot plus existing adapter suite: **67 passed**. Python AST parsing and protocol JSON parsing passed.
- Real source-table checks replay the exact eight literal FIT IDs and original row order without reading outcomes for selection. A separate loss test uses the selected real assay labels with synthetic scalar predictions, without fitting a model.
- Synthetic CPU checks cover strict-role/identity rejection, nonpositive/nonfinite assay-SE rejection, inverse-SE floor and weighted Huber conversion, canonical/tensor hashes, nested tensor copying, both-member matched sequential heads and LoRA, actual two-step synthetic loss gradients/updates, frozen weights/buffers, exact disk reload, parity failures, immutable packets and corruption, source/snapshot drift, and the CLI GPU opt-in guard.
- A no-GPU, read-only invocation in the existing image passed `--help` and read-only asset discovery: all eight identities replayed; 25 selected processed files (records, structures, conformers and ESM) were accessible, as were checkpoint/CCD assets and 37 installed Python sources. Image inspection matched the protocol's pinned digest.
- **No canonical stage was created, no real Nesso forward pass was run, and no GPU capture or training was executed.** Stage tests use temporary synthetic assets; they do not validate real-model feasibility or held-out efficacy. Parent-owned final staging and GPU acceptance remain outstanding.

## Costs and deliverables

Capture manifests record actual payload bytes, elapsed native capture time and allocated/reserved peak VRAM for every record. Training manifests record full weighted record-step count, wall-time throughput, measured allocated/reserved peak VRAM, all gradient norms, intended tensor changes and initial continuous-head hashes. Saved states retain per-tensor intended before/after and frozen hashes. Saved outputs retain each member's baseline, zero-update, adapted differentiable and adapted native-kernel outputs. Timing includes record hash verification, disk loading and host/device transfer; prediction throughput explicitly counts both native and differentiable passes. These are not GPU-kernel-only benchmarks.

Review measured packet sizes against available filesystem space before proposing any bulk capture. No projected throughput, storage total or efficacy score is fabricated here. CPU synthetic tests validate contracts, not real-model feasibility; only the parent-supervised real execution can satisfy the pilot's GPU acceptance criteria.
