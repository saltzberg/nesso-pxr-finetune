# Affinity-module LoRA: real-input engineering test

A rank-4 adapter successfully updated the first Nesso affinity member upstream of its final readout. This establishes a differentiable implementation path, not predictive improvement on PXR activation.

## Model flow

A previously processed development compound and its PXR inputs → released Nesso structure path (frozen) → captured real affinity-module inputs → affinity-module ESM projections and first Pairformer transition (LoRA fitted for an engineering test) → released continuous/binary readouts (frozen).

The test used record `nesso_0ba75904a03a269a6afc`. No assay label entered the objective. Two SGD steps targeted the detached native continuous output plus one; those outputs are not endpoint predictions for the scientific comparison.

The injected targets were `esm_proj.1`, `esm_proj.3`, and `pairformer_stack.layers.0.transition_z.fc1` in `affinity_module`, with rank 4 and alpha 4. The second affinity member was not adapted in this test.

## Verified observations

- 12,288 trainable adapter parameters.
- Zero-update predictions matched the unmodified differentiable path exactly.
- The differentiable PyTorch path matched captured native inference exactly for the tested continuous output, binary logit and representation.
- Intended adapter gradients were finite, with nonzero B gradients at the initial zero-B state and nonzero A/B gradients at the next step.
- All adapter tensors changed; frozen parameter hashes were unchanged.
- Saving and reloading the adapter reproduced the updated outputs exactly.
- An independent parent replay reproduced the same adapter SHA-256 and numerical observations.

The successful child attempt and independent replay are preserved under `artifacts/experiments/low_data_20260905/adapter_pilot/attempt2/` and `parent_replay/`. The initial failed capture attempt is retained in `attempt1/`; non-tensor bookkeeping from the dataloader had to be excluded from the captured module inputs.

## Runtime and limitations

The pilot used the existing Nesso 1.0.0 container (exact local image tag in the saved command), one local GPU, four numerical CPU threads, no network, and read-only source mounts. Native capture used its existing kernel path; differentiable replay used `use_kernels=False`, with the evaluation/crop policy unchanged and the captured RNG state restored for parity checks.

This is a single-record, first-member, two-step feasibility test, not full-model fine-tuning or a low-data experiment. No representation-level adaptation learning curve has been run. Broader execution needs a frozen upstream-input cache for the selected training and test rows, both-member design, measured extraction/storage costs, and the same label-budget accounting as the cached-head matrix. Head LoRA in that matrix remains a different, final-readout intervention.

## Reproduction

Use `scripts/audit_low_data_adapter.py --help`. A verified replay reads the preserved `real_affinity_inputs.pt` via `--inputs`, selects a fresh `--output-dir`, and uses `--device cuda`. Exact Docker arguments are in `adapter_pilot/replay_command.json`; the independent replay changed only the output directory and container name. Results and input/adapter hashes are recorded in each attempt's `pilot.json`.
