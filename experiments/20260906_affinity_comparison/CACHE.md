# Full-development native affinity cache

The cache contains the exact native inputs and outputs of **both released affinity members** for the original 3,344 development rows. It performs no fitting, outcome conversion, assay-value parsing, calibration, or held-out evaluation. Cohort selection uses original `artifacts/experiments/low_data_20260905_cut035_run2/prepared/inputs.csv` order (`row_index` 0–3343), joined exactly to `data/published/modeling_manifest.csv` development record IDs, feature rows and canonical SMILES. Outcome-bearing CSVs are hashed but **not copied** into this cache. The separate identity manifest contains only identity/order/seed fields.

## Bound and unbound sources

`stage/manifest.json` has a distinct `pxr-affinity-comparison-cache-v1` binding. Stage snapshots bind:

- `code/cache.py`, `cache_protocol.json`, and `tests/test_affinity_comparison_cache.py`;
- the original pilot helper and its protocol;
- all installed Nesso Python sources from the pinned image;
- released checkpoint files, CCD, original processed record/structure/conformer assets and ESM embeddings;
- exact input-table hashes and an explicit, outcome-free identity manifest.

Comparison trainer code, future protocols, README and this operational document are deliberately **not bound**. They may develop while capture runs. Changes to bound code/tests or original prepared assets require a new stage, not mutation of a live cache. Validation compares snapshot hashes and current live sources. No installed Nesso source is modified.

Image: `sha256:e3519ab0faa11d098f14f57b628530c21df748ab7c632f5b071754548e058a24`.

## Commands

Execute inside that pinned image with the repository mounted read-only at `/repo`, the parent-owned cache mounted read-write at `/cache`, and every external historical-asset symlink target mounted at its original target path read-only. Set working directory `/repo` and `PYTHONPATH=/repo/src:/repo`. The parent alone chooses the physical device and launches Docker. Stage and verify need no GPU. Capture requires exactly one exposed GPU and explicit approval.

```sh
python experiments/20260906_affinity_comparison/code/cache.py stage \
  --output /cache \
  --image-id sha256:e3519ab0faa11d098f14f57b628530c21df748ab7c632f5b071754548e058a24

# Production-format smoke: import all eight pilot records, then at most two NEW records.
python experiments/20260906_affinity_comparison/code/cache.py capture \
  --output /cache --allow-gpu --limit 2

# Resume the SAME full-cohort stage, skipping only validated complete records.
python experiments/20260906_affinity_comparison/code/cache.py capture \
  --output /cache --allow-gpu

python experiments/20260906_affinity_comparison/code/cache.py verify \
  --output /cache --require-complete
```

`--limit` is a positive cap on newly captured records for one invocation, **not a cohort size or alternate protocol**. Verification without `--require-complete` permits an explicitly `INCOMPLETE` cache; it never labels partial coverage `PASS`. Stage imports the eight pilot records using CPU only. Capture uses one persistent full `Nesso1` and one persistent Lightning `Trainer`; singleton records receive the old pilot's exact per-record seed namespace/base, native crops/refinement/recycles, highest matmul precision, BF16-mixed trunk and FP32 native affinity calls. Each member payload retains its independently captured CPU/CUDA RNG state for replay.

## Layout and pilot reuse

```text
stage/manifest.json
stage/identity_manifest.json
stage/snapshot/<repo-relative-source-or-image-path>
capture/<record_id>/affinity_module.pt
capture/<record_id>/affinity_module2.pt
capture/<record_id>/native_prediction.pt
capture/<record_id>/manifest.json
progress.json
events.jsonl
failures/<record_id>.json
.cache.lock
```

The actual historical filename **`native_prediction.pt`** is retained, per the parent API clarification; there is no `native.pt` alias. All three imported payload files are copied byte-for-byte, never reserialized. The new cache manifest necessarily has the new binding and full-cohort record order. Its `reused_from` field preserves the original binding, original record (including original pilot role/order), original manifest SHA-256, and source path. The stage lock also embeds the original parsed manifest and its SHA-256. Thus original manifest identity is preserved without pretending that new cache metadata is byte-identical.

Reuse requires all eight completed native pilot captures, exact old binding, matching source/checkpoint/CCD/prepared-asset hashes, identical native/precision/RNG semantics and exact identity fields including original `row_index`. Any mismatch is a `BLOCKER`; pilot records are never silently recaptured. Source pilot artifacts are read-only. Atomic copy/rename or the proven pilot packet publisher prevents interrupted payloads from becoming completed records.

Resume validates every existing packet's hash, byte count, exact file/member schema, binding, identity, precision/kernel policy, RNG states, tensor finiteness and native ensemble consistency. Corrupt packets are not overwritten. A per-record failure is retained in a durable failure ledger; unresolved failures prevent PASS. Hidden interrupted temporary directories are not completed records. `progress.json` partitions all 3,344 IDs into valid/missing/invalid, separately reports unexpected IDs and unresolved failures, and records reused counts. Events are append-only, flushed and fsynced. Commands are serialized with a nonblocking filesystem lock.

## Python API

Import `experiments/20260906_affinity_comparison/code/cache.py` with `importlib.util`.

- **`validate_stage(output) -> lock`**: CPU-only full immutable source/protocol/identity validation. The dictionary exposes `binding`, `protocol`, `records`, `files`, `reused`, and source identity proofs. `output` accepts a string or `Path`.
- **`load_model(lock, output) -> torch.nn.ModuleDict`**: loads both released affinity members using the original helper's `load_members(snapshot_checkpoint, protocol)`, on CUDA in eval mode. It does **not** return full `Nesso1`, configure adapters, select trainable parameters, or fit. Capture loads full `Nesso1` internally.
- **`load_record_packet(output, lock, rid) -> dict[str, payload]`**: CPU hash/schema/identity validation followed by safe tensor deserialization. Keys are `affinity_module`, `affinity_module2`. Each original-format payload contains `kwargs`, `cpu_rng`, `cuda_rng`, `autocast_enabled`, `removed_bookkeeping`, and `native_output`. Outputs include `affinity_pred_value`, `affinity_repr`, and `affinity_logits_binary`. A trainer may wrap this validated loader in its own bounded first-access cache; this implementation deliberately revalidates each API call rather than silently trusting changed files.
- **`verify_cache(output, require_complete=True) -> report`**: full CPU verification; writes exact accounting to `progress.json`; raises on corrupt/foreign/unresolved failures and, by default, missing records. Set `require_complete=False` for smoke/resume checks. Partial valid results have status `INCOMPLETE`.

Differentiable replay uses the old helper's **`forward(models, packet, kernels=False)`**, not a nonexistent `forward_member` function. It restores each member's saved RNG, disables affinity autocast, and changes only native kernel use for differentiability. Cache generation does not train models; future control/readout code may consume `affinity_repr` under its separate scientific protocol.

## CPU verification

```sh
PYTHONPATH=src:. python -m pytest -q tests/test_affinity_comparison_cache.py
```

The focused tests exercise the real ordered 3,344-row identity join, semantic/source parity against immutable pilot files, CPU validation and byte-identical copying of all eight actual pilot payload sets into temporary storage, corruption refusal, unchanged resume markers, failure accounting, the model-loader API and the CLI GPU permission gate. These CPU tests are not a production GPU smoke or proof that all 3,344 captures completed. Only the parent may stage the canonical destination or launch the approved GPU.
