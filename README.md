# Nesso-1 PXR fine-tuning

This repository implements the proof of concept in
`nesso1_pxr_finetuning_poc_spec.md`. The local ADMET-PXR checkout is an
immutable data dependency; generated code and experiment artifacts live here.

The first iteration is intentionally CPU-only and fast. It freezes the label
contract, constructs joint compound identities, and creates five development
folds by Butina clustering while enforcing intact Bemis-Murcko scaffold groups.
The existing scaffold lockbox remains immutable.

## Fast test loop

```bash
PYTHONPATH=src /home/dan/projects/ADMET-PXR/.venv312/bin/python -m pytest
```

The reproducible container is derived from the pinned local Nesso image:

```bash
docker build -t local/nesso-pxr:0.1.0 .
docker run --rm local/nesso-pxr:0.1.0 pytest
docker run --rm local/nesso-pxr:0.1.0 ruff check src tests
```

## Milestone 0 audit

```bash
PYTHONPATH=src /home/dan/projects/ADMET-PXR/.venv312/bin/python \
  -m nesso_pxr.audit \
  --config configs/protocol.yaml \
  --output artifacts/manifests/milestone0
```

The current pinned sources produce 11,361 joint development compounds in five
folds of 2,272--2,273 compounds. The 3,072 development DRC IDs exactly match the
established Emax >= 0.6 universe. The current files yield 714 eligible lockbox
rows; an older strategy note reports 718, so this discrepancy remains explicit.

The audit requires zero canonical-compound, original-ID, Bemis-Murcko-scaffold,
or Butina-component overlap across roles. It also verifies that all 513 blinded
activity rows remain label-free. The full split build takes about 20 seconds and
peaks near 5.7 GB RAM because standard RDKit Butina materializes the condensed
pairwise distance matrix; unit tests remain sub-second.

## GPU feature smoke test

The bounded GPU gate selects 16 DRC compounds across all five scaffold-aware
folds and the observed potency range. It runs Nesso through the same
`lightning.Trainer.predict` path as the upstream CLI, temporarily captures both
384-dimensional pre-regression affinity representations, and records a
versioned schema plus reason-coded failures.

The 2026-08-06 smoke run passed on an NVIDIA GeForce RTX 5070 Ti:

- 16/16 examples succeeded with no failures.
- Both reconstructed member scores agreed with Nesso within `1.2e-7`; the
  captured ensemble output agreed with a fresh upstream CLI run within
  `2.3e-16`.
- Two independent runs produced bit-identical tensors and identical
  safetensors SHA-256 values.
- Each run took about 63 seconds for prediction and peaked near 1.08 GB of GPU
  memory.

Exact score parity depends on preserving Nesso's seed (`42`), worker count
(`1`), five recycling steps, and `bf16-mixed` Lightning execution. Generated
smoke inputs, references, features, metadata, schemas, and determinism reports
live under `artifacts/experiments/feature_smoke/` and remain git-ignored.

The next gate is the 4,134-compound E0 frozen-feature pass using this exact
capture path.
