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

## Next gate

GPU feature extraction is deliberately gated on the protocol and split tests.
The extractor will cache only the two 384-dimensional affinity representations,
not Nesso's much larger pairwise tensors.

The next iteration is a 16-compound GPU smoke test that must prove member-1 and
member-2 feature capture, frozen-score parity, deterministic reruns, failure
logging, and a versioned feature schema before the 4,134-compound E0 pass.
