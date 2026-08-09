# Reproducibility data

This directory makes the published comparison independent of any sibling
checkout. It contains three kinds of inputs:

- `openadmet/`: the public OpenADMET PXR challenge tables used by this study.
- `derived/`: frozen curation, prepared-state, and historical holdout-assignment
  tables produced before the publication comparison.
- `published/`: the exact cached Nesso representations, 2D feature table,
  manifests, and historical prediction inputs used to generate
  `reports/model_comparison/`.

The OpenADMET files originate from
<https://huggingface.co/datasets/openadmet/pxr-challenge-train-test> and are
distributed there under Apache-2.0. The derived tables contain only
transformations, annotations, or model outputs based on those public challenge
records. `published/SHA256SUMS.json` records the exact packaged input bytes.

The released Nesso-1 checkpoint is intentionally not committed because it is
about 165 MB. Run `python scripts/download_nesso_checkpoint.py` to retrieve the
exact, checksum-verified model file used by the study.
