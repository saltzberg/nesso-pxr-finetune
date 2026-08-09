# Nesso-1 PXR fine-tuning

This repository is the self-contained analysis package for adapting Nesso-1 to
human PXR pEC50 prediction. Fine-tuning clearly improves the released Nesso-1
affinity head, but the matched 2D LightGBM comparator remains stronger.

| Evaluation | Fine-tuned Nesso MAE | 2D MAE | Fine-tuned Nesso Spearman | 2D Spearman |
|---|---:|---:|---:|---:|
| Nested development (n=3,344) | 0.633 | 0.516 | 0.625 | 0.730 |
| Historical shared lockbox (n=714) | 0.601 | 0.478 | 0.602 | 0.741 |
| Public challenge (n=513) | 0.601 | 0.516 | 0.657 | 0.748 |

The challenge labels were public before this analysis. Those results are
retrospective external-dataset evidence, not a temporal blind test. See the
[scientific report](reports/model_comparison/SCIENTIFIC_REPORT.md) for the
matched design, uncertainty intervals, subgroup results, and limitations.

## Public sources

- [OpenADMET PXR challenge](https://huggingface.co/spaces/openadmet/pxr-challenge)
- [OpenADMET PXR train/test dataset](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
- [Original Nesso-1 code](https://github.com/recursionpharma/nesso)
- [Released Nesso-1 model](https://huggingface.co/recursionpharma/nesso)

The OpenADMET tables are distributed under Apache-2.0. Nesso-1 code and model
weights are also released under Apache-2.0. This repository's `LICENSE` covers
the original code and documentation here; upstream assets retain their source
licenses.

## Standalone contents

No sibling repository, local container image, or machine-specific path is
required. The exact frozen inputs used for the published comparison are under
`data/`:

- public OpenADMET challenge tables;
- frozen curation, prepared-state, and holdout-assignment tables;
- cached Nesso representations and their aligned modeling manifest;
- the reduced 2D descriptor/fingerprint table;
- historical Nesso and 2D prediction inputs used for the lockbox and
  retrospective challenge comparisons.

`data/published/SHA256SUMS.json` verifies the packaged publication inputs. The
only large asset not committed is the released 165 MB Nesso-1 checkpoint. A
repository-local downloader retrieves the exact pinned checkpoint and verifies
its checksum.

## Install and test

Python 3.11 is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,publication,train]"
pytest
ruff check src scripts tests
```

The CPU-only container test environment is built from a public Python image:

```bash
docker build -t nesso-pxr-finetune .
docker run --rm nesso-pxr-finetune
```

## Reproduce the published analysis

The fastest exact check starts from the committed out-of-fold and historical
predictions and independently regenerates the aggregate metrics, cluster
bootstrap intervals, paired differences, complementarity, stratified results,
and active-tail diagnostics:

```bash
python scripts/run_publication_comparison.py \
  --stage analyze \
  --output-dir /tmp/nesso-pxr-analysis
```

This route is CPU-only and does not need the Nesso checkpoint.

To rerun the matched 2D training from the packaged 14,325-feature table:

```bash
python scripts/run_publication_comparison.py \
  --stage 2d \
  --output-dir /tmp/nesso-pxr-2d
```

To rerun the nested Nesso cached-head training, first download the exact
released checkpoint and then use a CUDA device:

```bash
python scripts/download_nesso_checkpoint.py
python scripts/run_publication_comparison.py \
  --stage nesso \
  --output-dir /tmp/nesso-pxr-nesso \
  --device cuda
```

A full rerun performs nested Nesso training, nested 2D training, and the final
analysis in sequence:

```bash
python scripts/download_nesso_checkpoint.py
python scripts/run_publication_comparison.py \
  --stage all \
  --output-dir /tmp/nesso-pxr-full \
  --device cuda
```

All input arguments remain overridable for sensitivity analyses; their defaults
resolve from the repository root rather than the caller's working directory.

## Rebuild the leakage audit

The public and frozen derived tables needed for the identity/split audit are
included locally:

```bash
python -m nesso_pxr.audit \
  --config configs/protocol.yaml \
  --output /tmp/nesso-pxr-milestone0
```

The audit verifies compound, canonical-SMILES, Bemis-Murcko-scaffold, and
Butina-component separation across development and historical lockbox roles.

## Repository map

- `src/nesso_pxr/`: audit, split, cached-head training, and comparison code.
- `scripts/`: publication runner, challenge evaluation, site utilities, and
  checkpoint downloader.
- `data/`: packaged public, derived, and exact publication inputs.
- `reports/model_comparison/`: scientific report and all reported result tables.
- `site/`: static AetherArk project pages.
- `nesso1_pxr_finetuning_poc_spec.md`: historical proof-of-concept protocol.
