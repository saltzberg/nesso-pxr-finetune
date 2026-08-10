# Cached Nesso-1 regression-head adaptation for PXR activity

This repository asks a narrow question: **does adapting Nesso-1 improve human
PXR activity prediction, and does it compete with an established 2D molecular
model?**

The answer is **yes, then no**. Updating Nesso-1's final regression heads from
cached molecular representations substantially improves on frozen Nesso-1, but
a matched 2D LightGBM model remains more accurate. The study is retrospective
and supports an assay-prediction result, not a claim of improved physical
binding-affinity prediction.

> **Evidence status:** nested development is the primary matched comparison.
> The historical lockbox has been opened, and the OpenADMET challenge labels
> were public before this analysis. The challenge result also has an unresolved
> cached-feature provenance warning and should be treated as provisional until
> that discrepancy is resolved.

## Study at a glance

This work addresses the **activity-prediction track** of the OpenADMET PXR
challenge, not the structure-prediction track. The measured endpoint is
functional human PXR activation reported as pEC50.

```text
4,139 raw PXR dose-response records
└── 4,134 curated and weighted compounds
    ├── 3,344 development compounds → five nested chemical-family folds
    └──   790 historical scaffold-lockbox compounds
          └── 714 Emax-qualified compounds shared by the historical models

513 public challenge compounds → separate retrospective external dataset
```

Development folds keep Bemis-Murcko scaffold groups and connected
Morgan-fingerprint/Butina components intact. The 714-compound lockbox comparison
is the Emax-qualified intersection for which both historical Nesso and 2D
predictions are available; the remaining 76 lockbox records have lower Emax.

## What was actually trained

This is **cached regression-head fine-tuning**, not end-to-end Nesso-1
fine-tuning.

- Nesso-1 first supplies two cached, 384-dimensional affinity representations
  per compound.
- Only the two final `384 → 384 → 384 → 1` regression MLPs are updated: 592,130
  trainable parameters combined.
- The cofolding trunk and all other parameters in the affinity pathways remain
  frozen. The packaged comparison starts from cached representations rather
  than rerunning the Nesso trunk.
- Inner folds choose learning rate, weight decay, and training duration. Each
  outer-fold prediction averages three refits with seeds 42, 43, and 44.
- Training uses the pretrained head initialization and a Huber objective over
  both member predictions and their mean. The selected outer-fold models use
  26–28 epochs.

Nesso's released affinity value is IC50-like and uses a different numerical
orientation from pEC50. Training and prediction use:

```text
Nesso-scale training target = 6 - observed pEC50
predicted pEC50            = 6 - affinity_pred_value
```

This transform aligns numbers; it does **not** make the endpoints biologically
equivalent. Cellular functional PXR EC50 also reflects efficacy, permeability,
receptor context, and assay behavior.

## Primary matched result

All four development estimates use the same 3,344 compounds and identical five
outer chemical-family folds. Confidence intervals are 95% intervals from 2,000
paired scaffold/Butina-component cluster-bootstrap replicates.

| Model | MAE (95% CI) | RAE | Spearman (95% CI) |
|---|---:|---:|---:|
| Outer-training-fold mean | 0.925 (0.874, 0.992) | 1.002 | -0.075 (-0.206, 0.032) |
| Frozen Nesso-1 | 0.915 (0.848, 1.007) | 0.991 | 0.385 (0.318, 0.453) |
| Fine-tuned Nesso-1 regression heads | 0.633 (0.611, 0.654) | 0.686 | 0.625 (0.573, 0.677) |
| Matched nested 2D LightGBM | **0.516 (0.498, 0.532)** | **0.559** | **0.730 (0.688, 0.771)** |

Relative to frozen Nesso, fine-tuning reduces MAE by 0.282 pEC50 units (95% CI
0.225–0.358) and increases Spearman by 0.240 (0.211–0.272). Relative to matched
2D, fine-tuned Nesso has MAE higher by 0.117 (0.101–0.133) and Spearman lower by
0.105 (0.084–0.131).

The mean baseline is recomputed from each outer-training partition. Its
out-of-fold predictions are therefore fold-specific constants; their variation
across folds permits a small nonzero, here negative, pooled Spearman value.

## Supporting retrospective comparisons

These comparisons use historical model outputs. They support the direction of
the primary result but are not additional pristine tests of models selected in
this repository.

| Evaluation | Model | n | MAE (95% CI) | Spearman (95% CI) | Status |
|---|---|---:|---:|---:|---|
| Historical lockbox | Fine-tuned Nesso heads | 714 | 0.601 (0.561, 0.644) | 0.602 (0.543, 0.656) | Opened lockbox |
| Historical lockbox | PXR-only 2D ensemble | 714 | **0.478 (0.446, 0.514)** | **0.741 (0.693, 0.780)** | Opened lockbox |
| Public challenge | Fine-tuned Nesso heads | 513 | 0.601 (0.548, 0.655) | 0.657 (0.596, 0.713) | Retrospective; provisional provenance |
| Public challenge | ChEMBL-augmented 2D submission | 513 | **0.516 (0.470, 0.566)** | **0.748 (0.693, 0.792)** | Retrospective public truth |

The label “2D” does not denote one fitted estimator across all three settings:

- **Development:** a newly nested, fold-matched LightGBM model.
- **Lockbox:** a pre-existing PXR-only, Emax-qualified five-model LightGBM
  ensemble.
- **Challenge:** a pre-existing ChEMBL-augmented, globally offset LightGBM
  submission.

Likewise, the lockbox and challenge Nesso rows are frozen historical prediction
artifacts, not outputs regenerated during the analysis-only command below.

## The matched 2D model

The primary 2D comparator is a deterministic LightGBM regressor over RDKit and
molfeat physicochemical descriptors plus hashed atom-pair, Morgan/ECFP,
layered, pattern, Avalon, ErG, pharmacophore, topological, CATS2D,
scaffold-key, EState, MACCS, and functional-group features.

A legacy outcome-blind variance/correlation filter reduced 32,491 numeric
features to the packaged 14,325-feature table. Within every outer fold, four
inner LightGBM fits rank those inputs by mean gain and select a fold-specific
top 1,000; a second inner pass chooses the boosting duration before the outer
refit. Training weights combine inverse pEC50 standard error with curation
weight. The fixed LightGBM configuration and every selected feature are
recorded under [`reports/model_comparison/`](reports/model_comparison/).

The 14,325-feature prefilter was created using development structures before
this comparison. It used neither pEC50 nor lockbox compounds, but it was not
recomputed inside each outer fold. Supervised selection and stopping are fully
nested; the legacy unsupervised prefilter is fixed and mildly transductive.

## What the compound-level errors say

The models do not appear to be succeeding on wholly different compound
populations. Across the three evaluations, signed residual correlations are
0.77–0.82 and absolute-error correlations are 0.66–0.72. Nesso nevertheless has
the smaller absolute error for about 39–40% of compounds. An untuned equal
blend does not improve on 2D MAE.

Both learned models compress the active tail. For the 55 nested compounds with
pEC50 above 6, Nesso MAE is 1.349 and 2D MAE is 1.245; Nesso predicts one above
6 and 2D predicts none. The subgroup, charge, similarity, potency-bin, and
activity-cliff analyses are exploratory rather than confirmatory.

See the [scientific report](reports/model_comparison/SCIENTIFIC_REPORT.md) for
paired differences, calibration, error complementarity, subgroup results, and
the complete limitations.

## Interpretation boundaries

- The supported conclusion is assay-specific: cached-head adaptation transfers
  useful PXR signal into Nesso-1, while conventional 2D features remain
  materially stronger for this endpoint.
- The opened historical lockbox cannot be reused for pristine optimization.
- Challenge labels were public before this analysis, so challenge performance
  is retrospective external-dataset evidence rather than a temporal blind test.
- The cached challenge extraction has strong internal parity, but its recorded
  run status fails a stricter comparison with a separate reference extraction.
  That provenance discrepancy must be resolved before publication.
- Cluster-bootstrap intervals address dependence within defined chemical
  families, not dataset choice, systematic assay error, or uncertainty in the
  fixed 2D prefilter.
- This package reproduces a scientific comparison. It does not yet distribute a
  deployable adapted Nesso checkpoint.

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

The CPU-only test container uses a public Python image:

```bash
docker build -t nesso-pxr-finetune .
docker run --rm nesso-pxr-finetune
```

## Reproduce the comparison

### 1. Verify packaged inputs

This test checks every file listed in `data/published/SHA256SUMS.json`:

```bash
pytest tests/test_portability.py::test_published_input_manifest_matches_files
```

### 2. Regenerate the reported statistics

The fastest exact route starts from committed out-of-fold and historical
predictions. It is CPU-only; the principal cost is 2,000 bootstrap replicates.

```bash
python scripts/run_publication_comparison.py \
  --stage analyze \
  --output-dir /tmp/nesso-pxr-analysis
```

Successful reproduction should leave these comparisons silent:

```bash
diff -u reports/model_comparison/metrics.csv \
  /tmp/nesso-pxr-analysis/metrics.csv
diff -u reports/model_comparison/paired_cluster_bootstrap_differences.csv \
  /tmp/nesso-pxr-analysis/paired_cluster_bootstrap_differences.csv
```

### 3. Refit the matched models from packaged features

Refit the 2D model from the packaged 14,325-feature table:

```bash
python scripts/run_publication_comparison.py \
  --stage 2d \
  --output-dir /tmp/nesso-pxr-2d
```

Refit the Nesso regression heads from packaged representations. The exact 165
MB Nesso-1 v1.0.0 checkpoint is pinned and checksum-verified by the downloader;
the nested refit requires a CUDA device.

```bash
python scripts/download_nesso_checkpoint.py
python scripts/run_publication_comparison.py \
  --stage nesso \
  --output-dir /tmp/nesso-pxr-nesso \
  --device cuda
```

The combined comparison rerun performs both nested refits and then regenerates
the analysis. It still starts from packaged Nesso representations, the reduced
2D feature table, and historical external predictions.

```bash
python scripts/download_nesso_checkpoint.py
python scripts/run_publication_comparison.py \
  --stage all \
  --output-dir /tmp/nesso-pxr-full \
  --device cuda
```

The repository does not currently regenerate the Nesso representations from
molecular structures, reconstruct the original 32,491-feature 2D table, or
recreate historical lockbox/challenge predictions. Those are the boundary of
the packaged reproduction claim. All runner defaults resolve from the
repository root and remain overridable for sensitivity analyses.

## Rebuild the leakage audit

```bash
python -m nesso_pxr.audit \
  --config configs/protocol.yaml \
  --output /tmp/nesso-pxr-milestone0
```

The audit checks compound, canonical-SMILES, Bemis-Murcko-scaffold, and
Butina-component separation across development and historical lockbox roles.

## Repository map

| Path | Purpose |
|---|---|
| [`reports/model_comparison/SCIENTIFIC_REPORT.md`](reports/model_comparison/SCIENTIFIC_REPORT.md) | Full methods, results, uncertainty, and limitations |
| [`reports/model_comparison/`](reports/model_comparison/) | Reported tables and compound-level aligned predictions |
| [`scripts/run_publication_comparison.py`](scripts/run_publication_comparison.py) | Current analysis and nested-refit entry point |
| [`scripts/download_nesso_checkpoint.py`](scripts/download_nesso_checkpoint.py) | Pinned upstream checkpoint downloader |
| [`src/nesso_pxr/`](src/nesso_pxr/) | Audit, splitting, cached-head training, and comparison code |
| [`data/`](data/) | Public challenge data, frozen derived tables, and exact comparison inputs |
| [`site/`](site/) | Static AetherArk project pages |
| [`nesso1_pxr_finetuning_poc_spec.md`](nesso1_pxr_finetuning_poc_spec.md) | Historical protocol, including completed and unexecuted branches |

The scripts `run_cached_screen.py`, `finalize_cached_training.py`, and
`predict_challenge.py` preserve earlier experiment-specific workflows. New
reproduction work should start with `run_publication_comparison.py`.

## Public sources and licensing

- [OpenADMET PXR challenge](https://huggingface.co/spaces/openadmet/pxr-challenge)
- [OpenADMET PXR train/test dataset](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
- [Original Nesso-1 code](https://github.com/recursionpharma/nesso)
- [Released Nesso-1 model](https://huggingface.co/recursionpharma/nesso)

The OpenADMET tables, Nesso-1 code, and Nesso-1 weights are distributed under
Apache-2.0. This repository's `LICENSE` covers its original code and
documentation; upstream assets retain their source licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.
