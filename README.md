# Nesso-1 adaptation: PXR prediction, molecular weight and assay uncertainty

Last edited: 2026-09-08.

Start with the [main experimental page](site/index.html). It follows the study from its original questions through each completed sub-experiment, with separate methods, results and source evidence. The historical initial comparison below retains its original cohort and scope.

## What the study now establishes

1. **Adaptation improves on native Nesso, but more training capacity is not automatically better.** Cached-head tuning and selected upstream affinity-module LoRA transfer useful PXR signal. At N500, frozen-vector ridge and descriptor LightGBM outperform the tested neural recipes. Main-trunk and external ESM training were not performed.
2. **A simple MW control recapitulates the conspicuous small-molecule correction.** Among the 237 compounds below 215 Da, N500 weighted MAE is 1.834 for unchanged Nesso, 1.179 for heads + LoRA, 1.163 for a fitted native-score + MW mapping and 0.863 with their interaction. MW alone is not the interaction model, and this subgroup is not synonymous with most of the aggregate gain.
3. **Frozen Nesso features add predictive information beyond descriptors.** A training-selected blend improves weighted MAE in all five reused development folds: 0.5006 to 0.4868, with pooled Spearman 0.6569 to 0.6864. Its improvement comes mainly from pEC50 ≥4 compounds; lower-potency error worsens. The agreed inverse-assay-SE score already downweights those less precise measurements. This is a locked-procedure retrospective result, not prospective superiority.
4. **The CYP extension is mixed, not universal replication.** Adding MW helps fitted Nesso scalar predictions for CYP2C9 and CYP3A4; an interaction gives no consistent further benefit. CYP1A2 small-molecule bias has the opposite direction. These are direct-inhibition scalar mappings, not CYP neural fine-tuning. CYP scores are unweighted because a justified sampling-SE conversion was unavailable.
5. **The efficacy diagnostic changes the biological interpretation.** Small PXR molecules have much higher pEC50 SE (median 0.558 versus 0.141), but also higher positive-control-normalized Emax (1.240 versus 0.977). Baseline-relative Emax is lower. All 237 small molecules pass the historical normalized-Emax qualification. The data support an uncertainty association, not the proposed proof of binding without activation; missing curve traces and normalization details prevent a causal explanation.

## Experimental progression

The original aim was to improve PXR activation prediction and test whether a protein-conditioned pretrained representation could beat or complement a ligand-only model. Head/probe and optimization controls led to a systematic low-data study, stricter chemistry controls and genuine affinity-module LoRA. The observed small-molecule correction motivated MW controls. A separate CPU hybrid then established complementary prediction, followed by a four-CYP scalar transfer test and a direct check of the PXR efficacy/uncertainty explanation.

The project is a completed retrospective case study, not an exhaustive fine-tuning benchmark. Repeated acquired-subset confirmation, a fresh assay-compatible evaluation, raw-curve investigation and broader upstream optimization remain distinct unexecuted follow-ups—not conditions silently counted as completed.

### Canonical result records

- [Architecture and exact fitted layers](reports/architecture/index.html)
- [Original transfer, learning-rate, loss and probe audit](reports/transfer_ablation/README.md)
- [Continuous/binary readout correction](reports/transfer_ablation/binary_head_reanalysis/REPORT.md)
- [Completed final pair-pooling ablation](reports/transfer_ablation/pair_pool_full/REPORT.md)
- [Low-data design](experiments/20260905_low_data_adaptation/README.md) and [completed follow-up](experiments/20260906_low_data_followup/RESULTS.md)
- [Upstream-LoRA predictive comparison](reports/architecture/results/RESULTS.md)
- [MW controls and existing detailed HTML](reports/architecture/results/mw-controls/index.html)
- [Hybrid pilot](experiments/20260908_descriptor_nesso_blend/README.md) and [locked all-fold comparison](experiments/20260908_descriptor_nesso_blend_allfolds/README.md)
- [PXR Emax and curve-uncertainty diagnostic](experiments/20260908_PXR-MW-efficacy-diagnostic/README.md)

The main experimental website packages the CYP companion and curated evidence inside its served tree. Original scientific predictions and experiment reports are preserved. Numerical conversion of native affinity does not equate binding with activation, and a favorable Nesso score does not establish measured binding.

## Historical initial comparison

PXR is a particularly challenging target in medicinal chemistry.  It's ability to bind many highly dissimilar chemotypes and the presence of severe activity cliffs make it particularly difficult to predict the activity of novel compounds using general learned methods.  [Nesso-1](https://github.com/recursionpharma/nesso) is a distogram and binding-affinity prediction model developed by Recursion Pharmaceuticals; it does not generate cofolded coordinates.

This study asks three questions: 

1. **does fine-tuning Nesso-1 improve its performance in human PXR activity prediction**
2. **does it compete with a simple 2D molecular model?** (RDKit descriptors and LightGBM)
3. **does Nesso-1 contain orthogonal information to a 2D molecular model?**

The answer is **yes, no, and not much**. 

Updating Nesso-1's final regression heads from cached molecular representations substantially improves on frozen Nesso-1, but
does not beat a 2D LightGBM model. Error correlation between the simple 2D model and finetuned Nesso-1 is high (~0.7-0.8 correlation). 
The study is based on data generated for the [OpenADMET PXR challenge](https://huggingface.co/spaces/openadmet/pxr-challenge).

> **Follow-up audit (2026-08-12):** Expanded nested LR selection, frozen
> linear and shallow probes, all-790 lockbox stratification, target/loss
> controls, learned stacking, and a protein-pair ablation protocol are reported
> in [the transfer and validation audit](reports/transfer_ablation/README.md).
> The strongest updated conclusion is that transferable PXR signal is largely
> linearly accessible in the frozen representation; PXR-specific protein use
> remains an unproven causal claim.

## A bit deeper:

This work addresses the **activity-prediction track** of the OpenADMET PXR
challenge, not the structure-prediction track. The measured endpoint is
functional human PXR activation reported as pEC50 in a cellular assay.

```text
4,139 raw PXR dose-response records
└── 4,134 curated and weighted compounds
    ├── 3,344 development compounds → five nested chemical-family folds
    └──   790 historical scaffold-lockbox compounds
          └── 714 Emax-qualified compounds shared by the historical models

513 public challenge compounds → separate external dataset
```

Development folds keep components formed by unioning Bemis-Murcko scaffold
groups with Morgan-fingerprint Butina clusters intact. The 714-compound
lockbox comparison is the Emax-qualified intersection for which both historical Nesso and 2D
predictions are available; the remaining 76 lockbox records have lower Emax.

## Initial cached-head fine-tuning approach

I performed cached head-only fine-tuning rather than on all layers. This was to test if Nesso-1's pretrained features contained transferable signal for PXR, and also reducing the computational cost and overfitting risk of retraining the entire model (mostly computational cost).  

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
- Nesso-1 feature extraction training time for the 4,647 training and test compounds was ~ 5 hours and training was approximately two minutes on a single RTX5070 Ti.

Nesso's predicted affinity value is IC50-like and uses a different numerical
orientation from pEC50. Training and prediction use:

```text
Nesso-scale training target = 6 - observed pEC50
predicted pEC50            = 6 - affinity_pred_value
```

This transform aligns numbers; it does **not** make the endpoints biologically
equivalent. Cellular functional PXR EC50 also reflects efficacy, permeability,
receptor context, and assay behavior. 

> **Private model checkpoint:** The tested three-seed all-label head ensemble
> is available at
> [`dargason/nesso-1-pxr-heads` revision `d163260`](https://huggingface.co/dargason/nesso-1-pxr-heads/tree/d163260696bfb6c35b429c3d690f3623681f19dd).
> This is a head-only release that requires compatible cached Nesso-1
> representations; Hugging Face sign-in and repository access are required
> while it remains private.

## Primary results (questions 1 and 2)

All four development estimates use the same 3,344 compounds and identical five
outer chemical-family folds. Confidence intervals are 95% intervals from 2,000
paired scaffold/Butina-component cluster-bootstrap replicates.

| Model | MAE (95% CI) | RAE | Spearman (95% CI) |
|---|---:|---:|---:|
| Outer-training-fold mean | 0.925 (0.874, 0.992) | 1.002 | -0.075 (-0.206, 0.032) |
| Frozen Nesso-1 | 0.915 (0.848, 1.007) | 0.991 | 0.385 (0.318, 0.453) |
| Fine-tuned Nesso-1 regression heads | 0.633 (0.611, 0.654) | 0.686 | 0.625 (0.573, 0.677) |
| Matched nested 2D LightGBM | **0.516 (0.498, 0.532)** | **0.559** | **0.730 (0.688, 0.771)** |

Relative to frozen Nesso, fine-tuning: 
* reduces MAE by 0.282 pEC50 units (95% CI 0.225–0.358)
* increases Spearman by 0.240 (0.211–0.272).

Relative to LightGBM, the fine-tuned Nesso-1 has:
* higher MAE by 0.117 (0.101–0.133)
* lower spearman lower by 0.105 (0.084–0.131)


## Baseline model - PK and Morgan fingerprint + LightGBM

The primary 2D comparator is a deterministic LightGBM regressor over RDKit and
molfeat physicochemical descriptors plus hashed atom-pair, Morgan/ECFP,
layered, pattern, Avalon, ErG, pharmacophore, topological, CATS2D,
scaffold-key, EState, MACCS, and functional-group features.

A variance/correlation filter reduced 32,491 numeric
features to the final 14,325-feature table. Within every outer fold, four
inner LightGBM fits rank those inputs by mean gain and select a fold-specific
top 1,000; a second inner pass chooses the boosting duration before the outer
refit. Training weights combine inverse pEC50 standard error with curation
weight. The fixed LightGBM configuration and every selected feature are
recorded under [`reports/model_comparison/`](reports/model_comparison/).

## Question 3 - does Nesso-1 contain orthogonal information? 

Mostly no.  The models do not appear to be succeeding on wholly different compound
populations. Across the three evaluations, signed residual correlations are
0.77–0.82 and absolute-error correlations are 0.66–0.72. Nesso nevertheless has
the smaller absolute error for about 39–40% of compounds. An untuned equal
blend does not improve on 2D MAE.

Both learned models compress the highly active molecules (pEC50 > 6.0). For the 55 nested compounds with
pEC50 above 6, Nesso MAE is 1.349 and 2D MAE is 1.245; Nesso predicts one above
6 and 2D predicts none.

See the [scientific report](reports/model_comparison/SCIENTIFIC_REPORT.md) for
paired differences, calibration, error complementarity, subgroup results, and
the complete limitations.

## Notes

- This conclusion is specific to this particular pEC50 assay: cached-head adaptation transfers
  useful PXR signal into Nesso-1, while conventional 2D features remain
  materially stronger for this endpoint.
- This analysis was performed after the OpenADMET challenge data were released. Thus it is
  a retrospective evaluation, not a true blind result.
- OpenAI Codex was used for most coding and method implementation and some of the writing of this report, especially text below this line.   

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
| [`site/`](site/) | Exact staged public release; [`release-manifest.json`](site/release-manifest.json) binds its file inventory and hashes |
| [`nesso1_pxr_finetuning_poc_spec.md`](nesso1_pxr_finetuning_poc_spec.md) | Historical protocol, including completed and unexecuted branches |

Raw task outputs, downloaded checkpoints, cloud-transfer bundles, local agent
state, and superseded publication workspaces are excluded by `.gitignore` and
`.dockerignore`. The public release retains only the curated evidence named in
its manifests. The eight directories under `experiments/` are the experiment
sources represented by the published site; new directories must be added to the
site manifest before publication.

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
