---
license: cc-by-4.0
language:
- en
tags:
- chemistry
- drug-discovery
- ADMET
- molecular-properties
- blind-challenge
- computational-chemistry

pretty_name: OpenADMET PXR Induction Blind Challenge
size_categories:
- 10K<n<100K
task_categories:
- tabular-regression
annotations_creators:
- expert-generated
source_datasets:
- original
configs:
- config_name: default
  data_files:
  - split: train
    path: pxr-challenge_TRAIN.csv
  - split: test
    path: pxr-challenge_TEST_BLINDED.csv
- config_name: counter_assay
  data_files:
  - split: train
    path: pxr-challenge_counter-assay_TRAIN.csv
- config_name: structure
  data_files:
  - split: test
    path: pxr-challenge_structure_TEST_BLINDED.csv
- config_name: single_concentration
  data_files:
  - split: train
    path: pxr-challenge_single_concentration_TRAIN.csv
- config_name: crudes_htchem
  data_files:
  - split: train
    path: pxr-challenge_htchem-libraries_TRAIN.csv
- config_name: semi_pure_htchem
  data_files:
  - split: train
    path: pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv
- config_name: phase_1_unblinded
  data_files:
  - split: test
    path: pxr-challenge_TEST_PHASE_1_UNBLINDED.csv
- config_name: phase_2_unblinded
  data_files:
  - split: test
    path: pxr-challenge_TEST_PHASE_2_UNBLINDED.csv
---

# PXR Challenge Train/Test Dataset

A high-quality experimental dataset for predicting human Pregnane-X Receptor (PXR) induction, comprising over 11,000 compounds screened using a high-fidelity in-house assay. This is the largest publicly available PXR activity dataset, released as part of the [OpenADMET PXR Induction Blind Challenge](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/).

**Blog post:** [Announcing the Next OpenADMET Blind Challenge: Predicting PXR Induction](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)

**Challenge Space:** [openadmet/pxr-challenge](https://huggingface.co/spaces/openadmet/pxr-challenge)

**Challenge period:** April 1 – July 1, 2026

**Produced by:** OpenADMET

## CHANGELOG

* Updated 2026-09-02 Corrected SMILES for 10 compounds in `pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv` (OCNT-2469084, -2469095, -2469107, -2469108 through -2469114). Regiochemical enumeration errors; all corrections are constitutional isomers (ΔMW = 0) so no activity or yield values changed. Re-pull if you have trained on a previous version of `semi_pure_htchem`.
* Updated 2026-07-03 Added phase 2 unblinded labels for the default test set subset in `phase_2_unblinded` config.
* Updated 2026-05-27 Added phase 1 unblinded labels for the default test set subset in `phase_1_unblinded` config.
* Updated 2026-05-27 Additional crude data added, see [here](https://openadmet.github.io/octant-pxr-htchem-blogpost/) for more detail. 
* Updated 2026-04-09 dropping some compounds, fixing minor confidence interval issues and improving naming join. See [here](https://docs.google.com/document/d/14-2EL4Zk8g3NNO7bi33gA3fHz_ef98bBigunJMH3Ui4/edit?usp=sharing) for more details.  

### Dataset contents

| Config | Split | Description |
|---|---|---|
| `default` | `train` | Primary assay training set (pEC50, Emax) |
| `default` | `test` | 513-compound blinded test set |
| `counter_assay` | `train` | PXR-null counter-assay training data |
| `structure` | `test` | 184 molecules with X-ray crystal structures |
| `single_concentration` | `train` | Single-concentration screening data (log2 fold change) |
| `crudes_htchem` | `train` | Direct to biology assays of crudes (see our blog post [here](https://openadmet.github.io/octant-pxr-htchem-blogpost/)) |
| `phase_1_unblinded` | `test` | Phase 1 unblinded subset of the primary blinded test set with assay labels |
| `phase_2_unblinded` | `test` | Phase 2 unblinded subset of the primary blinded test set with assay labels |

## Loading with Hugging Face `datasets`

```python
from datasets import load_dataset

# Default config (primary assay)
ds = load_dataset("openadmet/pxr-challenge-train-test")
train = ds["train"]
test  = ds["test"]

# Counter-assay config
ds_counter = load_dataset("openadmet/pxr-challenge-train-test", "counter_assay")
train_counter = ds_counter["train"]

# Structure config
ds_structure = load_dataset("openadmet/pxr-challenge-train-test", "structure")
test_structure = ds_structure["test"]

# Single-concentration config
ds_single = load_dataset("openadmet/pxr-challenge-train-test", "single_concentration")
train_single = ds_single["train"]

# crudes htchem config
ds_crudes = load_dataset("openadmet/pxr-challenge-train-test", "crudes_htchem")
train_crudes = ds_crudes["train"]

# semi-pures htchem config
ds_semi_pure = load_dataset("openadmet/pxr-challenge-train-test", "semi_pure_htchem")
train_semi_pure = ds_semi_pure["train"]

# Phase 1 unblinded test labels config
ds_phase1 = load_dataset("openadmet/pxr-challenge-train-test", "phase_1_unblinded")
test_phase1 = ds_phase1["test"]

# Phase 2 unblinded test labels config
ds_phase2 = load_dataset("openadmet/pxr-challenge-train-test", "phase_2_unblinded")
test_phase2 = ds_phase2["test"]

```

## Loading directly with pandas

```python
import pandas as pd

train          = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
test           = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")
train_counter  = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv")
test_structure = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_structure_TEST_BLINDED.csv")
train_single   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv")
train_crudes   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_htchem-libraries_TRAIN.csv")
train_semi_pure = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")
test_phase1    = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
test_phase2    = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
```
