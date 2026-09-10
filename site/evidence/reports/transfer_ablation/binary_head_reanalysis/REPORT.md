# Nesso-1 binary-head correction

## Correction

The original zero-shot comparison evaluated only Nesso-1's continuous affinity
output. That was an incomplete assessment of the released model. Nesso-1 also
produces a binder probability through a separate pretrained binary branch.

The continuous conversion itself was correct:

`pIC50-equivalent = 6 - log10(IC50 / micromolar)`.

It was not correct to label this value as a direct cellular PXR pEC50 prediction,
or to infer from its finite value that Nesso classified every compound as a
binder. The corrected analysis evaluates the continuous and binary outputs
separately and together. Nesso's [official prediction documentation](https://github.com/recursionpharma/nesso/blob/main/docs/prediction.md)
defines both outputs; the [model card](https://huggingface.co/recursionpharma/nesso)
describes Nesso-1 as a binding-affinity model rather than a functional-activity
model.

## Reconstruction

Each Nesso ensemble member maps its cached 384-dimensional `affinity_repr`
through two separate nonlinear branches:

- a continuous 384–384–1 affinity MLP;
- a 384–384–1 binary-score MLP followed by a learned 1–1 logit layer.

The released ensemble binder probability is the mean of the two member sigmoid
probabilities. The reported ensemble logit is the logit of that mean.

The reconstruction used checkpoint SHA-256
`9928a8a824d147d665e76656804af1cd91c86731516d0b561f9fd1c91ee45622`.
Replaying the continuous branch from the same cached representations matched the
stored output within `4.77e-7` across the labeled, challenge, and pair-pooling
artifacts. This establishes exact current-inference provenance for applying the
binary branch to those representations.

Retained historical JSON files are not the parity reference. They form a partial
earlier run: 2,755 JSON files exist, but only 361 record IDs intersect the current
4,134-row labeled set. On those 361 records, current versus historical continuous
scores differed by mean absolute 0.0293 and binary probabilities by 0.0133. Both
differences move together and are consistent with different run/input provenance.

## Cohorts and leakage controls

The fixed evaluation cohorts were retained without modification:

| Cohort | All rows | Emax-qualified | Lower-Emax supporting cohort |
|---|---:|---:|---:|
| Development | 3,344 | 3,072 | 272 |
| Validation | 790 | 714 | 76 |
| Public challenge | 513 | — | — |

All supervised development predictions are outer-fold predictions from the
existing chemical-family folds. Ridge strength was selected only within each
outer training partition. Full-development fits selected settings from grouped
development folds before evaluation on validation or challenge compounds. The
validation labels did not select models or hyperparameters.

Classification-like analyses used three thresholds fixed before inspection:
pEC50 >= 4, >= 5, and >= 6. These are **functional-potency proxies**, not physical
binding labels. A below-threshold compound is only "not potent at this threshold."
Chemical-family bootstrap intervals used development connected components and
validation scaffold groups. Challenge intervals resampled compounds because no
matching family assignment was established for that cohort.

## Corrected activity prediction results

| Readout or model | Development MAE | Development Spearman | Validation MAE | Validation Spearman |
|---|---:|---:|---:|---:|
| Zero-shot continuous affinity, pIC50-equivalent | 0.915 | 0.385 | 0.815 | 0.312 |
| Continuous affinity, nested affine calibration | 0.839 | 0.383 | 0.811 | 0.312 |
| Binder logit, nested linear calibration | 0.856 | 0.335 | 0.821 | 0.301 |
| Continuous + binary, nested linear calibration | 0.832 | 0.396 | 0.803 | 0.331 |
| Continuous + binary, regularized quadratic calibration | 0.774 | 0.447 | 0.753 | 0.366 |
| Frozen mean 384D representation + ridge | 0.633 | 0.654 | 0.596 | 0.619 |
| Fine-tuned continuous affinity head | 0.633 | 0.625 | 0.597 | 0.581 |
| Fine-tuned dual-branch activity adapter | 0.629 | 0.632 | 0.595 | 0.591 |
| Ligand-only LightGBM | 0.516 | 0.730 | 0.489 | 0.697 |

Adding the binary logit to a linear calibration reduced development MAE by 0.007
(family-bootstrap 95% CI -0.012 to -0.003). Its validation change was similar in
size but unresolved (CI -0.017 to 0.002). The quadratic two-readout calibration
made a larger, resolved improvement over affinity-only calibration: -0.066 MAE on
development and -0.058 on validation. It still remained far behind the 384D probe
and LightGBM.

The bounded dual-branch activity adapter initialized exactly at the released
continuous output, added a zero-initialized bounded contribution from the binary
pathway, and fine-tuned both pretrained readout branches with an L2-SP anchor.
The 384D representations stayed frozen. It was trained only on functional pEC50;
it is not a fine-tuned binder classifier. Relative to the continuous-head
fine-tune, it improved development MAE by 0.004 (CI -0.008 to -0.0003) and
validation MAE by 0.002 (CI -0.0048 to 0.0005). Validation Spearman increased by
0.010 (CI 0.0067 to 0.0139). On the public challenge cohort, however, its MAE was
0.647 versus 0.601 for the continuous-head fine-tune. The gain is small and not
robust across cohorts.

LightGBM remained better than the corrected dual-branch adapter by 0.113 MAE on
development (CI 0.098 to 0.128) and 0.106 on validation (CI 0.076 to 0.137).

## Functional-potency proxy analysis

| Cohort | Proxy threshold | Positive prevalence | ROC-AUC (95% CI) | Average precision (95% CI) |
|---|---:|---:|---:|---:|
| Development | pEC50 >= 4 | 69.0% | 0.700 (0.670–0.728) | 0.784 (0.743–0.821) |
| Development | pEC50 >= 5 | 31.4% | 0.656 (0.633–0.678) | 0.405 (0.369–0.441) |
| Development | pEC50 >= 6 | 1.7% | 0.750 (0.687–0.801) | 0.056 (0.027–0.104) |
| Validation | pEC50 >= 4 | 73.7% | 0.696 (0.645–0.742) | 0.830 (0.792–0.868) |
| Validation | pEC50 >= 5 | 35.6% | 0.627 (0.587–0.666) | 0.437 (0.385–0.497) |
| Validation | pEC50 >= 6 | 1.1% | 0.697 (0.472–0.892) | 0.038 (0.012–0.121) |

Binder probability contains reproducible, moderate information about functional
potency. It is not random. It is also not a useful activity classifier at the
released binder cutoff of 0.5:

- Development: 2,105 of 2,308 pEC50 >= 4 compounds had binder probability below
  0.5; 44 of 56 pEC50 >= 6 compounds were below 0.5.
- Validation: 535 of 582 pEC50 >= 4 compounds had binder probability below 0.5;
  6 of 9 pEC50 >= 6 compounds were below 0.5.

These are disagreements with an activation-potency proxy, not binding false
negatives. Activation can require binding, but pEC50 also reflects efficacy,
cellular exposure, receptor state, assay context, and possible indirect or
interference mechanisms. Conversely, a compound can bind without activating the
reporter.

Binder probability increased with activity, but remained low and overlapping.
Median probabilities across development activity bins were 0.13 (<3), 0.15
(3–4), 0.21 (4–5), 0.25 (5–6), and 0.35 (>=6). The same graded but compressed
pattern appeared in validation and challenge cohorts.

The Emax-qualified cohorts gave similar AUCs to the all-row analysis. Lower-Emax
subsets were small, particularly at pEC50 >= 6, and do not support strong
conclusions. They are reported separately in the machine-readable tables.

## Where the information resides

The frozen 384D representation remained much more informative than either
released scalar readout. Mean-representation ridge achieved 0.633 MAE and 0.654
Spearman on development, compared with 0.774 and 0.447 for the strongest
two-scalar calibration.

Appending the continuous and binary scalars to the 384D ridge probe changed MAE
by -0.00002 on development and -0.000002 on validation; both confidence intervals
included zero. The same null result held when member-specific readouts were
appended to the concatenated 768D representation. This is expected mechanistically:
the two scalar branches are deterministic nonlinear functions of the cached
representations. They expose a useful low-dimensional summary but do not add new
information to a sufficiently regularized representation probe.

A fixed 32-unit MLP probe did not benefit from appended readouts and performed
worse on validation. That result is architecture-specific and exploratory; it is
not evidence that every nonlinear probe would fail.

## Pair-pooling extension

The binary branch was applied exactly to cached all-pairs, ligand-ligand-only, and
receptor-ligand-only late-pooling representations. Continuous replay remained
within `4.77e-7` for every variant.

For the pEC50 >= 4 proxy, ligand-ligand-only pooling increased ROC-AUC relative to
all-pairs pooling by 0.019 on development (CI 0.010 to 0.030) and 0.024 on
validation (CI 0.007 to 0.041). It also changed the released-cutoff binder-positive
fraction from 8.0% to 29.1% on development. The absolute probability is therefore
sensitive to the late-pooling intervention.

After calibrating the continuous and binary scalars together, all-pairs pooling
had lower point MAE than ligand-ligand-only pooling by 0.011 on development and
0.009 on validation, but both paired confidence intervals included zero. This
does not overturn the earlier 384D ridge result favoring ligand-ligand-only
pooling on development. More importantly, this intervention changes only the
final affinity pooling route. Ligand-ligand cells were already protein-conditioned
upstream, so it is not a protein-causality test.

## What changes

| Prior statement | Status after correction | Reason |
|---|---|---|
| Nesso predicts that nearly every molecule binds PXR | Invalid | A finite continuous affinity is not a binder call; only about 7–8% exceed binary probability 0.5. |
| The continuous affinity output is strongly compressed for these PXR inputs | Unchanged | That measurement and its exact replay remain valid. |
| The original zero-shot result measured complete released Nesso-1 performance | Invalid | It measured only the continuous branch. |
| The released readouts transfer poorly to PXR activation potency | Supported, now properly tested | The binary branch adds measurable signal but neither readout nor their combination approaches the 384D probe or LightGBM. |
| The frozen 384D representation contains transferable PXR information | Unchanged | Ridge and nonlinear representation probes remain valid and substantially outperform released scalar readouts. |
| Fine-tuning did not beat LightGBM | Unchanged | The corrected dual-branch adapter remains materially worse on development and validation. |
| Endpoint mismatch is a significant confounder | Plausible, not proven | Binding outputs and activation potency are related but discordant; matched PXR binding measurements are still required to separate endpoint mismatch from target/domain difficulty. |
| Nesso uses protein information for this fixed-PXR prediction | Pending | PXR is constant, and the late-pooling ablation cannot establish protein causality. |

## Recommended dossier wording

> The first analysis evaluated only Nesso-1's continuous affinity branch. A
> correction reconstructed and evaluated the separate binder branch from the
> same cached representations and released checkpoint. Binder probability carried
> moderate information about PXR activation potency, and combining both pretrained
> readouts improved low-dimensional calibration. The improvement was not enough to
> approach a frozen 384D representation probe or ligand-only LightGBM. A bounded
> dual-branch activity fine-tune produced only a small, cohort-dependent gain.
> These results show that the omitted branch mattered scientifically, but did not
> reverse the model comparison. Because the labels measure cellular activation
> rather than physical binding, the study cannot distinguish endpoint mismatch
> from PXR-specific structural difficulty without matched binding data.

The old label "Frozen Nesso pEC50" should be replaced with "Zero-shot continuous
affinity, pIC50-equivalent." Binder probability should be shown separately. The
ROC, precision-recall, and contingency figures must retain the phrase
"functional-potency proxy; not binding ground truth."

## Reproduction

The analysis used at most four CPU threads and no GPU:

```bash
docker run --rm --cpus=4 --memory=8g \
  --env CUDA_VISIBLE_DEVICES='' --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 --entrypoint python \
  -v "$PWD:/work" -w /work ${NESSO_IMAGE:-nesso:1.0.0} \
  scripts/reconstruct_nesso_binary_head.py

docker run --rm --cpus=4 --memory=8g \
  --env CUDA_VISIBLE_DEVICES='' --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 --entrypoint python \
  -v "$PWD:/work" -w /work ${NESSO_IMAGE:-nesso:1.0.0} \
  scripts/run_dual_branch_activity_adapter.py

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 PYTHONPATH=src \
python scripts/run_binary_head_reanalysis.py --bootstrap-replicates 2000
```

Primary artifacts are `reconstruction_provenance.json`,
`regression_metrics.csv`, `proxy_classification_metrics.csv`, the family-bootstrap
tables, pair-pooling tables, prediction files, and four bounded PNG figures in
this directory. No site or external publication state was changed.
