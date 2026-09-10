# Fine-tuned Nesso-1 versus a 2D PXR model

## Answer to the scientific question

Fine-tuning Nesso-1 clearly improves PXR pEC50 prediction over released,
unmodified Nesso-1. It does not match the established 2D LightGBM model on
aggregate error or rank correlation. This conclusion holds in a new matched
nested development comparison, the historical scaffold lockbox, and the
public challenge set.

The models' errors are related rather than arising from separate easy and hard
compound populations. Signed residual correlations are 0.77--0.82 and
absolute-error correlations are 0.66--0.72. Nesso nevertheless has the lower
absolute error for about 39--40% of individual compounds, so the predictions
are not redundant. An untuned equal-weight blend does not beat 2D on MAE.

## Main results

Confidence intervals are 95% scaffold/Butina-component cluster-bootstrap
intervals from 2,000 paired replicates.

| Evaluation | Model | n | MAE (95% CI) | RAE | Spearman (95% CI) |
|---|---|---:|---:|---:|---:|
| Nested development | training-fold mean | 3,344 | 0.925 (0.874, 0.992) | 1.002 | -0.075 (-0.206, 0.032) |
| Nested development | frozen Nesso-1 | 3,344 | 0.915 (0.848, 1.007) | 0.991 | 0.385 (0.318, 0.453) |
| Nested development | fine-tuned Nesso-1 | 3,344 | 0.633 (0.611, 0.654) | 0.686 | 0.625 (0.573, 0.677) |
| Nested development | nested 2D LightGBM | 3,344 | 0.516 (0.498, 0.532) | 0.559 | 0.730 (0.688, 0.771) |
| Historical lockbox | fine-tuned Nesso-1 | 714 | 0.601 (0.561, 0.644) | 0.688 | 0.602 (0.543, 0.656) |
| Historical lockbox | established 2D LightGBM | 714 | 0.478 (0.446, 0.514) | 0.546 | 0.741 (0.693, 0.780) |
| Public challenge | fine-tuned Nesso-1 | 513 | 0.601 (0.548, 0.655) | 0.790 | 0.657 (0.596, 0.713) |
| Public challenge | established 2D challenge model | 513 | 0.516 (0.470, 0.566) | 0.679 | 0.748 (0.693, 0.792) |

In the matched nested comparison, fine-tuning changes MAE by -0.282 pEC50
(95% CI -0.358 to -0.225) and Spearman by +0.240 (0.211 to 0.272) relative to
frozen Nesso. Relative to nested 2D, fine-tuned Nesso has MAE higher by 0.117
(0.101 to 0.133) and Spearman lower by 0.105 (-0.131 to -0.084). The analogous
Nesso-minus-2D MAE differences are 0.124 (0.090 to 0.156) on the lockbox and
0.085 (0.044 to 0.124) on the public challenge set.

## Matched nested design

The development estimate uses all 3,344 curated dose-response compounds: 3,072
from the original Emax-qualified cohort and 272 additionally curated lower-Emax
records. Both models use the identical five outer folds. The folds keep
Bemis-Murcko scaffold groups and connected Morgan-fingerprint/Butina components
intact. No lockbox outcomes are used for development training, selection, or
stopping.

For each Nesso outer fold, the original bounded E2 grid (four learning rates by
three weight decays) is evaluated across the four remaining chemical-family
folds. The six E3 uncertainty/curation-weight choices are then evaluated using
the inner-selected E2 settings. The minimum pooled inner MAE, with Spearman as
tie-breaker, selects the outer configuration. The median of its four inner best
epochs fixes training duration, and three models (seeds 42, 43, and 44) are
refit on the complete outer-training partition. Selected durations range from
26 to 28 epochs. The final prediction is the three-seed mean.

The training-fold mean is recomputed from each outer-training partition. Frozen
Nesso uses the released score unchanged. The affine control fits an intercept
and slope to frozen Nesso using only each outer-training partition.

## 2D model and training recipe

The 2D comparator is a deterministic LightGBM regressor. Its molecular inputs
are RDKit/molfeat physicochemical descriptors and hashed structural features,
including atom-pair, Morgan/ECFP, layered, pattern, Avalon, ErG,
pharmacophore, topological, CATS2D, scaffold-key, EState, MACCS, and functional
group features. A legacy holdout-blind variance/correlation filter reduces
32,491 numeric inputs to 14,325. Invalid numeric values are replaced by zero.

Feature selection is nested for this comparison. In each outer fold, four
inner LightGBM fits rank all 14,325 inputs by mean gain and select a separate
top 1,000. A second four-fold inner pass chooses the number of boosting rounds;
the median inner best iteration is fixed before refitting all outer-training
rows. The five outer refits use 638--1,040 trees.

The fixed LightGBM settings are regression objective, learning rate 0.03, 63
leaves, minimum 20 child samples, L2 penalty 5, no L1 penalty, full row and
feature sampling, deterministic column-wise training, and early-stopping
patience 100 in inner folds. Training weights are inverse pEC50 standard error
multiplied by the curation weight. These choices reproduce the established 2D
branch as closely as possible while moving supervised feature selection and
stopping inside the outer clusters.

Only 240 features are selected in all five outer folds and 1,987 distinct
features appear at least once, indicating meaningful selection instability.
MolLogP and molecular weight recur near the top alongside VSA/BCUT descriptors
and several structural fingerprint bits.

The historical 714-compound result uses the pre-existing PXR-only,
Emax-qualified five-model LightGBM ensemble. The challenge comparator is the
pre-existing ChEMBL-augmented, globally offset 2D submission; it is therefore
an established challenge model, not the exact matched nested 2D estimator.

## Where performance differs

The 2D model is better in every nested nearest-training-similarity bin and in
all assay-SE quartiles. Its advantage is largest in the highest-SE quartile
(MAE 0.788 versus 1.087) and among negatively charged compounds (0.551 versus
0.872). It is also better in the low-potency and high-potency tails.

Fine-tuned Nesso has a small local advantage for compounds with observed pEC50
in (4, 5] in nested development (MAE 0.331 versus 0.349) and in the challenge
set (0.296 versus 0.377). On the 272 lower-Emax curated development records the
models are essentially tied on MAE (Nesso 0.533, 2D 0.531), although 2D retains
higher Spearman (0.426 versus 0.354). These are descriptive, outcome-defined
subgroups rather than confirmatory tests.

The two models' compound-level error comparison is:

| Evaluation | signed residual r | absolute-error r | Nesso wins | worst-decile Jaccard | oracle MAE | equal-blend MAE |
|---|---:|---:|---:|---:|---:|---:|
| Nested development | 0.779 | 0.665 | 40.1% | 0.393 | 0.404 | 0.541 |
| Historical lockbox | 0.769 | 0.666 | 38.5% | 0.440 | 0.377 | 0.502 |
| Public challenge | 0.817 | 0.723 | 39.6% | 0.529 | 0.404 | 0.528 |

The oracle column demonstrates theoretical complementarity but is not an
achievable prospective model. The equal blend slightly improves challenge
Spearman to 0.771 but worsens MAE relative to 2D, so it should not be promoted
without a new, fully nested ensemble study.

## Active-tail and activity-cliff stress tests

Both learned models compress the active tail. For the 55 nested compounds with
pEC50 > 6, Nesso MAE is 1.349 and 2D MAE is 1.245; Nesso predicts one above 6
and 2D predicts none. Neither predicts any of the eight lockbox or 31 challenge
actives above 6. Fine-tuning improves aggregate performance while making the
active-tail MAE worse than frozen Nesso on the lockbox and challenge sets.

For held-out points whose nearest training neighbor has Morgan similarity at
least 0.5 and differs by at least 1.5 pEC50, nested MAE is 0.770 for Nesso and
0.736 for 2D (n=50). Nesso is nominally better on the analogous lockbox subset
(0.690 versus 0.752, n=13), while 2D is better on the challenge subset (0.854
versus 0.947, n=103). Only one precomputed default cliff pair has both endpoints
in the same nested outer fold, and none has both endpoints in the lockbox, so
pair-direction results are too sparse for a publication claim.

## Calibration

On nested development, the calibration equation `observed = intercept + slope
* prediction` gives Nesso intercept -0.391 and slope 1.067, versus 2D intercept
-0.076 and slope 1.011. On the historical lockbox the corresponding values are
-0.386/1.080 and -0.226/1.051. The public challenge Nesso slope is 1.411,
indicating substantially greater range compression than 2D (slope 1.055).

## Interpretation and limitations

The supported claim is narrow: cached-head fine-tuning transfers useful PXR
signal into Nesso-1, but a conventional 2D model remains materially stronger
for this endpoint. The result does not establish a structure-aware advantage,
nor does it support use in the highly active tail.

- The 2D variance/correlation prefilter was created before this comparison from
  the 3,072 Emax-qualified development structures. It did not use pEC50 and did
  not use lockbox rows, but it was not recomputed inside each new outer fold.
  Thus the supervised selection and stopping estimate is nested, while the
  legacy unsupervised representation filter is fixed/transductive. A strict
  inductive sensitivity analysis would refit this prefilter within every outer
  training partition.
- The 714-compound lockbox predictions were generated historically and are not
  used to choose anything here, but the lockbox has already been opened. It
  cannot support further iterative optimization as a pristine holdout.
- The 513 challenge labels were public before this retrospective analysis. The
  challenge results are external-dataset evidence, not a temporal blind test.
- The cached challenge extraction has excellent internal score parity, but its
  recorded run status fails the much stricter comparison with a separate
  reference run. This provenance issue should be resolved before publication.
- Bootstrap intervals resample chemical-family clusters and address dependence
  within those groups. They do not account for dataset choice, assay-systematic
  error, or uncertainty from the fixed legacy 2D preprocessing.
- Potency-bin, charge, similarity, active-tail, and cliff results are
  exploratory and multiplicity is not controlled.

## Reproduction and artifact map

Regenerate the published aggregate analysis from the committed compound-level
prediction tables with:

```bash
python scripts/run_publication_comparison.py \
  --stage analyze \
  --output-dir reproduced/model_comparison
```

This CPU-only route requires no external checkout or model download. To refit
both nested models from the packaged feature inputs, first retrieve the exact
released Nesso checkpoint and then run the full GPU comparison:

```bash
python scripts/download_nesso_checkpoint.py
python scripts/run_publication_comparison.py \
  --stage all \
  --output-dir reproduced/full_comparison \
  --device cuda
```

All defaults resolve from the repository root. The packaged inputs and their
SHA-256 manifest are under `data/published/`.

Key outputs are:

- `metrics.csv`: aggregate and Emax-cohort metrics, including RAE and
  calibration.
- `cluster_bootstrap_intervals.csv` and
  `paired_cluster_bootstrap_differences.csv`: model intervals and paired
  differences.
- `nested_nesso_inner_screen.csv` and
  `nested_nesso_outer_selections.csv`: all inner choices and fixed refit epochs.
- `nested_2d_method.json`, `nested_2d_outer_summary.csv`, and
  `nested_2d_selected_features.csv`: complete 2D recipe and fold-local choices.
- `error_complementarity.csv`, `stratified_metrics.csv`,
  `active_tail_metrics.csv`, and `activity_cliff_metrics.csv`: compound-specific
  stress analyses.
- The three `*_aligned_predictions.csv` files: exact compound-level observed
  values, predictions, split annotations, and nearest-training similarity.
