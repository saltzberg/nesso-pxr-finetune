# CPU follow-up experiments

**Date:** 2026-08-12  
**Status:** complete; the 790-compound lockbox results are retrospective
validation because that set had already been examined.

The machine-readable protocol was frozen before any new lockbox scoring in
[analysis_spec.json](analysis_spec.json). No hyperparameter or epoch choice in
these experiments used a lockbox label.

## Headline conclusions

1. **Most transferable PXR signal is linearly accessible.** On the primary
   mean-384D frozen representation, fully nested ridge achieved development
   MAE 0.6328 and Spearman 0.6537. One- and two-hidden-layer probes improved
   point MAE only to 0.6237 and 0.6198, while reducing Spearman to 0.6445 and
   0.6363. Their development MAE gains over ridge were not resolved by the
   predeclared chemical-family bootstrap. The nonlinear probes also had
   appreciable seed variability and occasional extreme extrapolations.
2. **The historical chemical split is robust to the conservative primary
   standardization because the inputs were already unusually clean.** RDKit
   Cleanup followed by charge-retaining FragmentParent changed zero fold
   assignments. Every validation fold still had zero nearest-training
   neighbors at Morgan Tanimoto >=0.60. Nested ridge and LightGBM results were
   therefore numerically identical to the historical split.
3. **The historical LR boundary was worth fixing, but Huber delta 0.5 is not
   the principal cause of potent-tail compression.** The exact pretrained twin
   head was tested with five LRs and six objectives in nested CV. No selected
   objective predicted any development compound above pEC50 6. MSE was worse;
   MAE and Huber delta 1 gave modest overall/ranking gains, but did not restore
   the high end. Scarcity and regression to the mean remain the stronger
   explanation.

## A. Fully nested frozen-representation capacity ladder

The primary comparison is dimension-matched on the elementwise mean of the two
384D affinity representations. Scaling was fit inside each applicable training
partition. Ridge and each neural architecture had five predeclared tuning
candidates. Neural outer predictions average seeds 42, 43, and 44; fixed refit
duration is the rounded median of training-only inner best epochs.

| Model on mean 384D | Dev MAE | Dev Spearman | Lockbox-790 MAE | Lockbox-790 Spearman |
|---|---:|---:|---:|---:|
| Nested ridge | 0.6328 | **0.6537** | 0.5963 | **0.6191** |
| One-hidden-layer MLP | 0.6237 | 0.6445 | **0.5685** | 0.6129 |
| Two-hidden-layer MLP | **0.6198** | 0.6363 | 0.5759 | 0.6022 |
| Published pretrained twin-head fine-tune | 0.6330 | 0.6252 | 0.5969 | 0.5812 |

Primary paired family-bootstrap differences (first minus second) were:

- one-hidden minus ridge: MAE -0.0091, 95% CI [-0.0182, +0.0001];
  Spearman -0.0092, CI [-0.0187, -0.0008];
- two-hidden minus ridge: MAE -0.0129, CI [-0.0273, +0.0014];
  Spearman -0.0174, CI [-0.0296, -0.0066];
- ridge minus published fine-tune: MAE -0.0002, CI [-0.0141, +0.0151];
  Spearman +0.0285, CI [+0.0181, +0.0402].

The retrospective lockbox favors the one-hidden probe in MAE, but this cannot
be treated as untouched confirmation. The three-seed mean is materially more
stable than individual neural fits. Secondary member-wise results confirm that
member 2 is much stronger than member 1; concatenation adds dimensions but not
a decisive primary benefit. Full predictions, fold selections, seed metrics,
and intervals are in [experiment_a_capacity](experiment_a_capacity/).

Interpretation: the representation already exposes most useful signal to a
linear readout. Extra nonlinear capacity can slightly improve absolute error,
but the rigorous rerun does not reproduce the large exploratory MLP advantage
and trades away ranking performance. These single-representation probes also
differ architecturally from the published twin head, so the comparison is
about accessible readout capacity, not solely pretrained initialization.

## B. Chemical standardization and split stress test

The conservative primary identity was RDKit Cleanup followed by
FragmentParent, retaining charge. ChargeParent, TautomerParent, and the first
InChIKey block were explicit sensitivities rather than silently chosen
identities.

- 4,134/4,134 structures standardized successfully; no supplied raw SMILES was
  multifragment.
- Cleanup changed 9 structures; FragmentParent changed 1 more; ChargeParent
  changed 6; TautomerParent changed 330.
- Raw/canonical, cleaned, fragment-parent, and charge-parent identities had no
  duplicates. TautomerParent collapsed one pair with pEC50 difference 0.005.
- Connectivity identity collapsed four stereochemical pairs. Three differed by
  at least 0.5 pEC50 and one by 1.22, but none crossed historical folds or the
  development/lockbox boundary.
- Formal charge illustrates why identity and model input must be distinguished:
  12 raw-canonical structures and 13 fragment parents were nonzero-charge,
  whereas Nesso's selected prepared states had 819 nonzero-charge rows and 809
  prepared SMILES differed from the identity-canonical SMILES.
- The primary split had 2,931 families, largest family 155, and balanced fold
  sizes 668--669. After label alignment, **0/3,344 rows changed fold**.

Exact validation nearest-training Morgan-Tanimoto summaries were:

| Fold | n | Median | q95 | Maximum | Count >=0.50 | Count >=0.60 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 669 | 0.367 | 0.529 | 0.597 | 63 | 0 |
| 1 | 669 | 0.364 | 0.534 | 0.597 | 73 | 0 |
| 2 | 669 | 0.359 | 0.526 | 0.596 | 55 | 0 |
| 3 | 669 | 0.368 | 0.527 | 0.597 | 64 | 0 |
| 4 | 668 | 0.362 | 0.534 | 0.597 | 63 | 0 |

Because the fold assignment did not change, standardized-split development
metrics exactly reproduce the historical metrics: mean-384D ridge MAE/Spearman
0.6328/0.6537 and LightGBM 0.5160/0.7304. A newly refit retrospective all-790
LightGBM model scored 0.4886/0.6973 versus ridge 0.5963/0.6191. This new 790-row
LightGBM number is not the old 714-row historical prediction and should not be
presented as such. Full form records, collision groups, folds, neighbor rows,
label distributions, predictions, and method metadata are in
[experiment_b_standardization](experiment_b_standardization/).

RDKit emitted tautomer-enumeration/kekulization warnings for some sensitivity
calculations, although it returned all 4,134 outputs. This does not affect the
primary fragment-parent split, but it is another reason not to elevate the
tautomer sensitivity to the primary identity without a dedicated chemistry
review.

## C. Fully nested loss x learning-rate study

This used the released pretrained twin heads exactly: each member is
384->384->384->1 with ReLU, initialized from the pinned checkpoint, trained in
the released 6-pEC50 coordinate with 0.50/0.25/0.25 ensemble/member loss,
AdamW, batch size 64, 30 epochs maximum, patience 5, and gradient clipping 1.0.
Weight decay was fixed at zero to isolate objective and LR. The LR grid was
3e-4, 1e-3, 3e-3, 1e-2, and 3e-2.

| Nested objective (LR selected within each outer training set) | Dev MAE | Dev Spearman | Active-tail MAE | Predicted >6 | Prediction max |
|---|---:|---:|---:|---:|---:|
| MSE | 0.6546 | 0.6267 | 1.4703 | 0 | 5.875 |
| MAE | 0.6293 | **0.6402** | **1.2386** | 0 | 5.628 |
| Huber 0.25 | 0.6329 | 0.6279 | 1.2964 | 0 | **5.922** |
| Huber 0.5 | 0.6349 | 0.6254 | 1.3563 | 0 | 5.555 |
| Huber 1.0 | 0.6273 | 0.6378 | 1.3340 | 0 | 5.654 |
| Huber 2.0 | 0.6345 | 0.6339 | 1.3880 | 0 | 5.712 |
| Nested selector across objective and LR | **0.6224** | 0.6371 | 1.3343 | 0 | 5.555 |
| Published historical fine-tune | 0.6330 | 0.6252 | 1.3486 | 1 | 6.070 |

The nested global selector chose Huber 0.5 at LRs 3e-3, 1e-2, and 1e-3 in
outer folds 0, 2, and 3; MAE at 3e-3 in fold 1; and Huber 1 at 1e-3 in fold 4.
Compared with nested Huber 0.5, its MAE difference was -0.0125, 95% CI
[-0.0216, -0.0038], and Spearman difference +0.0117, CI [+0.0012, +0.0219].
Compared with the published historical fine-tune, differences were MAE -0.0106,
CI [-0.0165, -0.0046], and Spearman +0.0119, CI [+0.0035, +0.0202].

For the full-development retrospective lockbox refit, development CV selected
Huber 1 at LR 1e-3. It scored lockbox MAE 0.5748 and Spearman 0.6084, versus
0.5969 and 0.5812 for the published historical head. This is retrospective and
must be frozen for a new unopened set before being called a confirmed gain.

Huber delta 0.5 is 0.5 log10 concentration units, or a 3.16-fold concentration
error; it is 3.27 times the median assay SE (0.153) and 0.326 of the development
pEC50 IQR. The complete per-fold traces and aggregate uncertainty appear in
[validation_loss_vs_learning_rate.png](experiment_c_loss_lr/validation_loss_vs_learning_rate.png),
with raw screens, selections, predictions, calibration, tail metrics, seed
variability, and bootstrap intervals in
[experiment_c_loss_lr](experiment_c_loss_lr/).

## Reproduction and resources

The three commands were run sequentially with the same envelope:

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 MALLOC_ARENA_MAX=2 PYTHONPATH=src \
  taskset -c 6-11 nice -n 10 \
  python \
  scripts/run_cpu_followup_experiments.py --stage <stage> --max-workers 6
```

| Stage | Wall time | Peak parent RSS | Peak child RSS |
|---|---:|---:|---:|
| A: capacity | 302.8 s | 0.84 GiB | 0.68 GiB |
| B: standardization | 423.1 s | 3.36 GiB | 0.68 GiB |
| C: loss x LR | 519.6 s | 0.83 GiB | 0.68 GiB |

Exact argv, environment/affinity, versions, input SHA-256 hashes, runtime, and
RSS are recorded in each stage's `run_metadata.json`. The CPU jobs were confined
to physical CPUs 6--11, used at most six workers, and had CUDA disabled. They
did not touch the concurrently running protein-pooling GPU extraction.
