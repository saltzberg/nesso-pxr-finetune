# Nesso-1 PXR transfer and validation audit

**Date:** 2026-08-12  
**Status:** retrospective diagnostic analysis; the historical lockbox has already
been opened.

## Bottom line

The strongest supported result is now about the **representation**, not the
pretrained regression head. A leakage-safe nested ridge probe on the frozen
Nesso vectors matches the published head fine-tune and ranks compounds better.
Exploratory randomly initialized shallow heads improve further. This means
substantial PXR-relevant information is already accessible in the 384D
features, particularly affinity member 2.

It does **not** yet show that PXR-specific protein--ligand interactions cause
that signal. PXR is constant, and the cached vector is downstream of both
protein and ligand processing. A direct pair-pooling intervention has now been
implemented and smoke-validated exactly, but the full 4,134-compound causal
performance run has not been executed.

The original learning-rate grid was under-bounded: all five outer folds chose
its maximum, 3e-4. An expanded nested rerun selected 1e-3 to 1e-2 in every fold,
but did not improve untouched-fold performance. The methodological criticism
was right; the implied performance gain was not.

The 2D LightGBM model remains clearly stronger in absolute prediction. Nesso
adds a small, reproducible amount of ranking information to the 2D model, but
the MAE gain from stacking is not statistically resolved.

## What was actually fine-tuned

Only these released checkpoint parameters were updated:

- `affinity_module.affinity_heads.to_affinity_pred_value.*`;
- `affinity_module2.affinity_heads.to_affinity_pred_value.*`.

Each member is a 384 → 384 → 384 → 1 ReLU MLP. Together they contain 592,130
trainable parameters. The 384D vectors, affinity-output projection, affinity
pairformers, ESM projection, cofolding trunk, recycling path, pocket selection,
and all structure-related parameters were frozen by caching their outputs.

This was a defensible first transfer experiment because it preserved the
expensive pretrained representation, avoided the upstream inference/training
mode mismatch, reduced optimization cost, and protected the trunk from
catastrophic forgetting. It does not demonstrate that full or partial-trunk
fine-tuning is inferior: neither full affinity-module fine-tuning nor full
Nesso fine-tuning was run.

The original nested grid was:

- learning rate: 1e-5, 3e-5, 1e-4, 3e-4;
- weight decay: 0, 1e-4, 1e-3;
- batch size 64, at most 30 epochs, patience 5;
- gradient clipping at 1.0;
- three outer refits with seeds 42, 43, and 44;
- Huber delta 0.5 with loss weights 0.50 on the member mean and 0.25 on
  each individual member.

All five outer folds selected 3e-4. Selected training durations were 26--28
epochs. The trunk cannot forget because it never receives a gradient. There was
no source-domain replay, L2-SP, EWC, or other constraint protecting the
fine-tuned regression MLPs themselves.

For the small target-specific dataset, model selection and stopping were nested
inside chemical-family folds; outer predictions averaged three seeds, and no
lockbox outcome was used for the historical development fit. Freezing the trunk
was the main capacity and forgetting control. Inner screens considered assay-
uncertainty and curation weights, but the selected outer configurations used
neither. There was no dropout, source-domain replay, or explicit prior-weight
penalty, so 592,130 head parameters remain substantial relative to 3,344
development compounds. The nested ridge result is therefore an important,
more strongly regularized control.

Frozen **scores** did not work well; frozen **representations** did. Head-only
partial fine-tuning worked relative to the released score. Complete affinity-
module and full-trunk fine-tuning remain untested, so no ranking among those
strategies is currently justified.

## 1. Learning-rate boundary

The requested validation-loss-versus-LR plot is
[learning_rate_and_huber_sensitivity.png](learning_rate_and_huber_sensitivity.png).
Its screen is a post-hoc, single-seed five-fold OOF sensitivity analysis at
weight decay zero; it is diagnostic rather than a replacement for nested model
selection.

| LR | OOF Huber loss | OOF MAE |
|---:|---:|---:|
| 3e-4 | 0.21643 | 0.63474 |
| 1e-3 | 0.21192 | 0.62499 |
| 3e-3 | 0.21151 | 0.62441 |
| 1e-2 | 0.21149 | **0.62426** |
| 3e-2 | **0.21142** | 0.62516 |
| 1e-1 | 0.22387 | 0.65429 |

The curve is now bracketed: performance plateaus from 1e-3 through 3e-2 and
degrades at 1e-1. Testing only through 3e-4 was not adequately defended.

More importantly, the expanded grid was rerun inside each outer training
partition. Fold-selected LRs were 3e-3, 1e-2, 3e-3, 1e-3, and 1e-3.
Expanded-grid nested performance was MAE 0.6345 and Spearman 0.6264, versus
historical 0.6330 and 0.6252. The paired chemical-family bootstrap difference
(expanded minus historical) was:

- MAE +0.0016, 95% CI −0.0046 to +0.0074;
- Spearman +0.0012, 95% CI −0.0070 to +0.0097.

Thus the search boundary was a protocol flaw, but correcting it did not yield
a reliable generalization improvement. Full fold selections are in
[expanded_lr_nested/nested_nesso_outer_selections.csv](expanded_lr_nested/nested_nesso_outer_selections.csv),
and paired intervals are in
[expanded_lr_nested_paired_differences.csv](expanded_lr_nested_paired_differences.csv).

## 2. Does the model use protein information?

Architecturally, yes: projected ESM protein embeddings, trunk pair features,
single-to-pair projections, a predicted distogram, and an affinity pairformer
all precede the 384D vector. Empirically in this experiment, **not yet
demonstrated causally**. A constant protein permits a ligand-only function to
fit the labels just as well.

The new direct intervention separates the released final pool into
receptor--ligand and ligand--ligand cells. The 16-compound replay exactly
reproduced the old vectors and head outputs, establishing a valid hook. A full
run takes about 4.5 GPU-hours and is required before claiming a performance
drop. It must be followed by the same nested ridge/head fits for every
representation variant.

Even the ligand--ligand-only pool can contain protein information mixed
upstream. The complete causal ladder therefore also needs pre-pairformer
cross-block removal, shuffled PXR embeddings, a matched decoy receptor, and
ultimately a multi-target protein-swap experiment. The exact design and
interpretation rules are in
[PROTEIN_ABLATION_PROTOCOL.md](PROTEIN_ABLATION_PROTOCOL.md).

## 3. Where does transferable information reside?

The ridge regularization strength was selected inside each outer training
partition. Lockbox models selected their ridge penalty using development folds
only.

| Frozen representation model | Dev MAE | Dev Spearman | Lockbox-790 MAE | Lockbox-790 Spearman |
|---|---:|---:|---:|---:|
| Ridge, member 1 (384D) | 0.681 | 0.581 | 0.646 | 0.554 |
| Ridge, member 2 (384D) | **0.625** | **0.662** | 0.606 | 0.610 |
| Ridge, mean of members (384D) | 0.633 | 0.654 | 0.596 | 0.619 |
| Ridge, concatenated members (768D) | 0.626 | **0.664** | **0.592** | **0.623** |
| Random one-hidden-layer twin MLP | 0.605 | 0.656 | 0.585 | 0.602 |
| Random two-hidden-layer twin MLP | **0.598** | **0.665** | **0.561** | **0.613** |
| Pretrained two-hidden-layer heads, fine-tuned | 0.633 | 0.625 | 0.597 | 0.581 |

The nested ridge result is the cleanest inference: the PXR signal is largely
linearly accessible in the frozen representation, and member 2 is materially
more informative than member 1. The released nonlinear head initialization is
not necessary to recover that signal.

The random MLP capacity ladder is exploratory: LR was selected using the same
five development folds reported in its OOF result, so that development number
is not fully nested, and the lockbox assessment is retrospective. It motivates
a predeclared, fully nested rerun rather than a final superiority claim.

The raw frozen Nesso score itself is much worse (dev MAE 0.915), showing that
the principal transfer failure is the pretrained output mapping, not absence
of useful information in the vector. Full probe results are in
[representation_probe_metrics.csv](representation_probe_metrics.csv).

## 4. Development versus the historical lockbox

The apples-to-apples four-model comparison uses the 714 compounds for which
both historical Nesso and 2D predictions exist.

| Model | Dev n | Dev MAE | Dev Spearman | Common lockbox n | Lockbox MAE | Lockbox Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Nesso-1 | 3,344 | 0.915 | 0.385 | 714 | 0.857 | 0.324 |
| Affine calibration | 3,344 | 0.839 | 0.383 | 714 | 0.820 | 0.324 |
| Fine-tuned Nesso-1 | 3,344 | 0.633 | 0.625 | 714 | 0.601 | 0.602 |
| 2D LightGBM | 3,344 | **0.516** | **0.730** | 714 | **0.478** | **0.741** |

For all 790 historical lockbox compounds, Nesso-only results are:

| Model | MAE | Spearman |
|---|---:|---:|
| Frozen Nesso-1 | 0.815 | 0.312 |
| Affine calibration | 0.811 | 0.312 |
| Fine-tuned Nesso-1 | **0.597** | **0.581** |

There is no historical LightGBM prediction for the other 76 compounds, so no
790-row 2D number should be inferred. The full lockbox has a narrower pEC50
IQR than development (1.21 versus 1.53) and only 9 versus 55 observations above
6. Its slightly lower MAE therefore does not mean stronger transfer; rank
correlation falls from 0.625 to 0.581.

## 5. Chemical split and preprocessing

Chemical identities were RDKit-parsed and converted to canonical isomeric
SMILES. Development compounds were fingerprinted with count-free Morgan
radius-2, 2,048-bit fingerprints and clustered by RDKit Butina at Tanimoto
0.60 (distance 0.40). Compounds sharing either a Butina cluster or a
Bemis--Murcko scaffold group were unioned into indivisible components.
Components were ordered largest-first and greedily assigned to the currently
smallest of five folds, with seed 20260806 used for ties. Acyclic compounds use
their full canonical SMILES as their scaffold key rather than forming one giant
empty-scaffold group.

The observed nearest-training-neighbor distributions validate the intended
guard directly:

| Validation fold | Median Tanimoto | 95th percentile | Maximum | Fraction ≥0.60 |
|---:|---:|---:|---:|---:|
| 0 | 0.367 | 0.529 | 0.597 | 0 |
| 1 | 0.364 | 0.534 | 0.597 | 0 |
| 2 | 0.359 | 0.526 | 0.596 | 0 |
| 3 | 0.368 | 0.527 | 0.597 | 0 |
| 4 | 0.362 | 0.534 | 0.597 | 0 |

No canonical identity, scaffold group, prepared state, or audited Morgan pair
at or above 0.60 crosses a development fold. This is an empirical audit under
this fingerprint definition; it is not a universal definition of an analogue.

For the 790 lockbox compounds, nearest-development Tanimoto has median 0.370,
95th percentile 0.548, and range 0.172--1.000. There are 21 compounds (2.66%)
at or above 0.60. Fine-tuned Nesso MAE by similarity bin is 0.678 (≤0.3),
0.585 (0.3--0.4], 0.552 (0.4--0.5], 0.638 (0.5--0.6], and 0.642 (>0.6).
Performance is not monotonic with similarity, and the last bin has only 19
compounds. On the common 714 rows, LightGBM has lower MAE in every bin.

The preprocessing limitation is important. This repository performs RDKit
parsing plus canonical isomeric SMILES, but does not explicitly salt-strip,
neutralize, or tautomer-canonicalize. All 4,134 supplied records were already
single-fragment and unique under raw, identity-canonical, and prepared SMILES.
Nesso receives one precomputed `top_solution_state` per compound; alternate
tautomers/protonation states are not augmented. Formal charges are retained
(819 of 4,134 records are nonzero). The correct description is therefore
"single selected prepared state," not standardized parent chemistry.

See [chemical_similarity_distributions.png](chemical_similarity_distributions.png),
[nested_fold_similarity_quantiles.csv](nested_fold_similarity_quantiles.csv),
and [smiles_preprocessing_audit.json](smiles_preprocessing_audit.json).

## 6. Why 6 − pEC50?

It is an output-coordinate convention, not a biological hypothesis. If Nesso
emits f and the pEC50 prediction is 6 − f, then:

Huber(f, 6 − y) = Huber(6 − f, y)

because Huber loss is symmetric in the residual. Reparameterizing the final
linear layer as W' = −W and b' = 6 − b gives exactly the same initial function
in direct-pEC50 coordinates. The numerical check found identical loss and a
maximum gradient difference of 6.9e-18.

Thus orientation cannot improve or harm a correctly reparameterized nonlinear
head. Keeping 6 − pEC50 merely preserves the released head's output semantics.
Training directly on pEC50 **without** transforming the final layer would test
a deliberately misaligned initialization, not the orientation question. The
check is in [target_coordinate_equivalence.json](target_coordinate_equivalence.json).

## 7. Huber delta and high-end compression

Delta 0.5 was a fixed heuristic, not derived from this assay. It is 0.5 pEC50
units, corresponding to a 3.16-fold concentration error, 3.27 times the median
reported assay standard error (0.153), and about one third of the development
pEC50 IQR (1.533). Beyond that residual, Huber applies a linear rather than
quadratic penalty.

At the post-hoc selected LR of 1e-2:

| Huber delta | Overall MAE | MAE for observed pEC50 > 6 | Predicted >6 | Prediction maximum |
|---:|---:|---:|---:|---:|
| 0.25 | 0.626 | **1.326** | 0 | 5.601 |
| 0.5 | **0.624** | 1.343 | 0 | 5.498 |
| 1.0 | 0.637 | 1.351 | 0 | 5.553 |
| 2.0 | 0.643 | 1.364 | 0 | 5.838 |
| 10.0 (effectively MSE here) | 0.640 | 1.353 | 0 | 5.525 |

Delta 0.5 is near the overall optimum in this limited screen, but none of the
losses fixes the active-tail compression. Huber can contribute in principle by
downweighting large residuals, yet it is not the primary explanation here:
near-MSE training also predicts no compound above 6. Only 55 development
compounds exceed 6, so scarcity and regression to the mean are more plausible
drivers. Tail-aware sampling or a predeclared weighted/ranking auxiliary loss
should be tested separately from delta.

## 8. Does Nesso add information beyond 2D?

An equal blend is not a sufficient test and is worse than LightGBM. Meta-models
were fit on development OOF predictions and then applied to the common
714-compound lockbox.

| Model | Dev MAE | Dev Spearman | Lockbox MAE | Lockbox Spearman |
|---|---:|---:|---:|---:|
| 2D LightGBM | 0.5160 | 0.7304 | 0.4776 | 0.7405 |
| Equal blend | 0.5407 | 0.7337 | 0.5022 | 0.7368 |
| Learned convex blend | 0.5139 | 0.7368 | 0.4747 | 0.7477 |
| Unconstrained linear stack | **0.5138** | **0.7395** | **0.4722** | 0.7513 |
| 768D Nesso residual ridge | 0.5236 | 0.7389 | 0.4787 | **0.7572** |

The convex fit assigns 10.9% weight to Nesso and 89.1% to 2D. The
unconstrained fit is −0.436 + 0.217 × Nesso + 0.874 × 2D.

Against 2D alone, chemical-family bootstrap differences on lockbox are:

- convex blend MAE −0.0028, 95% CI −0.0070 to +0.0015;
- linear stack MAE −0.0054, 95% CI −0.0116 to +0.0010;
- linear stack Spearman +0.0108, 95% CI +0.0026 to +0.0195;
- representation-residual ridge MAE +0.0011, 95% CI −0.0122 to +0.0176;
- representation-residual ridge Spearman +0.0167, 95% CI +0.0041 to +0.0304.

So Nesso contains incremental **ranking** information beyond the 2D model, but
the stronger residual test does not establish an MAE gain. Its selected ridge
alpha is 1,000, indicating heavy shrinkage. Development stack metrics are not
fully nested at the meta-model level, and the lockbox is no longer prospective.
The proper next test is to freeze the stack and evaluate a new unopened set.

## Revised transfer claim

Nesso-1's frozen, protein-conditioned affinity representations contain
substantial PXR-relevant information that is mostly linearly accessible.
Head-only adaptation and randomly initialized shallow probes exploit it, but a
2D chemical model remains stronger. The current experiment does not establish
that PXR-specific pairwise interactions cause the transferable signal.
Transfer is weakest in the potent tail, in the released scalar head, and in
incremental absolute-error prediction after the 2D model.

The highest-priority remaining experiment is the full direct-pair ablation,
followed by protein counterfactuals. Only after those controls should
affinity-output projection, affinity pairformer, or trunk unfreezing be
interpreted as evidence for protein-aware transfer.
