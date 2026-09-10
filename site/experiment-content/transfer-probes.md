# Transfer, learning-rate sensitivity and frozen probes

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Is useful information in the vectors, or only in the pretrained nonlinear heads?

<!-- experiment-summary:start -->
<p class="summary-message">Frozen-vector ridge recovers substantial PXR signal without the pretrained nonlinear heads.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/transfer-probes.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/transfer-probes.png" alt="Frozen-vector ridge recovers substantial PXR signal without the pretrained nonlinear heads."></a><figcaption>Pooled outer-fold development metrics for 3,344 compounds. Ridge penalties selected within training folds; mean and member probes use 384D, concatenation 768D. Raw MAE lower and Spearman higher are better. These estimates do not prove equivalence or protein-specific causality; exploratory nonnested MLP screens are not shown.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/transfer-probes.png" download>PNG</a> · <a href="../../assets/experiment-summaries/transfer-probes.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/transfer-probes.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/transfer-probes.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen vectors → training-selected ridge or shallow MLP; compare expanded nested learning rates, target/loss sensitivities, residual ridge and learned stacking. Encoders remain frozen.

## Results

Frozen mean-vector ridge reaches about 0.633 raw development MAE with better ranking than the original heads. Expanding optimization does not close the descriptor gap; much useful signal is linearly accessible.

## Implication and limits

These controls cannot identify a protein-specific causal effect. Opened validation is retrospective.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
