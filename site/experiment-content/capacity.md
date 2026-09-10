# Readout capacity control

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Does greater nonlinear readout capacity help?

<!-- experiment-summary:start -->
<p class="summary-message">Extra readout capacity changes error modestly and inconsistently across frozen representations.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/capacity.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/capacity.png" alt="Extra readout capacity changes error modestly and inconsistently across frozen representations."></a><figcaption>Complete 4-representation × 3-readout matrix: 3,344 nested development and 790 retrospective validation compounds. Scaling and five tuning candidates per readout are training-only; neural predictions average three seeds. Raw MAE lower is better. Lines link architecture choices, not parameter counts. Primary mean-384D development gains over ridge have unresolved paired intervals; upstream pretraining is unchanged.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/capacity.png" download>PNG</a> · <a href="../../assets/experiment-summaries/capacity.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/capacity.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/capacity.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen affinity representations → matched random-initialized readouts of different capacities [fitted using nested training selection].

## Results

The completed CPU capacity matrix separates architecture from released-head initialization; larger readouts do not establish a solution to the persistent descriptor gap. Consult the complete per-model table rather than select a single favorable run.

## Implication and limits

Only downstream capacity changes; upstream pretraining is not ablated.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
