# Binary readout correction and dual heads

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Was evaluating only the continuous output an incomplete test of Nesso?

<!-- experiment-summary:start -->
<p class="summary-message">Using both branches helps scalar calibration, but dual-head adaptation does not improve consistently across cohorts.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/binary-dual.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/binary-dual.png" alt="Using both branches helps scalar calibration, but dual-head adaptation does not improve consistently across cohorts."></a><figcaption>Raw unweighted MAE, lower is better. Development uses 3,344 chemical-family outer-fold predictions; public challenge uses 513 retrospective public-label compounds. Calibration and adapters fitted on development only. Binary outputs predict binding, not cellular activation; calibrations here target functional pEC50. The challenge LightGBM is the established augmented/offset comparator, not the exact nested-development estimator.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/binary-dual.png" download>PNG</a> · <a href="../../assets/experiment-summaries/binary-dual.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/binary-dual.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/binary-dual.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen vectors → released continuous and binary branches → training-fitted linear/quadratic mapping, or both branches [fine-tuned] with bounded fusion.

## Results

Combined quadratic calibration reaches raw development MAE 0.774; dual-branch adaptation 0.629 versus continuous-only 0.633. Dual-head challenge MAE worsens to 0.647 versus 0.601.

## Implication and limits

Binder probability is not a cellular activity classifier. Potency thresholds are functional proxies, not binding labels; the small dual-head gain is not robust across cohorts.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
