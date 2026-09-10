# Conception: can binding features predict cellular activation?

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Does adapting Nesso improve PXR activity prediction, compete with ligand-only models, and add complementary information?

<!-- experiment-summary:start -->
<p class="summary-message">Head adaptation improves on the native continuous score, but ligand-only LightGBM remains stronger.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/conception.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/conception.png" alt="Head adaptation improves on the native continuous score, but ligand-only LightGBM remains stronger."></a><figcaption>Same 3,344 development compounds and five chemical-family outer folds; pooled unweighted MAE, lower is better. Native continuous affinity is a pIC50-equivalent score, not a measured cellular pEC50. Learned readouts use training-only selection; fixed legacy 2D unsupervised prefilter was not rebuilt within folds. Retrospective evidence.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/conception.png" download>PNG</a> · <a href="../../assets/experiment-summaries/conception.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/conception.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/conception.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Prepared protein + ligand → pretrained Nesso [frozen] → two affinity vectors → continuous readouts [PXR-fitted] → 6 − mean output. The original specification proposed staged calibration, head tuning and deeper adaptation.

## Results

The initial matched nested comparison answered yes to improvement over native, no to beating rich 2D descriptors. Later experiments refine the complementarity answer rather than erase this starting point.

## Implication and limits

Nesso predicts distograms and scalar affinity; it is not a cofolder that generates coordinates. Numerical affinity conversion does not make binding and cellular activation biologically equivalent.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
