# Emax and uncertainty: testing the biological explanation

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Are small molecules overcalled because they bind without activating PXR?

<!-- experiment-summary:start -->
<p class="summary-message">Small compounds have higher uncertainty, not lower normalized efficacy</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/efficacy-uncertainty.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/efficacy-uncertainty.png" alt="Small compounds have higher uncertainty, not lower normalized efficacy"></a><figcaption>Unweighted empirical cumulative distributions for all 3,344 saved development IDs: 237 below 215 Da and 3,107 at least 215 Da; no missing values or Emax refiltering. Each compound appears once in each panel; each stratum has its own denominator. Normalized Emax uses a logarithmic x-axis and retains values above one and the full tail. Raw baseline-relative Emax is a different scale and is lower in small compounds. Higher reported SE describes less precise fitted potency, not a demonstrated biological cause; dose-response curves and the normalization denominator are unavailable. No model fit or inference.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/efficacy-uncertainty.png" download>PNG</a> · <a href="../../assets/experiment-summaries/efficacy-uncertainty.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/efficacy-uncertainty.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/efficacy-uncertainty.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Exact 3,344 development IDs → provider Emax/SE fields + saved native/adapted predictions → MW strata and descriptive adjustments. No model inference or predictive fitting.

## Results

Below215 versus larger: normalized Emax medians 1.240 vs 0.977; raw log2 fold-change Emax 2.17 vs 2.53; pEC50 SE 0.558 vs 0.141. All 237 small compounds pass normalized Emax ≥0.6 qualification.

## Implication and limits

The proposed low normalized efficacy explanation is weakened, not established. Higher fit uncertainty is associated with size/potency, but neither scalar affinity nor low pEC50 proves binding without activation. Raw curves, control denominator and parameter covariance are unavailable; the adapter is not shown to learn efficacy.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
