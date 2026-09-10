# Movement, chemistry and native feature geometry

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

What distinguishes compounds whose predictions change?

<!-- experiment-summary:start -->
<p class="summary-message">Smaller compounds tend to move more under adaptation</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/geometry.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/geometry.png" alt="Smaller compounds tend to move more under adaptation"></a><figcaption>All 3,344 development compounds, one saved N500 head-plus-LoRA prediction each. N500 = 400 FIT + 100 CAL per fold, one draw and seed; five reused strict-chemical development folds, not independent replicates or a fresh holdout. Unweighted absolute movement |LoRA − native| is not error reduction. Native is the released IC50-equivalent scalar conversion, not measured cellular potency. Full MW and movement ranges retained; size association is descriptive, not a causal mechanism.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/geometry.png" download>PNG</a> · <a href="../../assets/experiment-summaries/geometry.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/geometry.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/geometry.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Saved native and adapted predictions + actual FIT-neighbor identities + MW/charge + frozen 768D native vectors → descriptive ranks, strata and PCA; nothing fitted for prospective prediction.

## Results

Large movers are not merely closest FIT neighbors. Smaller compounds move more; native feature norm is associated with movement more strongly than accuracy improvement. Native PCA’s first component captures 66.8% of variance.

## Implication and limits

Full-cohort PCA is descriptive. Adapted internal tensors were not analyzed; neither feature norm nor proximity proves mechanism. Low MW and low potency are different variables.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
