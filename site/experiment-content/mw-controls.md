# Can simple molecular-weight controls explain the correction?

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Is the low-MW correction uniquely neural?

<!-- experiment-summary:start -->
<p class="summary-message">A scalar interaction beats LoRA on small compounds—not larger ones</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/mw-controls.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/mw-controls.png" alt="A scalar interaction beats LoRA on small compounds—not larger ones"></a><figcaption>Pooled N500 held-out scores: 237 compounds below 215 Da and 3,107 above; none exactly 215 Da. N500 = 400 FIT + 100 CAL per fold, one draw and seed; five reused strict-chemical development folds, not independent replicates or a fresh holdout. Score+MW has three fitted coefficients; interaction adds a fourth. Controls used FIT-only weighted least squares, not the neural objective; the native reference is unchanged. Errors use weights 1/max(reported assay SE, 0.1); lower weighted MAE is better. Subgroup correction does not imply a biological mechanism.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/mw-controls.png" download>PNG</a> · <a href="../../assets/experiment-summaries/mw-controls.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/mw-controls.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/mw-controls.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen native scalar + continuous MW → FIT-only weighted linear scaling: score only, MW only, additive score+MW, or score+MW+interaction. CAL is reserved for intervals; no threshold-based rule.

## Results

All 40 fits completed. For 237 compounds below 215 Da at N500, weighted MAE: native 1.834; LoRA 1.179; additive score+MW 1.163; interaction 0.863.

## Implication and limits

The interaction is four coefficients, not MW alone. A conspicuous subgroup correction is not proof that MW explains all aggregate gains, binding, or activation biology.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
