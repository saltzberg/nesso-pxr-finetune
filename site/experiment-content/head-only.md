# Initial head-only versus rich 2D

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

How much can the final regression heads recover?

<!-- experiment-summary:start -->
<p class="summary-message">Both fitted models compress the potent tail despite lower overall error for LightGBM.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/head-only.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/head-only.png" alt="Both fitted models compress the potent tail despite lower overall error for LightGBM."></a><figcaption>Each panel shows all 3,344 identical development compounds, using five-fold nested out-of-fold predictions; diagonal is exact prediction. Heads average three seeds. The 55 observed pEC50 &gt;6 compounds include only one head prediction &gt;6 and no LightGBM prediction &gt;6. Closer to the diagonal is better; raw errors, not uncertainty-weighted scores. No lockbox or challenge rows.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/head-only.png" download>PNG</a> · <a href="../../assets/experiment-summaries/head-only.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/head-only.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/head-only.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Two cached 384D vectors [frozen] → twin pretrained continuous MLPs [fitted]. Five nested chemical-family folds; inner selection and three-seed refits. The matched rich 2D LightGBM selects features within training folds.

## Results

Development raw MAE: native 0.915, tuned heads 0.633, rich 2D 0.516. Both learned models compress the highest-potency tail.

## Implication and limits

Historical 714-row qualified lockbox, later all-790 validation, and 513 challenge rows are distinct cohorts. These retrospective raw-MAE scores are not the later low-data weighted scores.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
