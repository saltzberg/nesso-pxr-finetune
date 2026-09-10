# Nested loss × learning rate

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Would objective selection relieve head compression?

<!-- experiment-summary:start -->
<p class="summary-message">Joint loss and learning-rate selection reduces average error without restoring predictions above pEC50 6.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/loss-lr.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/loss-lr.png" alt="Joint loss and learning-rate selection reduces average error without restoring predictions above pEC50 6."></a><figcaption>All 3,344 nested development predictions per model; 55 observations exceed pEC50 6. Each objective selects LR within outer training folds, with weight decay zero; joint selector also selects objective. Raw MAE lower is better; maximum prediction is a range diagnostic, not an accuracy metric. Dashed line marks pEC50 6. No uncertainty inferred from these maxima.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/loss-lr.png" download>PNG</a> · <a href="../../assets/experiment-summaries/loss-lr.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/loss-lr.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/loss-lr.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Released twin heads [fitted] on frozen vectors; objective and learning rate chosen inside each outer training set; fixed weight decay isolates this comparison.

## Results

The joint nested selector reaches raw development MAE 0.6224, versus 0.6330 historically. None of its development predictions exceed pEC50 6: lower average error does not solve the active tail.

## Implication and limits

Retrospective optimization evidence; freeze a recipe before testing new data.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
