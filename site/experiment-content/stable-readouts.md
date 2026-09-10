# Stable scaling and readouts

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Were poor vector probes an avoidable preprocessing problem?

<!-- experiment-summary:start -->
<p class="summary-message">Stable scaling and target centering reduce MLP error; adding weight decay barely changes the curve.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/stable-readouts.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/stable-readouts.png" alt="Stable scaling and target centering reduce MLP error; adding weight decay barely changes the curve."></a><figcaption>Retrospective development cohort of 3,344 compounds; each mean-task point equally averages five outer folds × ten paired draws (50 tasks), not a pooled weighted score. Neural predictions average three seeds before scoring; repeated draws are dependent. No inferential bands are drawn. Weighted MAE uses 1/max(assay SE, 0.10); lower is better. Budgets include 80% FIT/inner selection and 20% CAL labels. Chemical-cluster split, original and follow-up MLP recipes, N25–500. All use the same frozen concatenated 768D vectors and 768–128–1 MLP architecture. Stable per-feature scaling has a training-only variance floor; centered variants subtract and restore the training weighted target mean. Only the final variant adds AdamW decay 0.01. The centered and regularized curves nearly coincide; no prediction clipping or outlier exclusion.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/stable-readouts.png" download>PNG</a> · <a href="../../assets/experiment-summaries/stable-readouts.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/stable-readouts.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/stable-readouts.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen native vectors → training-only stable preprocessing and fitted readout; ligand features → training-only reduction and fitted LightGBM. Strict branch purges FIT/CAL pools at Morgan similarity ≥0.35 to test. All labels remain inside the acquired budget.

## Results

Stable scaling and target centering substantially improve MLP performance; added weight decay contributes little. Stable ridge is competitive, but its small N500 advantage over the pretrained head is unresolved.

## Implication and limits

All 8,100 new follow-up tasks completed, plus 1,200 original comparator tasks replayed. Repeated draws are dependent; chemical novelty and prospective coverage remain unresolved.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
