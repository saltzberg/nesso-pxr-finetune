# Strict chemical separation

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Does purging neighbors change practical generalization?

<!-- experiment-summary:start -->
<p class="summary-message">Neighbor purging barely moves overall error, but the most novel strict-test predictions remain harder.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/strict-followup.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/strict-followup.png" alt="Neighbor purging barely moves overall error, but the most novel strict-test predictions remain harder."></a><figcaption>Retrospective development cohort of 3,344 compounds; each mean-task point equally averages five outer folds × ten paired draws (50 tasks), not a pooled weighted score. Neural predictions average three seeds before scoring; repeated draws are dependent. No inferential bands are drawn. Weighted MAE uses 1/max(assay SE, 0.10); lower is better. Budgets include 80% FIT/inner selection and 20% CAL labels. N500 = 400 FIT + 100 CAL. Strict pools exclude every candidate with Morgan similarity ≥0.35 to any test compound; test identities are unchanged, training draws can change. Left: mean-task scores. Right: strict pooled within-stratum scores across ten dependent draws. Similarity &lt;0.20 contains 906 prediction rows, 257 unique compounds and 1.93% of weight; 0.20–&lt;0.35 contains 32,534 rows, 3,319 unique compounds and 98.07% of weight. Stratum membership varies by FIT draw, so unique-compound counts overlap. The shared numerical y range supports comparison but the aggregation differs explicitly between panels; novelty results are descriptive, not prospective generalization guarantees.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/strict-followup.png" download>PNG</a> · <a href="../../assets/experiment-summaries/strict-followup.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/strict-followup.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/strict-followup.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen native vectors → training-only stable preprocessing and fitted readout; ligand features → training-only reduction and fitted LightGBM. Strict branch purges FIT/CAL pools at Morgan similarity ≥0.35 to test. All labels remain inside the acquired budget.

## Results

The completed strict audit changes aggregate scores little; actual nearest-FIT similarity below 0.2 remains harder. N500 90% intervals are near 89% coverage but broad.

## Implication and limits

All 8,100 new follow-up tasks completed, plus 1,200 original comparator tasks replayed. Repeated draws are dependent; chemical novelty and prospective coverage remain unresolved.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
