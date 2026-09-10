# Stronger low-data descriptor baseline

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Does a broader ligand-only baseline change the low-data conclusion?

<!-- experiment-summary:start -->
<p class="summary-message">Descriptors overtake the head from N250 overall, while the head retains lower N500 error at pEC50 ≥4.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/descriptor-followup.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/descriptor-followup.png" alt="Descriptors overtake the head from N250 overall, while the head retains lower N500 error at pEC50 ≥4."></a><figcaption>Retrospective development cohort of 3,344 compounds; each mean-task point equally averages five outer folds × ten paired draws (50 tasks), not a pooled weighted score. Neural predictions average three seeds before scoring; repeated draws are dependent. No inferential bands are drawn. Weighted MAE uses 1/max(assay SE, 0.10); lower is better. Budgets include 80% FIT/inner selection and 20% CAL labels. Strict chemical separation (retained FIT/CAL-to-test Morgan similarity &lt;0.35). Left: mean-task scores. Right: N500 pooled within-stratum weighted errors across ten dependent draws, not means of task errors; 1,036 unique compounds / 10,360 predictions below pEC50 4 and 2,308 / 23,080 at or above 4 (16.35% and 83.65% of evaluation weight). Descriptor features are canonical-SMILES RDKit2D, Mordred2D and Morgan with training-only reduction; this is not the historical richer descriptor pipeline. Potency strata are retrospective, not a routing rule.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/descriptor-followup.png" download>PNG</a> · <a href="../../assets/experiment-summaries/descriptor-followup.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/descriptor-followup.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/descriptor-followup.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen native vectors → training-only stable preprocessing and fitted readout; ligand features → training-only reduction and fitted LightGBM. Strict branch purges FIT/CAL pools at Morgan similarity ≥0.35 to test. All labels remain inside the acquired budget.

## Results

Descriptor LightGBM has lower mean-task weighted error from N250 upward. N500: 0.497 versus adapted head 0.542. The head remains better within the pEC50 ≥4 stratum in this repeated-draw experiment.

## Implication and limits

All 8,100 new follow-up tasks completed, plus 1,200 original comparator tasks replayed. Repeated draws are dependent; chemical novelty and prospective coverage remain unresolved.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
