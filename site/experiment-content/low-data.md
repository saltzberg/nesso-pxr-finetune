# Original low-data matrix — completed

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

How should 25–500 acquired PXR labels be spent?

<!-- experiment-summary:start -->
<p class="summary-message">Continuous heads improve with more labels; the original vector probes show large, irregular errors.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/low-data.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/low-data.png" alt="Continuous heads improve with more labels; the original vector probes show large, irregular errors."></a><figcaption>Retrospective development cohort of 3,344 compounds; each mean-task point equally averages five outer folds × ten paired draws (50 tasks), not a pooled weighted score. Neural predictions average three seeds before scoring; repeated draws are dependent. No inferential bands are drawn. Weighted MAE uses 1/max(assay SE, 0.10); lower is better. Budgets include 80% FIT/inner selection and 20% CAL labels. Original chemical-cluster split only, N25–500; full outer-pool reference omitted. Panels have different y ranges to keep the practical head comparison legible. Saved-model inspection in the independent assessment found scaling/extrapolation failures; the curves alone do not isolate a scaling cause. Random versus pretrained refers only to head initialization, not upstream pretraining. Native output is an unchanged binding-affinity anchor, not native cellular pEC50.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/low-data.png" download>PNG</a> · <a href="../../assets/experiment-summaries/low-data.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/low-data.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/low-data.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen Nesso representations or Morgan features → 12 fitted/native methods → within-budget interval calibration. Two split policies, five folds, ten paired draws, five small budgets plus full-pool reference; three neural seeds per task.

## Results

All 7,200 task systems completed. Continuous heads are practical Nesso candidates; released initialization has no consistent advantage. Original generic MLP/ridge failures expose scaling and extrapolation problems rather than absence of feature signal.

## Implication and limits

N25 means 20 FIT and five CAL labels; nominal 90% intervals can be unbounded. Full-pool is an outer-training-pool reference, not all 3,344 labels.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
