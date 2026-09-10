# Proximity diagnostic — design review

## Recommended figure

Use one compact 2×2 scatter grid: rows **N100 (80 FIT + 20 calibration)** and **N500 (400 FIT + 100 calibration)**; columns **Head-only** and **Head + LoRA**. Plot all **3,344 held-out compounds per panel**, pooling their saved development-fold predictions. Each compound's x-coordinate must use its own fold's actual FIT set at that budget, not a global training pool. Within a row, x-coordinates are identical across arms.

- **x:** `Nearest FIT Morgan Tanimoto similarity` (radius 2, 2,048 bits); larger means chemically closer. Use FIT similarity, not FIT+CAL similarity or `1 − similarity`.
- **y:** `Absolute prediction change from native (log10 units)`, calculated as `abs(adapted prediction − saved native prediction)`. The second column is not incremental LoRA minus head-only.
- Draw small neutral-gray points with moderate transparency; overlay the independently selected top 20 native-relative movers last, with slightly larger opaque markers. Reuse host colors: head-only blue `#2166ac`, head + LoRA orange `#b85c00`. Use saved selections from `proximity_top20.csv`, filtered to the two native-relative comparisons; do not accidentally select the incremental comparison. Keep identities overlapping between panels when selected.
- Square panel boxes, about 300–340 px each, with shared linear axes and thin left/bottom spines. Do **not** use equal data-unit aspect: similarity and log-score displacement are different quantities. One column header per arm, row labels outside the data area, a shared y label, and a short legend below. No regression, diagonal, median lines, correlation badges, molecule labels, or decorative grid. No jitter, binning, or downsampling.
- Match the renderer's existing Matplotlib → SVG/PNG workflow and `figure()` embedding rather than introducing Canvas. Limit this figure's HTML width to roughly 760 px (`width:100%; max-width:760px; height:auto`) so the existing global `img{width:100%}` does not enlarge it across the entire report. Place it with the proximity discussion before the decision, subordinate to the primary performance results.

## Semantics and interpretation

The summaries support **no positive concentration of the largest changes near the upper end of the observed FIT-similarity range**. Top-20 median similarities are 0.214/0.198 at N100 and 0.250/0.268 at N500, versus all-compound medians 0.237 and 0.277. Reported pooled change–similarity correlations are weakly negative. Avoid the stronger claim that chemical proximity never matters: the strict split excludes similarity ≥0.35, and the reported maximum is 0.349398. This is a deliberately restricted low-similarity regime, not a test involving close analogues across the full 0–1 range.

Magnitude is neither signed movement nor accuracy improvement. A large displacement can overshoot the observed response; this figure cannot establish better correction, memorization, causation, or absence of pretraining overlap. Native is an unchanged IC50-equivalent conversion, while the observed endpoint is pEC50: label the vertical quantity as prediction displacement in log units, not measured affinity gain or error reduction. Morgan proximity is not learned Nesso feature proximity. Pooling folds is descriptive, not independent replication. Higher N500 nearest-neighbor similarity also reflects a larger FIT reference set, not a change in held-out molecules.

## Full-range / no-clipping acceptance criteria

Use a shared x domain from 0 to just above 0.35 (for example 0.36), with a labeled 0.35 tick and enough right margin to retain complete markers. This deliberately displays the observed low-similarity range; it is not the full theoretical Tanimoto domain. Derive one y upper limit from the maximum absolute change across **all four complete panels**, then add headroom. Keep zero labeled and allow small lower marker padding so points at zero are not cut in half. Do not use quantile limits, winsorization, separate panel scales, or an assumed maximum from summary statistics. The supplied summary CSV has no maximum-displacement column, so the exact y upper bound must come from row-level predictions.

Before publishing, assert 3,344 finite, uniquely identified records per panel and exactly 20 highlighted IDs per panel; verify joins against record ID, budget, and fold as applicable. Check every center and marker extent lies within the displayed/exported region. Inspect PNG/SVG and the rendered HTML at desktop and narrow widths for cropped labels, boundary markers, and unreadably small facets. `bbox_inches='tight'` protects export layout, not data outside axis limits. Overlap is acceptable; silently dropping records is not.

## Suggested caption

> Chemical proximity and prediction displacement. Each panel shows all 3,344 held-out compounds; colored points mark the 20 largest absolute changes from the saved native baseline, selected separately for each arm and budget. Similarity is the maximum Morgan radius-2, 2,048-bit Tanimoto to the compound's own fold-specific FIT set (80 compounds at N100; 400 at N500); calibration compounds are excluded. Axes are shared and retain all values. The strict split excludes similarity ≥0.35, so this view covers only low-similarity compounds. The largest changes are not concentrated toward the high-similarity end of this range. Displacement is not accuracy improvement, and this retrospective diagnostic does not measure learned-feature proximity or pretraining overlap.

Reviewed `PROXIMITY_RESULTS.txt`, `proximity_summary.csv`, and `render_report.py`. This is design advice only; no figure was rendered or row-level verification performed.
