# Locked hybrid — all five folds completed

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Does the same training-selected blending procedure hold across the remaining folds?

<!-- experiment-summary:start -->
<p class="summary-message">Every fold gains overall—and pays a lower-potency penalty</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/hybrid-allfolds.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/hybrid-allfolds.png" alt="Every fold gains overall—and pays a lower-potency penalty"></a><figcaption>Paired within-fold weighted-MAE differences for the same held-out compounds: fold totals 669/669/669/668/669; 3,344 unique compounds overall, 1,036 below pEC50 4. N500 = 400 FIT + 100 CAL per fold, one draw and seed; five reused strict-chemical development folds, not independent replicates or a fresh holdout. Fold 0 reuses the inspected pilot byte-identically; folds 1–4 apply the locked procedure, selecting their own FIT-only blend weights. Errors use weights 1/max(reported assay SE, 0.1); lower weighted MAE is better. Below-4 compounds carry 16.4% of pooled weight; this is not uniform improvement or retraining-variability evidence.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/hybrid-allfolds.png" download>PNG</a> · <a href="../../assets/experiment-summaries/hybrid-allfolds.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/hybrid-allfolds.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/hybrid-allfolds.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Frozen vectors → stable ridge and ligand descriptors → LightGBM [independently FIT-selected]. Three grouped meta folds and deeper selection create honest inner OOF inputs; convex weighted-MAE mixture fitted inside FIT. Fold0 reused byte-identically; folds1–4 newly fit.

## Results

Pooled weighted MAE improves 0.500588 → 0.486790; raw MAE 0.599636 → 0.594731; Spearman 0.656863 → 0.686392. Weighted error improves in all five folds. At pEC50 ≥4: 0.434197 → 0.411839; below4: 0.840152 → 0.870134.

## Implication and limits

The lower-potency group worsens in every fold. Weights are 1/max(SE,0.1); below4 is 31% of compounds but 16.4% of weight. Locked recipe means procedure, not one fixed mixture coefficient. No prospective holdout or retraining variability estimate.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
