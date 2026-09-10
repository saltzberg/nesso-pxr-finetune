# Matched upstream-LoRA predictive comparison

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Does actual upstream adaptation add to the same trained heads?

<!-- experiment-summary:start -->
<p class="summary-message">LoRA adds a small gain; descriptors lead at 500 labels</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/upstream-comparison.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/upstream-comparison.png" alt="LoRA adds a small gain; descriptors lead at 500 labels"></a><figcaption>Pooled held-out scores, 3,344 compounds once per method and budget. N100 = 80 FIT + 20 CAL; N500 = 400 FIT + 100 CAL per fold, one draw and seed; five reused strict-chemical development folds, not independent replicates or a fresh holdout. Both neural arms used 40 full-FIT updates with matched optimizer; controls had different tuning procedures. Errors use weights 1/max(reported assay SE, 0.1); lower weighted MAE is better.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/upstream-comparison.png" download>PNG</a> · <a href="../../assets/experiment-summaries/upstream-comparison.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/upstream-comparison.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/upstream-comparison.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Both affinity members: pretrained continuous heads [fitted], plus rank-4 LoRA at esm_proj.1/.3 and pairformer_stack.layers.0.transition_z.fc1 [fitted only in LoRA arm]. Main trunk and external ESM encoder frozen. N100=80 FIT+20 CAL; N500=400+100; 40 full-FIT optimizer updates, one draw and seed, no tuning.

## Results

All 20 neural fits and 30 control cells completed. N500 pooled weighted MAE: head 0.5773, LoRA 0.5610, frozen ridge 0.5345, descriptors 0.5006. N100 favors LoRA on weighted MAE but not all metrics.

## Implication and limits

Low-potency correction supplies most net LoRA-versus-head gain; pEC50 ≥4 weighted error slightly worsens. This fixed training recipe does not exhaust optimization; no fresh holdout.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
