# Descriptor + frozen-Nesso hybrid pilot

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Can a weaker standalone representation add useful complementary signal?

<!-- experiment-summary:start -->
<p class="summary-message">The pilot blend helps higher potency, but hurts lower potency</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/hybrid-pilot.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/hybrid-pilot.png" alt="The pilot blend helps higher potency, but hurts lower potency"></a><figcaption>One preselected reused outer fold (fold 0), 669 compounds: 476 at pEC50 ≥4 and 193 below 4. N500 = 400 FIT + 100 CAL; CAL outcomes unused. Frozen-vector ridge receives a FIT-inner-OOF-selected mixture weight of approximately 0.3983; this is not LoRA. Errors use weights 1/max(reported assay SE, 0.1); lower weighted MAE is better. Group rows are within-stratum scores, not contributions to the total; the higher-potency group carries 83.42% of evaluation weight. Exploratory pilot, not independent confirmation.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/hybrid-pilot.png" download>PNG</a> · <a href="../../assets/experiment-summaries/hybrid-pilot.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/hybrid-pilot.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/hybrid-pilot.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

N500 strict fold0: 400 FIT → nested grouped OOF descriptor and stable-ridge predictions → convex mixture weight [FIT-selected]; native vectors frozen. CAL outcomes unused.

## Results

The pilot improves weighted MAE from descriptor 0.502895 to blend 0.485238, with selected Nesso weight about 0.3983. Higher-potency accuracy improves while lower-potency error worsens.

## Implication and limits

This is frozen-representation ridge, not LoRA. One inspected fold motivated the locked continuation; it is not independent confirmation.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
