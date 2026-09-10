# Four-CYP transfer of scalar MW controls

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Does the PXR MW effect recur for direct CYP inhibition?

<!-- experiment-summary:start -->
<p class="summary-message">MW helps two CYP assays; the interaction is not a general gain</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/cyp-transfer.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/cyp-transfer.png" alt="MW helps two CYP assays; the interaction is not a general gain"></a><figcaption>Primary five-fold Butina pooled same-row direct-inhibition evaluation, not TDI: CYP1A2/2C9/2D6/3A4 n=1,412/1,285/1,493/2,335 (6,525 compound–endpoint observations, 4,905 unique compounds). Equal-weight OLS and MAE: supplied std/CI are not justified sampling SE, so no PXR uncertainty floor. Left: additive minus fitted-score-only; right: interaction minus additive. Whiskers are saved 95% paired whole-group bootstrap intervals (2,000 draws, seed 42), conditional on saved fits, no retraining uncertainty or multiplicity adjustment. Reused development splits, no fixed PXR N500 budget.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/cyp-transfer.png" download>PNG</a> · <a href="../../assets/experiment-summaries/cyp-transfer.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/cyp-transfer.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/cyp-transfer.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Existing native protein/ligand/heme inference [frozen] + identity-SMILES MW → training-fold scaling and fitted OLS scalar controls → same-row pIC50 evaluation across CYP1A2, 2C9, 2D6 and 3A4.

## Results

Additive MW helps CYP2C9 and CYP3A4; the score–MW interaction does not consistently outperform additive. Same-CYP inhibitor baselines remain stronger overall. Small-MW groups contain only 22 / 11 / 21 / 30 compounds.

## Implication and limits

CYP uses unweighted fitting and MAE: supplied std/CI are not justified sampling SE. Do not transfer PXR’s uncertainty floor or potency threshold. This is not a general replication of the PXR effect.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
