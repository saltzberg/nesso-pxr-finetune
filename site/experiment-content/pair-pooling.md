# Final pair pooling — completed

Status: **completed** · Synthesis updated 2026-09-08. Original dates and immutable protocol history remain in the source evidence.

## Question

Are direct receptor–ligand cells needed at the final pooling operation?

<!-- experiment-summary:start -->
<p class="summary-message">Ligand–ligand final pooling improves development MAE, but its validation advantage is unresolved.</p>
<figure class="experiment-summary"><a href="../../assets/experiment-summaries/pair-pooling.svg" title="Open full-size figure"><img src="../../assets/experiment-summaries/pair-pooling.png" alt="Ligand–ligand final pooling improves development MAE, but its validation advantage is unresolved."></a><figcaption>Nested mean-384D ridge, 3,344 development and 790 previously opened validation compounds. Left: unweighted compound-level pEC50 MAE for three final pools (lower is better). Right: ligand–ligand minus all-pairs MAE; negative favors ligand–ligand. Saved paired 95% chemical-family bootstrap intervals use 2,000 replicates (2,931 development and 735 validation clusters). Upstream features were frozen and already protein-conditioned; this is not a protein-removal ablation.</figcaption></figure>
<p class="summary-downloads"><a href="../../assets/experiment-summaries/pair-pooling.png" download>PNG</a> · <a href="../../assets/experiment-summaries/pair-pooling.svg" download>SVG</a> · <a href="../../assets/experiment-summaries/pair-pooling.csv" download>Plotted data</a> · <a href="../../assets/experiment-summaries/pair-pooling.json" download>Figure provenance</a></p>
<!-- experiment-summary:end -->

## Design and fitted/frozen flow

Upstream protein-conditioned features [frozen] → all-pairs, ligand–ligand-only or receptor–ligand-only final pool → nested fitted mean-384D ridge.

## Results

All 4,134 compounds extracted with zero failures and positive-control parity. Ligand-only final pooling improves raw development MAE from 0.63279 to 0.60624; all-790 validation changes 0.59630 to 0.59421 with an unresolved paired interval.

## Implication and limits

Ligand–ligand cells were already protein-conditioned. This does not show that Nesso ignores protein or establish a causal protein ablation.

All evidence is retrospective. Historical budgets, splits and raw versus uncertainty-weighted scores must remain distinct.
