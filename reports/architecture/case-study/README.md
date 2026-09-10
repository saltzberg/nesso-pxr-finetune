# Fine-tuning Nesso-1 for PXR activation

Created: 2026-09-08. Last edited: 2026-09-08.

The [main experimental page](http://192.168.6.154:8890/) now follows the complete study with individual sub-experiments, including the hybrid, CYP extension and efficacy diagnostic. This page retains the detailed PXR adaptation/MW comparison.

This case study asks what Nesso-1 learns when adapted from binding-affinity prediction to cellular PXR activation, and whether that adaptation is more useful than simpler molecular models.

**The central diagnostic finding is the correction of low-molecular-weight compounds that native Nesso overpredicts. Fine-tuning makes large downward corrections for these molecules. Adding molecular weight to a simple fitted mapping of the native Nesso score recapitulates this behavior; including a score–MW interaction produces an even stronger small-molecule correction.** This is a useful explanation of an important adaptation effect, not evidence that neural fine-tuning uniquely learned the biology separating binding from activation.

[Explore the molecular-weight control HTML](../results/mw-controls/index.html) · [Inspect Nesso's architecture and trainable layers](../index.html) · [Full adaptation results and compound plots](../results/index.html)

## The low-MW finding

The native model strongly overpredicts activation potency for the small-molecule subset. Both head-only adaptation and upstream LoRA correct much of this error. Molecular weight is sufficient to reproduce this conspicuous pattern when combined with the native score in a small fitted model.

The MW analysis includes all 3,344 development compounds: 237 below 215 Da and 3,107 above 215 Da, with none exactly 215 Da. The threshold describes the displayed subgroup; the simple models fit all compounds and use continuous MW, not a threshold rule. It was a retrospective diagnostic, not a prespecified model-selection subgroup.

### Error among compounds below 215 Da

Weighted mean absolute error in pEC50 units; lower is better. Each column compares predictions on the same 237 compounds.

| Prediction system | 100 labels | 500 labels |
| --- | ---: | ---: |
| Native Nesso, unchanged | 1.834 | 1.834 |
| Native score, fitted linear mapping | 1.520 | 1.570 |
| Native score + MW | 0.951 | 1.163 |
| Native score + MW + interaction | 0.773 | 0.863 |
| Fine-tuned continuous heads | 1.165 | 1.320 |
| Fine-tuned heads + upstream LoRA | 1.038 | 1.179 |

The additive MW model recapitulates the small-molecule correction. The interaction model exceeds the LoRA correction in every fold at both label budgets. It has only four fitted coefficients:

    predicted pEC50 = intercept + a × native score + b × MW
                     + c × native score × MW

The actual fitting uses training-only centered/scaled inputs. Native score means the unchanged continuous Nesso output converted to the pIC50-equivalent numerical orientation. All coefficients are fitted to cellular PXR pEC50 with inverse-assay-SE weighted least squares. This is a fitted activation mapping, not an unchanged native binding prediction. The interaction allows the association between the native score and activation to vary with molecular weight.

### What this does—and does not—explain

- **A core visible improvement is calibration of small molecules, not strong ranking within that subset.** The interaction model's small-molecule Spearman correlations are only 0.112 and 0.083 at the two budgets.
- **MW alone is not the whole explanation.** Adding native score to MW improves full-cohort weighted error in every fold at both budgets. The sources contain complementary predictive information.
- **This does not account for every neural improvement.** At 500 labels, LoRA has lower weighted error than the interaction model among larger compounds: 0.544 versus 0.575. Descriptor LightGBM remains stronger than either on that subset.
- **Small-molecule correction is not synonymous with most of the aggregate gain.** The separate potency decomposition attributes most of LoRA's net improvement over head-only adaptation to compounds below pEC50 4. Low MW and low potency overlap strongly here; they are not interchangeable groups, and the data do not establish a causal MW mechanism.

[Complete MW methods, subgroup metrics and replay checks](../results/mw-controls/README.md)

## What was fine-tuned?

Nesso-1 predicts distograms and binding-related outputs; it does not generate cofolded coordinates. Its protein-conditioned representations are useful inputs for an assay-specific predictor, but a binding score is not itself a cellular activation prediction.

    Protein sequence + prepared ligand
      → released Nesso input processing and main trunk [frozen]
      → two affinity modules
      → two 384-dimensional representations
      → continuous prediction heads
      → fitted cellular PXR pEC50 prediction

The study distinguishes three adaptation boundaries:

| Approach | What uses PXR labels? | What stays frozen? |
| --- | --- | --- |
| Frozen-vector probe | A new ridge or small neural readout | All native Nesso weights and cached representations |
| Head-only fine-tuning | Both released continuous 384 → 384 → 384 → 1 MLPs | Everything upstream of the cached vectors |
| Head + upstream LoRA | Both continuous heads plus rank-4 adapters in each affinity member's ESM-feature projections and first Pairformer transition | External ESM encoder, main trunk, original affinity base weights and remaining parameters |

The upstream adapters target `esm_proj.1`, `esm_proj.3`, and `pairformer_stack.layers.0.transition_z.fc1` in each affinity member. Updating an ESM-feature projection is not updating the ESM encoder. This is genuine adaptation before the pooled affinity vector, but not full-Nesso fine-tuning.

The latest comparison uses the same pretrained heads, acquired compounds, initialization and fixed optimizer schedule for head-only and LoRA. Both run for 40 full-fitting-set optimizer updates, without tuning or early stopping. It isolates the effect of adding these adapters under one recipe; it does not establish the best attainable performance of either model family.

[Interactive architecture and source inventory](../index.html)

## Did it improve prediction?

Yes, compared with unchanged Nesso. LoRA also improves its matched head-only recipe. At 500 labels, however, both frozen-vector ridge and ligand-descriptor LightGBM have lower weighted error than the neural recipes.

### Full-cohort weighted error

Each entry pools one held-out prediction per compound across the five reused development folds, on the same 3,344 compounds.

| Prediction system | 100 labels | 500 labels |
| --- | ---: | ---: |
| Native Nesso, unchanged | 0.6272 | 0.6272 |
| Fine-tuned continuous heads | 0.5894 | 0.5773 |
| Fine-tuned heads + upstream LoRA | 0.5831 | 0.5610 |
| Ridge on frozen Nesso vectors | 0.6186 | 0.5345 |
| Ligand-descriptor LightGBM | 0.6103 | 0.5006 |
| Native score + MW + interaction | 0.5868 | 0.5828 |

At 100 labels, LoRA leads this panel on weighted MAE, but descriptor LightGBM has lower unweighted MAE and slightly better ranking. At 500 labels, descriptor LightGBM leads on all three measures. There is no all-budget, all-metric Nesso win.

LoRA's gain over head-only adaptation is modest and uneven. Error improves mainly below pEC50 4; among compounds at or above 4, weighted error slightly worsens. Both neural methods still underpredict every compound in the 56-compound pEC50-at-least-6 subset at 500 labels. More spread in the predictions is not a solution to the active tail.

[Complete results, paired uncertainty and potency decomposition](../results/RESULTS.md) · [Compound-level predictions](../results/heldout_predictions.csv)

## Subsequent experiments change the overall conclusion

The adaptation/MW panel above remains unchanged. A later, locked-procedure CPU comparison combined descriptor LightGBM with a ridge readout of frozen Nesso representations. Across all five reused development folds, weighted MAE improved from 0.5006 to 0.4868 and pooled Spearman from 0.6569 to 0.6864. The benefit occurred mainly among compounds with pEC50 ≥4; lower-potency error worsened. This is complementary information from frozen representations, not evidence of extra LoRA benefit. It remains retrospective and conditional on one acquired-subset draw.

A four-CYP direct-inhibition extension found a small additional MW benefit for CYP2C9 and CYP3A4, but no consistent benefit from the interaction and no uniform small-molecule bias direction. Those were scalar mappings with unweighted MAE, not neural fine-tuning or a replication of PXR's assay-SE-weighted protocol.

Finally, the proposed explanation that small PXR molecules bind without activating was tested against Emax and assay uncertainty. The small molecules have higher pEC50 SE (median 0.558 versus 0.141), but higher normalized Emax (1.240 versus 0.977), while baseline-relative response is lower. All 237 small compounds pass the historical normalized-Emax qualification. The data support an uncertainty association; they do not establish binding without activation. Raw curves and reconstructable positive-control normalization are missing.

These studies and their source evidence are linked from the main experimental page. A favorable native score is not proof of binding, and poorly determined EC50 is not equivalent to absent efficacy.

## How strong is the evidence?

The latest matched comparison evaluates 100- and 500-label budgets. These include interval calibration: 80 fitting + 20 calibration compounds, or 400 + 100. The fitting/calibration pool excludes compounds with Morgan Tanimoto similarity at least 0.35 to the outer test set. The main score is absolute pEC50 error weighted by `1 / max(reported assay SE, 0.1)`; weights are identical across models on the same evaluated compounds.

These are retrospective results on reused development folds, not a new untouched test. The upstream comparison uses one acquired-subset draw and one initialization. Reported group-bootstrap intervals condition on the saved fitted systems and do not measure retraining variability. The MW controls were proposed after inspecting the neural results, although their coefficients were fitted without outer-test labels.

The broader project also tested native continuous and binary readouts, low-data learning curves, stable frozen-vector readouts, stronger ligand-only baselines, expanded head learning-rate/loss settings, and final pair pooling. Those studies have different budgets and evaluation protocols; their metrics should not be merged into this table. The root project README links the historical records.

## What can a reader take away?

1. **Separate binding outputs from the target assay.** Numerical conversion does not turn binding affinity into cellular activation potency.
2. **Test simple controls before assigning a biological explanation to fine-tuning.** A scalar molecular property recapitulated the most conspicuous small-molecule correction.
3. **Test frozen representations before increasing trainable capacity.** A simple readout can outperform a more expensive adaptation recipe.
4. **Name the trained layers.** Cached heads, affinity-module LoRA and full-trunk training answer different questions.
5. **Distinguish the standalone and hybrid results.** Descriptor LightGBM leads the original N500 standalone panel; the later frozen-feature hybrid improves it consistently within the reused development evaluation. Neither establishes prospective superiority.
6. **Test the explanation as well as the prediction.** The efficacy diagnostic supports an uncertainty association but does not support a simple low-normalized-efficacy explanation for small molecules.

This is a case study of Nesso adaptation for one PXR cellular pEC50 assay. It does not establish a clinical activation-liability model, general fine-tuning advice across targets, or a causal protein-specific mechanism.

## Reproduce the summary

From the project root, using Python with `markdown-it-py` installed:

    python scripts/render_case_study.py

This regenerates this HTML from the editable README and independently checks its displayed tables against saved predictions. It performs no model training or inference. The existing MW HTML retains its original figures, model states and verification downloads. Full training remains in the linked experiment records; the compact summary does not claim a clean-environment end-to-end training reproduction.

[Editable case study](README.md) · [Summary verification](verification.json)
