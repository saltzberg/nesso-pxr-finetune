# 4 · What the correction does—and does not—mean

A tempting explanation for small-molecule overprediction is “they bind but do not activate.” The saved data allow an efficacy and uncertainty diagnostic, but not a direct test of binding or cellular mechanism. The native affinity-equivalent score is itself a model output, not an experimental binding measurement.

## Efficacy does not support a simple non-activation story

The diagnostic joins molecular weight, native scores, observed pEC50 and reported assay summary fields. Raw baseline-relative log2 fold-change Emax and positive-control-normalized Emax are different quantities and must not be interchanged.

| Assay summary (median) | MW <215 Da (n=237) | MW ≥215 Da (n=3,107) |
|---|---:|---:|
| Positive-control-normalized Emax | 1.240 | 0.977 |
| Baseline-relative Emax (log2 fold change) | 2.170 | 2.530 |
| pEC50 standard error | 0.558 | 0.141 |

Small molecules have higher normalized efficacy, lower baseline-relative response, and less precise fitted potency. These are subgroup medians, not a causal explanation.

All **237** small molecules meet the historical normalized-Emax qualification of ≥0.6. The association with larger standard errors matters for interpreting the weighted evaluation, but does not justify treating lower-potency measurements as meaningless. Raw curves and the normalization denominator are unavailable, so the diagnostic cannot resolve the assay-scale discrepancy or prove that adapters learned efficacy.

[Full efficacy diagnostic and source summaries](../../experiments/efficacy-uncertainty/index.html)

## Supporting side branch: direct CYP inhibition

The four-CYP study asks whether scalar native-score/MW controls transfer to direct enzyme inhibition, where the endpoint differs from cellular PXR activation. It is **not a CYP tuning experiment** and should not be read as a second neural benchmark.

Additive MW helps CYP2C9 and CYP3A4, while the interaction is not consistently better. Low-MW groups are sparse; CYP1A2 even has the opposite subgroup direction. This is mixed support for a simple size-related calibration effect, not a universal low-MW mechanism.

CYP errors are unweighted: supplied standard-deviation/confidence-interval fields do not establish per-row sampling standard errors. Importing PXR precision weights would create a false comparability. [Supporting CYP report and evidence →](../../experiments/cyp-transfer/index.html)

## Conclusion: retain the simple reference; test the incremental candidate

Three observations survive the controls. Tuning improves native transfer but does not beat descriptors at N500. Much of the visible small-molecule correction is reproducible by a simple score + MW fit. Frozen Nesso features nevertheless add a modest improvement in all five reused development folds when blended with descriptors, with a real lower-potency penalty.

Retain descriptor LightGBM as the standalone reference. Treat the locked frozen-feature blend as a candidate whose value depends on the intended error objective—not as an automatically promoted model. The next decisive study should freeze its selection rule, use independent label draws/seeds, and evaluate genuinely untouched data. More retrospective selection cannot supply that confirmation.

[Return to the visual overview](../../index.html) · [Complete supporting archive](../../index.html#experiments)
