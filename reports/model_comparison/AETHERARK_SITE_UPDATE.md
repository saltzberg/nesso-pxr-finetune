# AetherArk site update map

## How the current site differs

The existing Nesso-only measurements remain valid:

- Public challenge: fine-tuned Nesso MAE 0.601 and Spearman 0.657 versus
  frozen Nesso MAE 0.672 and Spearman 0.451.
- Earlier development repeats: MAE 0.629--0.635 and Spearman 0.624--0.631.
  The new nested three-seed estimate is essentially unchanged at 0.633 and
  0.625.
- Earlier full 790-compound lockbox: fine-tuned Nesso MAE 0.597 and Spearman
  0.581 versus frozen Nesso 0.815 and 0.312. The new 714-compound table is the
  Emax-qualified subset shared exactly with the historical 2D model, not a
  correction to the 790-compound Nesso-only result.

The scientific interpretation is materially different because the current
site omits:

1. The matched 2D LightGBM comparator, which is stronger than fine-tuned Nesso
   in nested development, the historical shared lockbox, and the public
   challenge set.
2. The training-fold mean control and RAE.
3. The fully nested Nesso configuration/epoch selection and nested 2D
   supervised feature selection/stopping.
4. Scaffold/Butina-cluster bootstrap confidence intervals.
5. Residual agreement, calibration, assay/similarity/potency strata,
   active-tail failure, and activity-cliff stress tests.
6. The evidence-status statement that challenge labels were public before this
   analysis, making it retrospective rather than a temporal blind test.

Two statements are directly stale: `data.html` says challenge labels are not
available, and `modeling.html` says challenge prediction is the next step.

## Smallest accurate revision

### `index.html`

- Change the headline to: **Fine-tuning Nesso helps, but a 2D model remains
  stronger.**
- Retain the Nesso challenge improvement, but add the matched 2D result.
- Add a prominent evidence-status callout: **The challenge labels were public
  before this analysis. These results are retrospective external-dataset
  evidence, not a temporal blind test.**
- Summarize shared errors and the active-tail limitation.

### `data.html`

- Keep the 3,344/790/513 counts.
- Split 3,344 development rows into 3,072 Emax-qualified and 272 additionally
  curated lower-Emax rows.
- Explain that the shared historical 2D lockbox comparison covers 714 of the
  790 rows because the existing 2D model was Emax-qualified.
- Replace “challenge labels unavailable/final test” with “public-label
  retrospective external comparison.”

### `modeling.html`

- Preserve the cached Nesso-head architecture and original final-model recipe.
- Distinguish the historical pooled screen/final model from the new nested
  development estimator.
- Add the deterministic LightGBM recipe: 14,325 descriptor/fingerprint inputs,
  inner-only top-1,000 gain selection, inner-only stopping-round selection,
  fixed outer refit, inverse-SE times curation weighting.
- State that the legacy label-free 2D variance/correlation prefilter is fixed,
  rather than refit inside every new outer fold.
- Remove “challenge predictions are the next step.”

### `training-results.html`

- Relabel the existing loss plot as a historical Nesso training diagnostic; it
  depicts the original five-fold/three-seed confirmation, not the new nested
  model comparison.
- Lead with the matched nested table:

| model | n | MAE | RAE | Spearman |
|---|---:|---:|---:|---:|
| training-fold mean | 3,344 | 0.925 | 1.002 | -0.075 |
| frozen Nesso-1 | 3,344 | 0.915 | 0.991 | 0.385 |
| fine-tuned Nesso-1 | 3,344 | 0.633 | 0.686 | 0.625 |
| nested 2D LightGBM | 3,344 | 0.516 | 0.559 | 0.730 |

- Add compact shared-lockbox and retrospective-challenge tables:

| evaluation | model | n | MAE | Spearman |
|---|---|---:|---:|---:|
| historical shared lockbox | fine-tuned Nesso | 714 | 0.601 | 0.602 |
| historical shared lockbox | established 2D | 714 | 0.478 | 0.741 |
| retrospective public challenge | fine-tuned Nesso | 513 | 0.601 | 0.657 |
| retrospective public challenge | established 2D challenge model | 513 | 0.516 | 0.748 |

- Report paired differences with cluster-bootstrap intervals and one short
  error-complementarity table.
- End with the active-tail warning: neither final model predicts any of the
  eight shared-lockbox or 31 challenge compounds above pEC50 6.

## Implementation effort

This is an editorial update to four static HTML files. The existing CSS already
supports all required callouts, tables, lists, and responsive behavior. The
training-history JSON and chart JavaScript can remain unchanged. Existing site
tests already validate no-index tags, local links, and completed chart data.
No new site framework, dependency, or client-side analysis is needed.
