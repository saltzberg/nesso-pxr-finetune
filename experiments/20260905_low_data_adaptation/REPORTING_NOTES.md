# Low-data reporting notes

## Entry point

From the repository root with the scientific environment active:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 python scripts/report_low_data.py \
  --run-dir artifacts/experiments/low_data_20260905 \
  --report-dir /tmp/nesso-low-data-zero-report
PYTHONPATH=src:. OMP_NUM_THREADS=2 python -m pytest tests/test_low_data_report.py -q
```

`build_report(run_dir, report_dir)` returns the summary dictionary. The report
must be outside the immutable run directory. It never launches fitting, loads
pickled model state, accesses the network, or changes root-site navigation.
Parent owns serving and matrix execution. A prepared-only run is supported:
its frozen protocol defines pending tasks, without invented performance data.

## Evidence and outputs

- Full protocol Cartesian product is reconciled against task_manifest.json when
  present. Unrecognized task directories are listed separately, not counted.
- Complete markers must include predictions, metrics, label-access and raw seed
  artifacts. Every listed artifact is read and SHA256 checked, including saved
  model state when listed; an empty hash mapping is not successful completion.
- Preparation artifact/source hashes and run source/input bindings are checked
  against actual files. Source drift excludes completed evidence rather than
  silently mixing implementations. Finish code integration before runner binding.
- Held-out identities, labels, errors, weights, exact acquired budgets,
  fit/calibration/test disjointness and counts are checked. Scores and interval
  coverage/width are recomputed. Seed means/SDs are checked against predictions.
- Failed verification counts as failed. Missing/in-flight output remains pending.
  No source claim of successful completion overrides direct artifact checks.
- HTML, five PNG/SVG figure families, aggregate/task metrics, exact task states,
  matched effects, novelty/prediction-bin CSVs, strict JSON summary, recorded model
  audits, provenance, protocol/split audit and output-hash manifest are emitted.
  JSON uses null for nonfinite numeric values; CSV retains infinite interval width
  and task records give explicit `unsupported_unbounded` interval status.

## Interpretation

Primary weighted MAE uses normalized inverse assay SE with the protocol floor;
0.05/0.20 score sensitivities are available in task metrics. Unique compound
counts and unique-weight Kish effective N are separate from repeated prediction
observations. Paired effects require identical fold/draw/budget/test identities;
plotted error bars are observed min–max variability, **not confidence intervals**.
No winner is declared, including when all tasks finish: inferential uncertainty
and a practical improvement margin remain separate scientific requirements.

N includes reserved calibration labels; full reference uses n_train=-1 and is
exported separately. At N=25 with five calibration rows, 90% split-conformal
intervals are legitimately unbounded. Calibration does not guarantee coverage
under chemical shift. Seed spread is not a predictive interval.

Model flow is audited against current source: representation MLP is
768→128→1, scalar ridge has nine reconstructed scalar inputs, representation
ridge concatenates paired 384D vectors, pretrained continuous heads and dual
continuous/binary heads remain distinct. Head LoRA updates final readouts, not
upstream representations. Parameter/time classifications are recorded in
model_audits.json, not guessed from method names. Morgan LightGBM is Morgan-only.

Singleton fractions come dynamically from the current split audit; the old
historical percentage is explicitly a different grouping. Butina can retain
cross-group similarities above its cutoff. Novelty figures use fit-set similarity,
not the full-outer-pool diagnostic. Largest observed non-full budget diagnostics
are not necessarily complete panels; all budgets/counts are downloadable.

The separate parent-replay pilot JSON is read and hashed when available. Its
engineering pass is not assay-label training or evidence of sample efficiency.
This report does not rerun the GPU pilot or independently replay fitted models.
Tests use explicitly synthetic temporary fixtures; never publish those as results.
