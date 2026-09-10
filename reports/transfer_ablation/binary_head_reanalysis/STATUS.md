# Status log

## 2026-08-13 — audit started

- Scope: local, CPU-only reanalysis; no publication or external writes.
- Resource cap: at most four CPU threads/workers; one experiment family at a time.
- Official semantics confirmed: `affinity_pred_value` is `log10(IC50 / uM)`; `affinity_probability_binary` is the model's separate binder probability.
- Architecture confirmed in pinned Nesso 1.0.0: each member maps its 384D `affinity_repr` through independent continuous and binary nonlinear branches; the released ensemble averages member probabilities and converts that mean back to an ensemble logit.
- Checkpoint SHA-256: `9928a8a824d147d665e76656804af1cd91c86731516d0b561f9fd1c91ee45622`.
- Cached labeled feature rows: 4,134 = 3,344 development + 790 validation. The primary/common cohorts are 3,072 development + 714 validation; the other 272 + 76 rows are lower-Emax/excluded-initial-Emax supporting cohorts.
- Challenge feature rows: 513.
- Continuous replay provenance already passes: per-member replay maximum absolute error `4.768e-7`; the all-pairs cached representation exactly reproduces the historical representation.

Commands and runtimes will be appended after each completed experiment family.

## 2026-08-13 — completed experiment families

- Binary reconstruction: 0.4 seconds inside the pinned Nesso 1.0.0 image; four CPU threads; continuous replay maximum absolute error `4.77e-7`.
- Dual-branch activity adapter: 74.9 seconds; four CPU threads; seeds 42/43/44; fixed 24-epoch budget; no GPU visible.
- Nested probes, 2,000-replicate family bootstraps, pair-pooling extension, and figures: 169.0 seconds; four CPU threads.
- Focused tests: 6 passed.
- Key result: the binary branch adds measurable activity-proxy information but does not reverse the comparison with the 384D representation or LightGBM.
- External state: no publication, push, or AetherArk modification.
