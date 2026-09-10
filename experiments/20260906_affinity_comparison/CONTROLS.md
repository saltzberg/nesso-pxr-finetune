# Matched CPU controls

These retrospective controls use the unchanged strict-chemical folds 0–4, draw 0, N=100/500. The strict preparation removes pool compounds with Morgan radius-2/2048 Tanimoto **>=0.35 to any test compound**; equality is excluded, with no threshold fallback. The test set is not a fresh holdout.

## Prediction construction

- **Fresh native:** both released-member native FP32 scalars → `6 - (member1 + member2) / 2` in torch FP32 → float64 storage. No point-label fitting, clipping, affine correction, or calibration. This remains an endpoint-mismatched pIC50-equivalent anchor, not a native cellular pEC50 head.
- **Fresh stable ridge:** freshly captured member vectors (384+384) → fit-only weighted mean/floored SD → weighted ridge. Grouped three-fold inner selection uses alpha `[1,100,10000,1000000]`; every inner scaler sees only its inner-training rows. Seed 42, one final ridge model per cell; no historical Nesso vectors or ridge models are reused.
- **Reused descriptor:** exact completed historical `descriptor_lightgbm_rdkit_mordred` cells, unchanged saved reducer/selector/LightGBM state and predictions. No refitting. This is the followup RDKit2D/Mordred2D/Morgan recipe, not restoration of an unavailable earlier feature universe.

There are **30 control evaluation cells**: ten fresh-native, ten fresh-ridge, ten reused-descriptor. Ridge additionally performs its declared inner selection fits. The separately owned trainer has 20 matched neural fits; they are not part of this CPU runner.

Raw predictions and fitted state are durably saved before calibration/test scoring. Within-N calibration suffix labels adapt only absolute-residual 80%/90% intervals. Headline error is inverse-`max(reported SE,0.1)` weighted MAE; raw MAE, Spearman, potency strata and unclipped predictions remain available. No chemical-shift coverage guarantee is asserted. CSV reload uses round-trip float parsing; interval coverage uses `abs(y-pred) <= radius`, not rounded endpoints.

## Exact CLI

Run from the repository root with the existing `cheminf` environment; no installation, GPU or Nesso import is needed. Set `CACHE` to the fully completed native cache directory and `OUT` to a new controls artifact directory. Machine-specific runtime paths belong in local environment/manifest data, not public protocol files.

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$(conda info --base)/envs/cheminf/bin/python"
"$PY" -m pytest -q tests/test_affinity_comparison_controls.py
"$PY" -m experiments.20260906_affinity_comparison.code.controls verify-history
"$PY" -m experiments.20260906_affinity_comparison.code.controls build --cache "$CACHE" --output "$OUT"
"$PY" -m experiments.20260906_affinity_comparison.code.controls run --cache "$CACHE" --output "$OUT"
"$PY" -m experiments.20260906_affinity_comparison.code.controls verify --cache "$CACHE" --output "$OUT"
```

`build` requires all 3344 packets; do not invoke it against the partial production cache. It hashes the actual `stage/manifest.json` schema (`records`, `protocol`, `files`), source snapshots, ordered identities, reused-pilot proofs and all three payloads per record. The native filename is **`native_prediction.pt`**, not a renamed substitute. On reuse, it rechecks cache/feature hashes without deserializing the large torch payloads. `run` resumes only source-bound, hash-valid cells and replays saved models and metrics. `verify` requires every cell to exist; it does not fit missing cells. Trusted-local pickle states must never be accepted from untrusted providers.

## Dependencies and frozen sources

Runtime: Python, NumPy, pandas, torch (CPU sufficient), SciPy, scikit-learn, threadpoolctl, LightGBM, RDKit and mordredcommunity. The existing cheminf environment supplies these. Descriptor schema verification imports RDKit/Mordred but does not recompute descriptors.

`source_paths()` is the authoritative controls source allowlist: controls.py, controls_protocol.json, controls tests; old readouts/baselines/run_strict/strict_split and their three protocols; low_data_models, low_data_contract (metrics), low_data_adapter, chemistry, labels, protocol, splits and package initializer; original input labels; all strict preparation outputs and manifests; cache.py, cache_protocol.json and staged cache manifest. The stage manifest binds checkpoint/image/source bytes. Public image references are digest-only. Mutable parent reports and the new trainer are not bound by controls. Historical source snapshots are verified against their original run binding; only actually required helpers/protocols also require matching live bytes.

Required historical artifacts: `artifacts/experiments/low_data_followup_20260906/strict_split` (preparation plus `panel/run_bindings.json`, source snapshots and the ten descriptor task directories), original `artifacts/experiments/low_data_20260905_cut035_run2/prepared/inputs.csv`, and `artifacts/experiments/low_data_followup_20260906/baselines/features/{descriptors.npz,descriptor_metadata.json}`. Saved models, all output hashes, exact role/ID/order membership, labels, weights, per-seed predictions, point metrics and interval endpoints/metrics are replayed without fitting.

## Verification evidence

The real CPU `verify-history` CLI returned `verified_descriptor_cells: 10`, `new_fits: 0`, `read_only: true`, with all ten exact task IDs and their completion-marker SHA256 digests in `verified_tasks`. It covers the Cartesian product of five folds and N=100/500 at draw 0. Focused controls tests passed **14/14**, including the read-only real historical replay test and temporary synthetic feature/ridge/corruption/leakage/resume fixtures. Temporary synthetic tests are engineering fixture evidence only, never published assay performance. The actual partial production cache passed stage/source-snapshot/identity checks and failed closed at a missing capture manifest, as expected; no Nesso module was imported. That was the pre-launch state. Update 2026-09-07: all 30 production control evaluation cells completed and their saved predictions, roles, output hashes and metrics were independently audited. See RESULTS.md for the completed comparison; no controls were refitted during the analysis.
