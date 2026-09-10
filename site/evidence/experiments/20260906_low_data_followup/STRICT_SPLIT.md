# Strict chemical-separation follow-up

Retrospective evaluation on the same development compounds and original chemical-cluster TEST folds. This changes training support; it is **not untouched, temporal, or prospective validation**. The approved protocol is `protocol_strict.json`, interpreted with `IMPLEMENTATION_PLAN.md`. No separate `protocol/STRICT_SPLIT.md` existed at recovery.

## Frozen preparation and measured feasibility

Existing preparation was preserved, not regenerated into a new partition. Both actual `strict_split.py prepare` (existing-packet verification) and `strict_split.py verify` passed structure-only semantic and byte replay. Original prepared input hashes and Morgan bits are checked by these commands. No change was made to `strict_split.py` or its frozen source snapshot.

Morgan radius 2, 2,048-bit Tanimoto is the explicitly approved separation stress test, not a claim of optimal chemical coverage. Purge every original outer-pool compound whose similarity is **>=0.35 to any test member**, including all future calibration candidates. Compute all pool/test pairs and independently recompute test-to-retained-pool nearest neighbors. Test row identities remain fixed.

| Fold | Test rows | Original pool | Retained | Removed | Retention | Retained/test maximum |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 669 | 2675 | 2364 | 311 | 88.374% | 0.3493975903614458 |
| 1 | 669 | 2675 | 2365 | 310 | 88.411% | 0.3493975903614458 |
| 2 | 669 | 2675 | 2327 | 348 | 86.991% | 0.3493975903614458 |
| 3 | 668 | 2676 | 2359 | 317 | 88.154% | 0.3493975903614458 |
| 4 | 669 | 2675 | 2345 | 330 | 87.664% | 0.3493975903614458 |

All 3,344 identities remain in their test folds. Retained-pool/test identity and canonical-SMILES overlap are zero in every fold. All five folds support N25/50/100/250/500 and eligible-full; no budget is excluded. The packet contains 300 subset tasks and 163,850 acquired-row records. No subset has fewer than three fit chemical groups.

Filter the exact original seeded full draw orders by eligibility. Each acquired set is its filtered N-prefix; its last `ceil(.2*N)` rows are calibration, the rest fitting/inner selection. Acquired prefixes nest; fit-only subsets need not nest as calibration roles move. Preserve explicit removed identities/reasons, draw ranks, test-neighbor distributions and scaffold/group coverage in the prepared CSVs and `audit.json`. Preparation parses structure/identity/split columns only, not outcome columns or scores.

## Fixed nine-method panel

`native_continuous`, `weighted_median`, `native_affine`, `head_pretrained`, `head_random`, `morgan_ridge`, `descriptor_lightgbm_rdkit_mordred`, `repr_ridge_unscaled`, `repr_ridge_stable`.

Readouts use their own frozen four-alpha protocol, not the original three-alpha grid. The descriptor comparator is the separately named feasible RDKit/Mordred/Morgan recipe documented in `BASELINES.md`, not an exact historical restoration. Its descriptor schema identity is passed into the saved model/audit. Runner integration filters descriptor cache identity columns to `row_index`, `record_id`, `canonical_smiles`: the B API rejects `feature_row` despite its identity-only semantics. The original preparation's richer identity table remains unchanged. Descriptor arrays go only to baseline factories; original/readout factories reject extraneous keys.

All fitting/selection receives only fit labels, with fit-only chemical groups. Calibration and test predictions and replay state are persisted before their outcomes are scored. Headline inverse-assay-SE MAE uses fixed `1/max(SE,.10)`, with raw MAE, potency bias, prediction tails and declared SE-floor sensitivities retained. Calibration labels set absolute-residual conformal radii only. Small calibration sets can legitimately produce infinite radii; no distribution-free guarantee is asserted under chemical shift. Point predictions are neither clipped nor dropped.

## Execution and verification

Repository root; interpreter from the validated `cheminf` scientific environment (`python`). Set `PYTHONPATH=src:.`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.

```bash
python experiments/20260906_low_data_followup/code/strict_split.py prepare
python experiments/20260906_low_data_followup/code/strict_split.py verify
python experiments/20260906_low_data_followup/code/run_strict.py dependencies
python experiments/20260906_low_data_followup/code/run_strict.py preflight
python experiments/20260906_low_data_followup/code/run_strict.py smoke
python experiments/20260906_low_data_followup/code/run_strict.py run
python experiments/20260906_low_data_followup/code/run_strict.py verify --replay
```

Preflight checks A/B modules, protocols, docs, tests, descriptor identity/schema/file/array hashes, preparation replay and passing combined tests. At recovery the combined suite passed **55 tests**, including **10 strict regression tests**: exact inclusive threshold/any-member purge, deterministic filtered prefixes, calibration roles, unsupported budgets, original-order corruption, preparation output/source corruption, complete task cardinality, completion corruption, frozen task drift, feature/provider routing, literal-ID descriptor adapter and finite/infinite interval radii.

The actual N25/fold0/draw0 smoke completed **all nine methods**, with saved-state exact same-batch seed replay and metric/CSV replay passing: nine valid, zero failed, zero invalid. Its full-panel verifier correctly reports 2,691 pending rather than treating a smoke as full completion. `strict_split/smoke/smoke_gate.json` and `verification.json` are the evidence. These nine genuine completed fits can seed `panel/` by a byte-preserving directory copy before first launch; they are not new independent results or fabricated fits.

The full panel is 2,700 systems: five folds × ten draws × six supported sizes × nine methods. Service handle: `nesso-pxr-followup-strict.service`. Two numerical threads and CPUQuota=200%; no elapsed-time or task-count cap. The runner has an exclusive file lock, atomic attempts/completions, preserved failures, hash-checked resume and source/input/runtime snapshots. A source change fails closed; do not edit bound code/tests/protocols while it is running. Re-run preflight only before a new versioned run, because the gate bytes are bound by existing runs.

Artifact root: `artifacts/experiments/low_data_followup_20260906/strict_split/`. `dependency_preflight.log` and `dependency_gate.json` record combined test evidence. `panel/status.json` records live task/counts; `panel/failures/` preserves technical failures; final `panel/verification.json` reconciles all expected artifacts and replays saved models. A service launch or exit is not full scientific completion. The parent must report final completion only when all 2,700 are valid with no invalid/pending/failed cells.
