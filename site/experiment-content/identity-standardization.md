# Chemical splits: Random, Butina and strict separation

## Question

How were compounds separated when the PXR models were tested? **Random compound holdout and Butina cluster holdout were both executed in the low-data comparison.** Later strict experiments kept the Butina test compounds but purged neighboring training compounds. The earlier head-versus-descriptor comparison used a different, scaffold-aware Butina split.

## Design and fitted/frozen flow

### Dataset and shared fingerprint

The OpenADMET dose-response TRAIN table has 4,139 records; curation retained **4,134 original molecules**, split into **3,344 development / 790 lockbox**. The 513 public challenge records are separate. Earlier efficacy qualification selected 3,072 development / 714 lockbox rows; the all-curated nested comparison and low-data studies use all 3,344 development compounds. A historical 714-row score and later all-790 score therefore evaluate different cohorts.

These splits keep original compound identities together, not independently sampled prepared protonation states. Low-data preparation explicitly requires unique canonical SMILES: one input row per compound. **No pEC50, activity-class or uncertainty stratification** is used.

Both Butina generations use standard **Morgan radius 2, 2,048-bit binary fingerprints**, RDKit `GetMorganGenerator` / `GetFingerprint`: atom invariants, not feature-Morgan/FCFP; default bond types included, chirality and count simulation off. Similarity is Tanimoto; **distance = 1 − similarity**. RDKit Butina uses `isDistData=True, reordering=True`. Butina groups centroid neighborhoods—it does not guarantee that all pairs across clusters fall below its similarity cutoff.

### Random and Butina: the completed low-data comparison

The authoritative run is **`low_data_20260905_cut035_run2`**. It constructs new assignments on the 3,344 development compounds; the older published manifest’s `fold` and `cluster_component` are **not** these low-data labels.

- **Random (`random`)** groups by canonical isomeric SMILES, unique in this input. `_balanced_group_folds` orders groups largest first with seeded random tie ordering, then assigns each to a randomly chosen currently smallest fold. Seed **20260905**. It is identity-held-out compound CV, not arbitrary assay-row splitting, sklearn `RepeatedKFold`, or series holdout. Related structures can occur on both sides.
- **Butina (`chemical_cluster`)** uses similarity cutoff **0.35**, hence Butina distance cutoff **0.65**, and **no scaffold union**. Its 1,975 saved `chemical_group` IDs remain indivisible during fold allocation, using the same balancing algorithm and seed **20260905**. There are 1,458 singleton compounds; the largest cluster contains 36. The completed matrix uses this 0.35 preparation, selected by structure-only sensitivity analysis; the earlier 0.50 preparation is retained separately.

Each policy has **one fixed five-fold partition**, approximately 80% outer training pool / 20% TEST. For each fold, **ten draws (0–9)** permute sorted outer-pool indices using `default_rng(SeedSequence([1701, policy_code, fold, draw]))`: policy code 0 for Random, 1 for Butina. Acquired prefixes are **N25, N50, N100, N250, N500**, plus full-pool (`n_train=-1`). The final `ceil(0.20*N)` compounds are **CAL** (`calibration`); the rest **FIT** (`fit`). N25 = 20 FIT + 5 CAL; N500 = 400 + 100. Acquired sets nest, but FIT-only sets need not nest because the calibration suffix moves. Acquisition samples compounds, not whole clusters; FIT/CAL need not be chemically separated.

All 12 methods share these **600 policy × fold × draw × budget cells**. Saved status confirms **7,200 completed model tasks: 3,600 Random and 3,600 Butina**. Neural seeds 42/43/44 repeat fitting, not splitting. Actual inner selection uses **three-fold GroupKFold for both policies**: chemical groups for Butina, unique record IDs for Random. The factory’s shuffled-KFold fallback is not the executed policy. CAL and TEST labels are excluded from model selection.

### Strict follow-up: unchanged TEST, purged pool

Strict splitting retains the **exact Butina TEST assignments**, then removes any pool compound with Tanimoto similarity **≥0.35 to any TEST member**. Equality is removed; every retained pool–TEST pair is **<0.35**. This is pairwise neighbor purging, not another Butina clustering. It matters because the unpurged Butina audit finds cross-fold nearest-neighbor similarity as high as about 0.772, with shared scaffolds remaining.

| Fold | Random TEST | Butina / strict TEST | Butina pool | Strict retained | Purged |
|---|---:|---:|---:|---:|---:|
| 0 | 669 | 669 | 2,675 | 2,364 | 311 |
| 1 | 669 | 669 | 2,675 | 2,365 | 310 |
| 2 | 669 | 669 | 2,675 | 2,327 | 348 |
| 3 | 669 | 668 | 2,676 | 2,359 | 317 |
| 4 | 668 | 669 | 2,675 | 2,345 | 330 |

Each original Butina draw order is filtered to eligible compounds **without reshuffling**. The same budgets plus eligible-full prefixes and 20% CAL suffix produce **300 strict fold × draw × budget cells**. All folds support N500. Retention varies by fold; strict fitting is not a new 80:20 split, nor a guarantee of separation between FIT and CAL.

### Earlier scaffold-aware Butina: a different split

The inherited scaffold lockbox uses `scaffold_group_shuffle`, original-molecule Bemis–Murcko groups, seed **20260505**, requested holdout fraction **0.20**, with realized development/lockbox sizes 3,344 / 790. The packaged code restores the saved roles; it does not contain the original shuffle implementation, so further allocation details cannot be reconstructed from that code alone.

Within development, `assign_scaffold_aware_butina_folds` uses Morgan/Tanimoto **similarity 0.60 → distance 0.40**, then unions Butina clusters sharing a Bemis–Murcko scaffold into indivisible connected components. Acyclic grouping is `ACYCLIC::<canonical_smiles>`, not one empty-scaffold group. Largest-first, smallest-fold allocation uses `default_rng` seed **20260806** for ties. Saved test counts are **669 / 669 / 669 / 669 / 668**. Each outer model trains on the other four folds; nested selection uses those four fold labels as inner validation folds, training on three at a time. Final neural seeds are 42/43/44. There is no separate interval-CAL subset in this historical comparison.

The August repeated learning curves reuse these test folds: ten whole-family training-subset draws at targets 100 and 250, approximately proportional across outer-training folds. Seed formula: `20260813 + draw*10000000 + outer_fold*100000 + target*10 + inner_fold`. These are repeated family-subset fits, **not repeated random outer partitions** or the later exact-N acquisitions.

## Results: which experiments use which split?

| Experiment family | Saved split used |
|---|---|
| Initial head-only/rich-2D, transfer, capacity, loss/LR, binary/dual-head and pair-pooling probes | Initial scaffold-aware Butina 0.60 + Murcko-union folds; no low-data CAL |
| Original 12-method low-data matrix; stable-readout and descriptor follow-ups | Random and Butina 0.35; same five folds, ten draws, five budgets + full; follow-ups reuse assignments and acquisitions |
| Strict follow-up panel | Same Butina TEST; purged pool, ten filtered draws, five budgets + eligible-full |
| Matched upstream head-only/LoRA, matched controls, MW controls | Strict, **all five folds, draw 0, N100/N500**. Head/LoRA use seed 42, fixed recipe without inner tuning |
| Descriptor + frozen-Nesso hybrid | Strict **draw 0, N500**. Pilot fold 0; continuation folds 1–4 with byte-identical pilot reuse. Three grouped meta folds inside FIT plus deeper selection; CAL outcomes unused for mixture fitting |
| Geometry and Emax/uncertainty diagnostics | Saved predictions/identities; no new model-test splits |

The four-CYP scalar-control study has a [separate dataset/protocol](../../evidence/CYP/protocol.md), not an additional PXR split.

## Compound-labeled downloads

Joined CSVs include **Nesso `record_id`, original OADMET `original_id`, `OCNT_ID`, canonical/raw SMILES and saved chemical-group labels**. Original fold and FIT/CAL values and row order are unchanged. Repeated rows denote tasks, not additional compounds.

| Download | Rows | Labels |
|---|---:|---|
| [Initial scaffold holdout CSV](../../evidence/reports/chemical-splits/initial-scaffold-holdout.csv) | 4,134 | Native molecule names, SMILES, scaffold, development/holdout, method/seed/fraction |
| [Initial modeling manifest CSV](../../evidence/reports/chemical-splits/initial-modeling-manifest.csv) | 4,134 | Native compound/state IDs, SMILES, development/lockbox, initial `fold`, `butina_cluster`, `cluster_component`; lockbox folds intentionally empty |
| [Random + Butina TEST-fold assignments](../../evidence/reports/chemical-splits/low-data-outer-assignments.csv) | 6,688 | One compound × policy row; its TEST `outer_fold`, and 0.35 `chemical_group` |
| [Random + Butina acquired FIT/CAL roles](../../evidence/reports/chemical-splits/low-data-acquired-roles.csv) | 360,020 | Task `split`, `outer_fold`, `draw`, `n_train`, native `role`, IDs/SMILES/group |
| [Strict TEST-fold assignments](../../evidence/reports/chemical-splits/strict-outer-assignments.csv) | 3,344 | Unchanged Butina TEST; native `split=chemical_cluster` retained |
| [Strict acquired FIT/CAL roles](../../evidence/reports/chemical-splits/strict-acquired-roles.csv) | 163,850 | All 300 tasks; native `split=strict_chemical`, roles, compound IDs/SMILES/group |
| [Strict pool eligibility and purge audit](../../evidence/reports/chemical-splits/strict-pool-audit.csv) | 13,376 | Pool compound labels, retained/purged flag, maximum TEST similarity, nearest TEST ID and reason |

For a task, select TEST compounds by its fold in the assignment file; select FIT/CAL by matching policy/fold/draw/budget in the acquired file. In acquired rows, `outer_fold` names the **task’s TEST fold**, not the acquired compound’s own test assignment. Other pool compounds are not automatically FIT: they may be unacquired or purged. `n_train=-1` means full eligible outer pool, never all development data.

[Counts, source bindings and SHA-256 manifest](../../evidence/reports/chemical-splits/manifest.json) · [Executable exact-ID-join recipe](../../evidence/scripts/build_chemical_split_downloads.py) · [Methods Markdown](../../experiment-content/identity-standardization.md)

## Implication and limits

Random tests new identities among related chemistry; Butina keeps centroid groups apart; strict purging additionally controls measured pool–TEST proximity. None provides fresh prospective validation. Draws and model seeds share TEST compounds and are not independent replications. The original specification mentions an earlier random five-fold scalar diagnostic, but packaged PXR artifacts do not provide its compound-level assignments; it must not be relabeled as the completed low-data Random comparator.

The downloads are byte-equal native copies or exact joins validated on `row_index`, `record_id`, `feature_row` and SMILES. No splits, models or predictions were rerun. The superseded identity-standardization content remains in the physical revision archive, not on this methods page.
