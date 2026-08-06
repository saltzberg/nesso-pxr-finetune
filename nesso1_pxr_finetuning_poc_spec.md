# Nesso-1 Fine-Tuning for PXR: Proof-of-Concept Specification

Status: proposed proof of concept
Project root: `/home/dan/projects/ADMET-PXR`
Prepared: 2026-08-06
Scope rule: this specification is based only on the local Nesso-1 source, local PXR artifacts, and experiments completed in this workspace. It does not assume unpublished Nesso training code or undocumented training procedures.

## 1. Executive summary

This project will test two claims:

1. The released Nesso-1 checkpoint can be adapted to a new target/assay using a small, auditable PyTorch training harness even though the public repository is inference-only.
2. Adapting Nesso-1 to PXR functional assay data can improve PXR potency ranking or error relative to frozen Nesso-1 and simple calibration baselines.

This is deliberately not a production model. The intended output is a reproducible scientific and engineering proof of concept with honest negative-result handling. The first implementation will freeze Nesso's cofolding trunk and fine-tune only its affinity regression heads. A second stage may fine-tune the two full affinity modules. Full cofolding-trunk fine-tuning is outside the initial scope because the released forward path detaches trunk features before the affinity modules, the repository provides no training losses or optimizer, and the available PXR labels are too small and assay-specific to justify that risk.

The primary supervised data will be the 4,139-compound PXR dose-response (DRC) table, preferably through the current 4,134-row curated/weighted artifact. The primary target is pEC50 with its reported standard error. Single-concentration measurements will be an auxiliary functional-response task, not fabricated pEC50 labels. There are 21,003 single-point observations over 10,870 unique compounds, usually at two concentrations. Because 2,735 DRC structures also occur in the single-point table, all rows derived from the same canonical compound must share the same train/validation/holdout role.

The core model work is PyTorch/Lightning because Nesso is a PyTorch model. A bounded JAX workstream may reproduce a frozen-feature calibration or heteroscedastic head and demonstrate parameter/data interchange. It must not be described as a JAX port of Nesso or as end-to-end Nesso fine-tuning.

## 2. Goals and non-goals

### 2.1 Primary goals

- Demonstrate that pretrained Nesso-1 weights can be loaded, selectively frozen, optimized on PXR labels, saved, reloaded, and used through a reproducible inference path.
- Determine whether PXR adaptation improves compound-level pEC50 MAE and Spearman correlation on compound-held-out data.
- Use reported dose-response uncertainty without allowing extremely small standard errors to dominate training.
- Test whether single-point PXR induction data add useful auxiliary signal without treating induction at one concentration as a direct potency label.
- Produce a portfolio-quality record of data provenance, leakage controls, experiments, profiling, tests, model limitations, and negative findings.

### 2.2 Secondary goals

- Compare head-only adaptation with full affinity-module adaptation.
- Evaluate active-tail and activity-cliff behavior, not only global averages.
- Demonstrate a small, clearly delimited JAX/Optax or JAX-native calibration experiment over frozen Nesso outputs or representations.

### 2.3 Non-goals

- Reproduce Recursion/Valence Labs' original Nesso training procedure.
- Claim a production-quality binding-affinity model.
- Fine-tune the 48-block cofolding trunk in the first proof of concept.
- Treat PXR activation, counter-assay response, single-point induction, EC50, and biochemical binding affinity as interchangeable labels.
- Train on either blinded challenge set.
- Optimize against leaderboard results or use blinded predictions as labels.
- Port Nesso to JAX.

## 3. Known facts and starting evidence

### 3.1 Installed Nesso implementation

The installed source is:

- repository: `/home/dan/containers/protein_structure/nesso`
- upstream: `https://github.com/recursionpharma/nesso.git`
- commit: `f0156e9a22326448684bae09ee96f73415902dcd`
- container image: `local/nesso:1.0.0`
- image ID: `sha256:e3519ab0faa11d098f14f57b628530c21df748ab7c632f5b071754548e058a24`
- wrapper: `/home/dan/containers/bin/nesso`

The checkpoint contains approximately 41.22 million parameters:

| Component | Parameters |
|---|---:|
| Complete Nesso-1 | 41,223,928 |
| Input embedder | 1,053,616 |
| ESM projection module | 888,448 |
| Main 48-block pairformer trunk | 26,873,856 |
| Affinity module 1 | 6,132,996 |
| Affinity module 2 | 6,132,996 |
| Complete affinity heads, combined | 1,316,360 |
| Regression MLPs only, combined | 592,130 |

The public code is explicitly inference-only:

- `Nesso1.configure_optimizers()` raises `NotImplementedError`.
- There is no training CLI, training dataset, supervised loss, or optimizer configuration.
- The inference data module is predict-only and fixes `batch_size=1`.
- The standard CLI globally disables gradients before `Trainer.predict`.
- The forward path calls `.detach()` on the trunk-derived `s_aff`, `z_aff`, and affinity distogram inputs. An affinity loss therefore cannot update the cofolding trunk without changing upstream model code.
- Two-stage refined pocket cropping is gated on `not self.training`. Naively calling `model.train()` changes the computational path relative to standard inference.

These are engineering constraints, not evidence that the weights are intrinsically non-trainable. The modules are ordinary PyTorch modules and local checkpoint load/save methods exist.

### 3.2 Checkpoint and PXR protein artifacts

Use the warm checkpoint already present on this device:

```text
/data1/datasets/openadmet_pxr/cofolding_bulk/home_data_folding/cache/nesso/
  huggingface/models--recursionpharma--nesso/snapshots/
  1896c84c7186c506c7efd79051480809d51098bf/v1.0.0/
    hparams.json
    model.safetensors
```

Checksums:

| Artifact | SHA-256 |
|---|---|
| `hparams.json` | `06aa0c44fcd44eaa2c5c2473bfcf0d9a3af892d867e5279859bb22eb0f6e2d72` |
| `model.safetensors` | `9928a8a824d147d665e76656804af1cd91c86731516d0b561f9fd1c91ee45622` |

Reuse the precomputed PXR LBD ESM embedding:

```text
campaigns/openadmet_pxr/runs/nesso_pxr_unique_topstates_20260806/pxr_esm.safetensors
```

- SHA-256: `cff9ae494cb99a14e3164856db6b025458b662ef4019624176d7bdcfa2c8bfc5`
- protein length: 293 residues
- sequence MD5: `9ee8e7ba7e2644473a2ccd2a9ff22928`
- sequence SHA-256: `fc233cf76f56fcf810c204dddab40a5a5278ab6ca607b65bee3de183bb6e6b5b`

The PXR sequence must be copied from the existing campaign builder or YAML, not manually re-entered in a second source of truth.

### 3.3 Existing zero-shot evidence

The validated Nesso campaign is:

```text
campaigns/openadmet_pxr/runs/nesso_pxr_ec50_actual_challenge_topstates_20260806/
```

Key artifacts:

- `validation_summary.json`: 1,777/1,777 records predicted, zero missing.
- `source_predictions.csv`: all source-level results.
- `ec50_truth_predictions.csv`: 653 exact-relation ChEMBL EC50 compounds joined to Nesso outputs.
- `ec50_performance_summary.json`: direct-scale and diagnostic calibration metrics.

On those 653 external ChEMBL compounds, conversion of Nesso's documented output using `predicted pIC50 = 6 - affinity_pred_value` produced:

- MAE: `0.6081` pChEMBL units;
- Spearman rho: `0.4798`;
- Pearson r: `0.4819`;
- diagnostic random five-fold affine-calibrated MAE: `0.5559`.

This is evidence that frozen Nesso carries PXR-relevant rank signal. It is not the baseline for the proposed fine-tuning claim because it uses an external heterogeneous ChEMBL set, not the challenge-matched DRC assay. The proof of concept must first generate frozen Nesso predictions for every eligible DRC compound under the same splits used for fine-tuning.

## 4. Exact data artifacts

All relative paths in this section are relative to `/home/dan/projects/ADMET-PXR`.

### 4.1 Primary dose-response labels

Raw current project source:

```text
data/pxr-challenge_TRAIN.csv
```

- rows: 4,139;
- unique SMILES: 4,139;
- unique molecule names: 4,139;
- SHA-256: `55f2fa5910352a7516aa8239eebfd6d7cd659bfc57c01f64ae061a7da6e517eb`;
- primary label: `pEC50`;
- label uncertainty: `pEC50_std.error (-log10(molarity))`;
- confidence interval: `pEC50_ci.lower (-log10(molarity))`, `pEC50_ci.upper (-log10(molarity))`;
- auxiliary efficacy: `Emax_estimate (log2FC vs. baseline)` and `Emax.vs.pos.ctrl_estimate (dimensionless)`, each with standard errors and confidence intervals.

Observed ranges in the current file:

- pEC50: 1.61 to 7.5486, median 4.65;
- pEC50 standard error: 0.00325 to 0.739, median 0.15;
- Emax versus positive control: 0.205 to 12.5.

Current curated source recommended for modeling:

```text
data/curated/activity_curation_v2/pxr-challenge_TRAIN_curated_weighted.csv
```

- rows: 4,134 after five named hard exclusions;
- SHA-256: `bcf4d0aea019bf8c7f73cb9f65fae8c2901b7fcfe3a8284f951425bca014e3ed`;
- adds canonical SMILES, applicability-domain annotations, `curation_sample_weight`, exclusion reasons, and review flags;
- curation policy: `docs/activity_curation_v2.md`;
- full audit ledger: `data/curated/activity_curation_v2/pxr-challenge_TRAIN_curation_ledger.csv`.

The curated file is the baseline input. The raw file remains the immutable label audit source. Do not train from both as if they were independent rows.

A dated release copy exists at `data/final_hf_release_20260703_171822Z/pxr-challenge_TRAIN.csv`, but it is not byte-identical to the root file; at minimum, the stereochemical representation of `OADMET-0001228` differs. Because the current curation and split artifacts were built from the root file, this proof of concept must pin the root checksum above. Any later public release must explicitly adjudicate the difference rather than silently changing sources.

### 4.2 Single-concentration auxiliary data

```text
data/pxr-challenge_single_concentration_TRAIN.csv
```

- rows: 21,003;
- unique canonical compounds: 10,870;
- unique `OCNT_ID`: 10,870;
- SHA-256: `e781f8405e919a248b9b832a7cb24e8a615d400ee9dc50beea0d70d8aeaa8c2f`;
- response: `log2_fc_estimate`;
- reported uncertainty: `log2_fc_stderr`;
- significance: `p_value`, `fdr_bh`, `neg_log10_fdr`;
- effect-size context: `cohens_d`, `median_log2_fc`, `n_replicates`;
- conditioning variables: `concentration_M`, `plate_id`, `experiment_name`, `compound_class`.

Important observed properties:

- 8,937 compounds occur at two concentrations and 585 at three concentrations;
- the two dominant concentrations are `8.251e-06 M` (10,747 rows) and `3.3e-05 M` (9,523 rows);
- 20 plates and three experiment names are present;
- median `n_replicates` is 1;
- 2,735 canonical compounds overlap the DRC table.

The existing conservative aggregation/weak-inactive methodology is documented in:

```text
docs/strategies/single_point_inactive_mining.md
```

Its generated audit artifacts are under:

```text
outputs/single_point_inactive_mining/
outputs/single_point_inactive_mining_sim0p5/
```

Single-point data are not pEC50 measurements. They may only enter as an auxiliary response, ordinal/classification task, or carefully documented sample-weighting signal.

### 4.3 Optional counter-assay data

```text
data/pxr-challenge_counter-assay_TRAIN.csv
```

- rows/unique compounds: 2,859;
- SHA-256: `65017e7487b349b7c56f137265e30e0ac218fdd7d8036fc56512eb30c0e01a5e`;
- all 2,859 structures and molecule names overlap the primary DRC table;
- 212 rows have missing pEC50 and Emax values.

This is PXR-null counter-assay evidence and may support an assay-specificity auxiliary task. It is not a second independent potency dataset and must inherit the DRC compound split. It is deferred until the DRC-only and single-point experiments are stable.

### 4.4 Blinded evaluation inputs

Activity challenge:

```text
data/pxr-challenge_TEST_BLINDED.csv
```

- 513 rows/unique compounds;
- SHA-256: `e8a8eed3e81e4fb6c443cba2200a89f855f769541ce786508f7efc2b4edb517c`.

Structure challenge:

```text
data/pxr-challenge_structure_TEST_BLINDED.csv
```

- 78 rows;
- SHA-256: `02ba381d90a5d80fcd30c164fa73e10459dc765d5fef6b9d8dc231aa7faaccce`.

Neither file supplies labels. Neither may influence model selection, early stopping, calibration, or hyperparameters. Predictions may be generated only after a candidate is frozen and must be treated as deployment-distribution diagnostics or challenge submissions.

### 4.5 Prepared ligand states

Baseline one-state-per-ligand mapping:

```text
data/prepared_ligands/pxr_prepared_top_ligands_simple.csv
```

- 15,600 original ligand IDs;
- 12,855 unique prepared SMILES;
- SHA-256: `f31d6f1e016ac14f362872bba41924705e4b0906a629d357377380bbf0f6dabe`;
- includes 4,139 DRC IDs, 10,870 single-point IDs, 513 activity-test IDs, and 78 structure-test IDs;
- records formal charge, estimated population, selected state, selection reason, source set, and source file.

Join rules:

- DRC: `Molecule Name` to `original_ligand_id`;
- single point: `OCNT_ID` to `original_ligand_id`;
- activity blind: `Molecule Name` to `original_ligand_id`;
- structure blind: `structure` to `original_ligand_id`.

All retained states are available at:

```text
data/prepared_ligands/pxr_prepared_ligands_simple.csv
```

- 34,941 state rows;
- 28,830 unique prepared SMILES;
- SHA-256: `57d2ea4eed986ae36e74090c5f75d30c21861829986d38c614bce4030c434649`.

The proof-of-concept baseline must use only the selected top state. Alternate-state augmentation is a later ablation. All states derived from one original ligand must remain in the same split and must not be counted as independent biological labels.

### 4.6 Existing split and audit artifacts

Primary scaffold lockbox:

```text
cache/holdout/activity_curation_v2_scaffold_20pct_seed20260505/holdout_assignments.csv
```

- 4,134 assignments;
- SHA-256: `5681c5d22bbec9fe464ebc8bc7553437629836a2110763a90c2ff34d8bade81f`;
- split unit: original molecule;
- grouping: Bemis-Murcko scaffold;
- seed: 20260505.

Development-only folds for the established Emax >= 0.6 universe:

```text
cache/splits_dev_only/activity_curation_v2/emax_0p6/scaffold_folds.csv
cache/splits_dev_only/activity_curation_v2/emax_0p6/random_folds.csv
cache/splits_dev_only/activity_curation_v2/emax_0p6/cliff_cluster_folds.csv
```

Each contains 3,072 development compounds. Their SHA-256 values are, respectively:

- `d0f0d7a1344fd68bad69a77dc0a1562aa171e22707605d5ff09a1437e4602e7c`;
- `ea0ce194823018c9e7ee1020b76a6c174a596ff283fd2d1b4ab878663565a72d`;
- `8e2aa76a26e4d82f96f582cc6a6aac94564a570a77c34756bd9e3917b3f6747c`.

Primary model selection uses scaffold folds. Random folds are diagnostic. Cliff-cluster folds test local SAR/cliff behavior. The lockbox is opened only for frozen candidate families according to `docs/strategies/internal_holdout_and_feature_reduction_guard.md`.

## 5. Label contracts

### 5.1 Dose-response target

Nesso documents its regression output as:

```text
y_nesso = log10(IC50 / micromolar)
```

The PXR assay supplies pEC50:

```text
pEC50 = -log10(EC50 / molar)
```

For numerical alignment, train against:

```text
target_nesso_scale = 6 - pEC50
predicted_pEC50 = 6 - affinity_pred_value
```

The pEC50 standard error is unchanged in magnitude by this sign/offset transform.

This is a numerical alignment, not a biological equivalence claim. Nesso's pretrained endpoint is IC50-like binding affinity; the PXR endpoint is functional cellular EC50 and depends on efficacy, permeability, receptor context, and assay behavior. Any gain is target/assay adaptation, not proof of improved physical binding free energy.

### 5.2 Uncertainty-aware regression

Implement two prespecified loss variants.

Baseline robust loss:

```text
L_reg = weighted Huber(predicted_nesso_scale, target_nesso_scale)
```

Define an effective uncertainty:

```text
sigma_eff^2 = sigma_assay^2 + tau^2
weight_raw = 1 / sigma_eff^2
weight = normalize(clip(weight_raw, w_min, w_max)) * curation_sample_weight
```

Requirements:

- `tau`, the uncertainty floor, is selected using development folds only;
- start with candidate floors 0.10, 0.15, and 0.20 pEC50 units because observed assay standard errors reach an implausibly dominant minimum of 0.00325;
- cap the total uncertainty weight, for example at the development-set 95th percentile;
- normalize mean training weight to 1 per fold;
- report ablations with and without `curation_sample_weight` and uncertainty weighting;
- never tune the floor or cap on lockbox results.

Optional heteroscedastic extension:

Add a second output per ensemble member for nonnegative model uncertainty `sigma_model`, using `softplus(raw_sigma) + epsilon`, and optimize:

```text
variance_total = sigma_assay^2 + sigma_model^2
L_nll = 0.5 * (error^2 / variance_total + log(variance_total))
```

This separates reported curve-fit uncertainty from learned residual uncertainty only approximately. The assay standard error is not a complete measurement-error model, and one measurement per compound cannot identify all uncertainty components. Report NLL, interval coverage, and sharpness; do not interpret `sigma_model` as calibrated without held-out coverage evidence.

Target jitter sampled from `Normal(pEC50, standard_error)` may be included as an ablation, but it is not the primary uncertainty method because the reported standard error may not represent the full label distribution and repeated sampling can obscure reproducibility.

### 5.3 Emax

Use the established Emax >= 0.6 universe for the first comparable experiment. Then run a full-curated-data ablation with Emax and its uncertainty as an auxiliary target or sample-quality input.

Do not silently drop low-Emax rows and then claim performance on the complete DRC assay. Every experiment must record its Emax rule and resulting row count.

### 5.4 Single-point target

Preferred auxiliary formulation:

- response target: `log2_fc_estimate`;
- known uncertainty: `log2_fc_stderr` with a floor;
- required conditioning input: `log10(concentration_M)`;
- optional nuisance covariates: experiment embedding and plate embedding, used only if they are fit inside each training fold and handled explicitly for unseen validation groups;
- loss: uncertainty-weighted Huber or Gaussian NLL;
- sampling unit: observation, with compound-balanced sampling so compounds with more rows do not dominate.

The auxiliary head must be distinct from the pretrained Nesso binder/non-binder head because PXR transcriptional induction is not equivalent to physical binding classification.

Fallback conservative classification formulation:

- weak inactive: use the existing audited rule `max log2FC <= 0.2`, `min FDR >= 0.1`, and no significant positive induction;
- positive/active thresholds are not yet established locally and must be preregistered as a sensitivity grid before results are examined;
- report AUROC and AUPRC, with prevalence and exact thresholds;
- never convert weak inactive or positive single-point labels into numeric pEC50 values.

### 5.5 Counter-assay target

If added, use a separate specificity/assay-response head. Do not append counter-assay pEC50 rows to the primary regression target. All counter rows inherit the corresponding DRC compound split.

## 6. Leakage and split contract

The split unit is the original biological compound, expanded through canonical-SMILES equivalence.

For every fold:

1. Assign the DRC molecule using the frozen holdout/fold artifact.
2. Build a canonical-SMILES equivalence map across DRC, counter-assay, single-point, and every prepared state.
3. Propagate the DRC role to all matching counter-assay and single-point rows.
4. Keep all concentrations, plates, protonation states, tautomers, conformers, and repeated measurements for one compound in the same role.
5. If one auxiliary compound maps to multiple roles through unexpected aliases, quarantine it and emit a reason-coded collision report.
6. Fit label transforms, uncertainty floors, calibration, nuisance embeddings, and class thresholds on the training partition only.
7. Do not use blinded challenge rows in any fit or selection step.

Required automated audits:

- zero canonical-SMILES overlap between training and validation roles;
- zero original-ID overlap;
- zero prepared-state parent overlap;
- zero Bemis-Murcko scaffold splitting for scaffold folds;
- zero cliff-cluster splitting for cliff-cluster folds;
- counts of DRC/single-point/counter rows by role;
- counts and identities of quarantined aliases;
- explicit check that the 513 blinded activity compounds remain label-free and absent from fitting.

## 7. Proposed model architecture and training modes

### 7.1 Common wrapper

Create an ADMET-PXR-owned `LightningModule` wrapper rather than editing the upstream Nesso checkout in place. It should:

- load the exact checkpoint with `Nesso1.from_pretrained`;
- record upstream commit, checkpoint hashes, hparams, image ID, and PXR sequence hash;
- attach labels and weights to batches without modifying Nesso's serialized inference records;
- implement `training_step`, `validation_step`, `configure_optimizers`, metric accumulation, and delta-checkpoint saving;
- preserve standard five-recycle/refined-pocket inference for validation unless a predeclared cheaper training approximation is under test;
- explicitly control `.train()`/`.eval()` modes so frozen trunk behavior does not change accidentally;
- save only changed head/module weights plus a manifest referencing the immutable base checkpoint;
- support deterministic single-batch overfit and checkpoint round-trip tests.

### 7.2 Mode A: cached regression-head fine-tuning

This is the minimum viable proof of Nesso fine-tuning.

1. Run frozen Nesso in standard inference mode.
2. Extract each affinity module's `affinity_repr` vector immediately before `to_affinity_pred_value`.
3. Cache the two representations, labels, standard errors, IDs, and split roles in a versioned artifact.
4. Train only the two `to_affinity_pred_value` MLPs (592,130 parameters combined).
5. Average the two member predictions exactly as standard Nesso does.

Advantages:

- fast cross-validation;
- small memory footprint;
- no trunk-mode mismatch during repeated epochs;
- clean demonstration of loading and modifying real Nesso parameters;
- easy PyTorch/JAX parity experiments over fixed representations.

Limitations:

- cannot adapt pocket selection or pairwise representations;
- cached features make state/conformer changes a preprocessing decision rather than an augmentation;
- feature extraction support is not exposed by the current CLI and requires a wrapper/hook. `--save_metadata` does not currently persist both affinity representations directly.

### 7.3 Mode B: live regression-head fine-tuning

Run the frozen model per batch and train only the regression MLPs. Keep the parent Nesso module in evaluation mode so refined pocket inference remains enabled, then set only the trainable regression MLPs to training mode. Because Lightning recursively toggles module modes, the wrapper must reassert this policy and test it.

This is slower than cached mode but verifies gradients through the live inference path. It should be a parity check, not the first hyperparameter sweep.

### 7.4 Mode C: full affinity-module fine-tuning

Freeze the input embedder, ESM module, main pairformer trunk, recycling layers, and distogram head. Train `affinity_module` and `affinity_module2` (12.27 million parameters combined).

The existing `.detach()` operations are compatible with this mode: they block trunk gradients but not gradients inside the affinity modules. Use activation checkpointing already configured in the affinity pairformers, batch size 1, gradient accumulation, mixed precision, gradient clipping, and conservative learning rates.

This mode is promoted only if head-only adaptation shows reproducible PXR signal and live-mode memory/runtime are acceptable.

### 7.5 Mode D: trunk fine-tuning

Deferred and out of initial scope. It would require:

- removing or conditionally bypassing the three detach boundaries;
- deciding whether gradients should pass through distogram-derived pocket selection;
- resolving the `not self.training` refined-pocket behavior;
- adding structural/distogram losses or accepting unconstrained representation drift;
- establishing memory and numerical stability on the 16 GB RTX 5070 Ti;
- substantially stronger regularization and more diverse labels.

Completing Mode D is not required to claim the proof of concept succeeded.

## 8. Optimization defaults

Initial head-only defaults, to be changed only within development folds:

- optimizer: AdamW;
- regression-head learning-rate candidates: `1e-5`, `3e-5`, `1e-4`, `3e-4`;
- affinity-module learning-rate candidates: `1e-6`, `3e-6`, `1e-5`;
- weight decay candidates: `0`, `1e-4`, `1e-3`;
- gradient norm clipping: 1.0;
- early stopping: development-fold MAE with Spearman as a required companion metric;
- maximum epochs: small bounded value such as 30 for cached heads and 10 for live affinity modules;
- minimum three seeds for any promoted configuration;
- mixed precision: bf16 where supported;
- batch size: 1 for live Nesso; larger batches allowed for cached representations;
- gradient accumulation selected from measured memory/runtime, not assumed in advance.

Do not conduct a broad unconstrained sweep. The dataset is small enough that repeated hyperparameter search can overfit the validation protocol.

## 9. Experiment matrix

Run experiments in this order.

| ID | Model/data | Purpose | Promotion condition |
|---|---|---|---|
| E0 | Frozen Nesso, all eligible DRC rows | Required challenge-matched zero-shot baseline | Complete predictions and split metrics |
| E1 | Fold-fit affine calibration of E0 | Establish benefit available without neural fine-tuning | Out-of-fold only; no lockbox tuning |
| E2 | Cached regression heads, unweighted Huber | Minimum fine-tuning proof | Gradient/checkpoint tests pass; reproducible CV signal |
| E3 | Cached heads, uncertainty-weighted Huber | Test assay-SE benefit | Beats E2 on paired folds without active-tail regression |
| E4 | Cached heteroscedastic heads | Test predictive intervals | Better NLL/coverage with competitive MAE/rho |
| E5 | Live regression heads | Verify cached/live parity | Similar predictions and no mode-control defect |
| E6 | Full affinity modules, DRC only | Test deeper target adaptation | E3/E5 justify cost; stable memory/runtime |
| E7 | DRC plus continuous dose-conditioned single-point task | Test auxiliary functional signal | Improves primary held-out DRC metrics across seeds |
| E8 | DRC plus conservative single-point classification | Alternative auxiliary formulation | Same evidence standard as E7 |
| E9 | Optional counter-assay auxiliary head | Test assay specificity | Only after E7/E8 are understood |
| J0 | JAX affine/heteroscedastic calibrator on frozen outputs | Framework/parity portfolio artifact | Numerically checked against PyTorch baseline |
| J1 | JAX head over exported affinity representations | Optional interoperability demonstration | Clearly labeled as frozen-feature adaptation |

Stop deeper experiments if E2/E3 do not beat E1 reproducibly. A negative result is a valid outcome and prevents unnecessary full-module tuning.

## 10. Multitask sampling and loss

For E7/E8, use alternating homogeneous batches because DRC and single-point rows have different labels and conditioning variables.

```text
L_total = L_drc + lambda_sp * L_single_point + lambda_reg * L_regularization
```

Requirements:

- select `lambda_sp` only on development folds;
- start with a small grid such as 0.05, 0.1, 0.25;
- ensure each optimizer step or accumulation window includes DRC supervision;
- sample compounds before observations for the single-point task;
- log raw and weighted losses separately;
- report gradient norms from each task into shared trainable layers;
- include a shuffled-label or disconnected-head control to show that any gain is not caused solely by extra optimization steps;
- compare sequential pretraining-then-DRC-fine-tuning against simultaneous multitask training if resources permit.

If only the final DRC regression MLP is trainable, a single-point auxiliary head cannot improve the frozen representation unless it shares trainable layers. E7/E8 therefore require either a shared trainable projection above cached representations or full affinity-module tuning. This dependency must be visible in the architecture diagram and experiment config.

## 11. Evaluation contract

### 11.1 Primary DRC metrics

- MAE in pEC50 units;
- Spearman rho;
- RMSE;
- Pearson r;
- RAE, consistent with existing project reports;
- mean prediction bias and calibration slope/intercept;
- paired per-compound error deltas versus E0 and E1.

Report 95% paired bootstrap confidence intervals where practical. The main claim requires consistent direction across scaffold folds and seeds, not only a favorable pooled number.

### 11.2 Uncertainty metrics

- Gaussian NLL where applicable;
- empirical 50%, 80%, and 95% interval coverage;
- average interval width/sharpness;
- residual magnitude versus predicted uncertainty;
- calibration by assay standard-error quartile.

### 11.3 Stress metrics

- MAE for true pEC50 > 6.0;
- active-tail recall above predeclared potency thresholds;
- cliff sign accuracy and cliff delta MAE;
- performance by nearest-neighbor similarity and applicability-domain tier;
- performance by formal charge/state-change status;
- counter-assay discrepancy strata if that task is enabled.

### 11.4 Single-point metrics

Continuous formulation:

- MAE/RMSE for log2 fold change;
- Spearman within concentration strata;
- NLL/coverage if uncertainty is modeled;
- performance by concentration, experiment, plate, and replicate count.

Classification formulation:

- AUROC and AUPRC;
- prevalence, threshold, sensitivity, specificity, and calibration;
- threshold-sensitivity table rather than a single post-hoc threshold.

### 11.5 Baseline comparisons

Required comparisons on identical rows/splits:

- constant median predictor;
- frozen Nesso E0;
- affine calibration E1;
- existing project 2D baseline where matching prediction artifacts and split roles are available.

Do not compare the new model's scaffold-holdout MAE directly to the existing ChEMBL zero-shot MAE or to a random-CV model and call it an improvement.

## 12. Proof-of-concept success criteria

### 12.1 Engineering success

All are required:

- deterministic tiny-subset overfit test;
- nonzero gradients only for the declared trainable parameters;
- frozen-parameter checksum or equality audit before/after training;
- base plus delta checkpoint reload reproduces predictions within tolerance;
- no split leakage in automated audits;
- complete per-row prediction/provenance files;
- one-command or config-driven reproduction inside a pinned container;
- measured peak GPU memory and examples/second;
- clean failure manifest for invalid conformers or preprocessing errors.

### 12.2 Scientific signal

Treat the following as a useful PoC signal, not a production gate:

- head fine-tuning beats frozen Nesso and fold-fit affine calibration on paired scaffold-fold predictions;
- the improvement is present across at least two of three seeds and is not confined to random CV;
- MAE improves by a practically visible amount, provisionally at least 0.03 pEC50 units, or Spearman improves by at least 0.03, with paired uncertainty reported;
- active-tail and cliff metrics do not show a material compensating regression;
- single-point auxiliary training improves primary DRC validation, not merely its own auxiliary metric.

These provisional thresholds must be frozen before lockbox evaluation. Failure to cross them should be reported as a negative or inconclusive result, not rescued through post-hoc subgroup selection.

## 13. Proposed implementation layout

The implementation phase should create a contained package such as:

```text
nesso_pxr_finetune/
  README.md
  configs/
    e0_zero_shot.yaml
    e2_head_huber.yaml
    e3_head_uncertainty.yaml
    e7_multitask_single_point.yaml
  src/nesso_pxr/
    data.py
    splits.py
    features.py
    module.py
    losses.py
    train.py
    predict.py
    evaluate.py
    checkpoint.py
    audit.py
  tests/
    test_label_transform.py
    test_split_propagation.py
    test_gradient_scope.py
    test_checkpoint_roundtrip.py
    test_tiny_overfit.py
    test_pytorch_jax_parity.py
  artifacts/
    manifests/
    frozen_features/
    experiments/
```

This document does not authorize or implement that package; it specifies what a later implementation should contain.

## 14. Required artifact contract

Every experiment must save:

- immutable config and resolved config;
- git status and commit IDs for ADMET-PXR and Nesso;
- container image ID and package versions;
- source data paths, SHA-256 values, schemas, and row counts;
- base checkpoint and PXR ESM hashes;
- training-row manifest with original ID, prepared-state ID, raw/prepared SMILES, split role, labels, standard errors, weights, and reason codes;
- alias/state propagation and leakage audit;
- trainable/frozen parameter manifest and counts;
- epoch metrics and learning-rate history;
- per-fold and per-seed predictions;
- paired comparison metrics against E0/E1;
- checkpoint delta plus base-checkpoint reference;
- blind predictions only for frozen candidates;
- model card and limitations note;
- GPU memory, wall time, throughput, and failure records.

For compatibility with this repository, also satisfy `docs/model_artifact_contract.md`, including:

- `dev_oof_predictions.csv`;
- `holdout_predictions.csv`;
- `blind_predictions.csv`;
- `training_row_manifest.csv`;
- `reason_coded_curation_ledger.csv`;
- `chemotype_domain_review.csv`;
- `assay_replicate_quality_metadata.csv` when available.

## 15. PyTorch/JAX portfolio plan

### 15.1 PyTorch evidence

The main portfolio narrative should show:

- reading and auditing an unfamiliar Lightning model;
- identifying detach and mode-control boundaries;
- selective parameter freezing and optimizer construction;
- custom uncertainty-aware losses;
- compound-aware multimodal data loading;
- checkpoint deltas and reproducible inference;
- GPU profiling and failure recovery;
- scientifically defensible validation rather than benchmark chasing.

### 15.2 JAX evidence

Use JAX only where it adds a clean, testable artifact:

1. Export frozen Nesso scores or 384-dimensional affinity representations with stable IDs and split roles.
2. Implement an affine, linear, or small heteroscedastic calibration head in JAX.
3. Train it with the same folds and loss definition as the corresponding PyTorch head.
4. Export parameters in a framework-neutral NPZ/JSON representation.
5. Compare JAX and PyTorch forward outputs and gradients on a fixed minibatch within declared tolerances.
6. Document compile time, steady-state throughput, device placement, PRNG handling, and numerical differences.

This demonstrates JAX competence and framework interoperability. It does not demonstrate end-to-end JAX Nesso fine-tuning, and the portfolio must say so explicitly.

## 16. Milestones

### Milestone 0: freeze the protocol

- Pin every checksum in Section 4.
- Freeze split roles and provisional promotion thresholds.
- Decide the initial Emax universe and uncertainty floor grid.
- Generate the canonical compound equivalence map and leakage report.

### Milestone 1: challenge-matched zero-shot baseline

- Produce E0 predictions for all eligible DRC compounds.
- Evaluate by existing scaffold/random/cliff splits.
- Save frozen affinity representations for cached-head work.

### Milestone 2: minimum trainable head

- Pass tiny-overfit, gradient-scope, and checkpoint-round-trip tests.
- Run E2/E3 over development scaffold folds.
- Compare against E0/E1 across three seeds.

### Milestone 3: uncertainty

- Run weighting-floor ablations.
- Add heteroscedastic head only if weighted regression is stable.
- Evaluate interval calibration.

### Milestone 4: single-point auxiliary task

- Build dose-conditioned observation table and compound-balanced sampler.
- Propagate DRC splits to 2,735 overlapping single-point compounds.
- Run E7/E8 and negative controls.

### Milestone 5: frozen candidate audit

- Select a candidate using development folds only.
- Run scaffold lockbox and project validation suite once.
- Generate the 513 blinded activity predictions without using them for tuning.

### Milestone 6: portfolio closeout

- Publish experiment table, architecture diagram, tests, model card, runtime profile, and negative findings.
- Include the bounded JAX parity/calibration artifact if completed.
- State clearly whether the evidence supports fine-tuning feasibility, PXR benefit, both, or neither.

## 17. Limitations and unresolved questions

1. **No official training recipe.** Loss scaling, optimization, augmentation, and regularization must be designed locally and cannot be claimed to reproduce Nesso training.
2. **Endpoint mismatch.** PXR functional EC50 is not Nesso's pretrained IC50-like binding endpoint. A model can improve assay prediction while becoming less physically meaningful as a binding model.
3. **Trunk gradients are blocked.** Full-model tuning requires modifying detach boundaries and training/inference mode behavior.
4. **Small assay-specific dataset.** Four thousand DRC labels are useful for head adaptation but weak support for a 41-million-parameter full model.
5. **Uncertainty is incomplete.** Curve-fit standard errors do not capture all experimental, biological, and assay-domain uncertainty.
6. **Single-point ambiguity.** Response depends on dose, plate, experiment, efficacy, permeability, and noise; median replication is only one.
7. **Auxiliary overlap.** DRC and single-point data overlap for 2,735 structures, creating severe leakage if rows are split independently.
8. **Prepared-state uncertainty.** One selected protonation/tautomer state is a practical baseline, not known ground truth. Alternate states multiply inputs, not labels.
9. **Constant protein target.** Every example uses the same PXR sequence. Adaptation can learn a PXR-assay-specific ligand function without demonstrating cross-target generalization.
10. **Conformer sensitivity.** Nesso preprocessing can fail for difficult stereochemical structures; retries must preserve graph and stereochemistry and remain reason-coded.
11. **Hardware estimates are not training benchmarks.** Five-recycle inference measured about 3.72 seconds per compound on the RTX 5070 Ti, but training throughput and memory have not been measured.
12. **Existing zero-shot result is external.** The current 653-compound ChEMBL analysis supports plausibility but is not a fair fine-tuning baseline for the primary DRC assay.
13. **JAX scope is limited.** A frozen-feature JAX head is useful portfolio evidence but does not make Nesso a JAX model.
14. **Challenge truth is unavailable.** Any claimed blind benefit must await legitimate challenge evaluation; prediction distributions alone are not evidence of accuracy.

## 18. Definition of done

The proof of concept is complete when one immutable experiment bundle demonstrates:

- a reproducible frozen Nesso DRC baseline;
- a successfully trained and reloaded Nesso head or affinity-module delta;
- leakage-safe compound/scaffold-held-out evaluation;
- direct comparison of MAE and Spearman against frozen and calibrated baselines;
- an uncertainty ablation;
- a completed, honestly interpreted single-point auxiliary experiment or a documented technical/scientific reason it was stopped;
- full provenance and failure manifests;
- measured resource use;
- a limitations/model-card document;
- optional JAX parity/calibration evidence clearly separated from PyTorch Nesso fine-tuning.

The final conclusion must answer two questions independently:

1. Was Nesso-1 fine-tuning technically demonstrated?
2. Did it show credible potential benefit for PXR under leakage-safe validation?

Either answer may be “no” or “inconclusive” without invalidating the engineering value of the project.
