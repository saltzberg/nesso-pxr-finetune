# Nesso-1 binary-head correction

Status labels: `[ ]` pending, `[~]` in progress, `[x]` complete.

## 1. Provenance and reconstruction

- [x] Confirm official continuous-output units and binary-output semantics.
- [x] Audit the installed Nesso 1.0.0 affinity architecture and ensemble rule.
- [x] Pin checkpoint and cached-feature hashes.
- [x] Reconstruct both member binary logits/probabilities and the ensemble probability for every cached development, validation, and challenge row.
- [x] Prove current-checkpoint forward parity and characterize retained historical-JSON differences without conflating them.
- [x] Reconcile development (3,344), common validation (714), full validation (790), and challenge (513) coverage explicitly.

## 2. Leakage-safe supervised readouts

- [x] Evaluate continuous affinity alone, binary probability/logit alone, and both readouts together using the fixed chemical-family outer folds.
- [x] Select all tunable calibration/readout hyperparameters inside the outer training partition.
- [x] Report development nested-CV and validation metrics with chemical-family bootstrap intervals.
- [x] Compare against the existing continuous-head fine-tune, frozen 384D probes, and LightGBM without mixing cohorts.

## 3. Functional-potency proxy classification

- [x] Prespecify proxy-positive thresholds at pEC50 >= 4, >= 5, and >= 6.
- [x] Report prevalence, ROC-AUC, average precision, ROC/PR curves, and family-bootstrap intervals.
- [x] Report proxy contingency matrices at Nesso binder probability >= 0.5.
- [x] Count high-pEC50 proxy positives below the released binder cutoff.
- [x] Stratify by Emax-qualified and lower-Emax cohorts where the metadata support it.
- [x] State throughout that pEC50 thresholds are functional-potency proxies, not binding ground truth.

## 4. Representation and adaptation tests

- [x] Compare 384D member representations with and without appended continuous/binary pretrained readouts.
- [x] Evaluate ridge and bounded small nonlinear probes under nested chemical-family CV.
- [x] Evaluate a bounded dual-readout adapter initialized from both pretrained branches without claiming binder-classifier fine-tuning.
- [x] Name the historical supervised model precisely as `fine-tuned continuous affinity head`.

## 5. Pair-pooling extension

- [x] Apply the binary branch exactly to cached all-pairs and ligand-ligand-only 384D representations.
- [x] Validate shape and forward parity.
- [x] Evaluate binary-only and combined-readout pooling ablations and preserve the late-pooling-only interpretation.

## 6. Deliverables and checks

- [x] Write machine-readable prediction, metric, bootstrap, and provenance artifacts.
- [x] Generate bounded, publication-ready ROC/PR, proxy matrix, distribution, and comparison figures.
- [x] Write a correction report and concise dossier replacement wording.
- [x] Add focused tests for reconstruction, split isolation, thresholds, and metrics.
- [x] Run tests and `git diff --check`.
- [x] Do not publish, push, or change external AetherArk state.

