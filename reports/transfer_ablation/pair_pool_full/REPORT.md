# Nesso-1 direct final pair-pooling ablation

## Positive control

All-pairs historical parity: **pass**. 
Extraction covered 4,134 / 4,134 compounds with 0 failures.

## Prespecified primary probe: ridge on mean 384D representation

| Dataset | Final pool | N | MAE | Spearman |
|---|---:|---:|---:|---:|
| nested_development | all_pairs | 3344 | 0.63279 | 0.65371 |
| lockbox_all_790 | all_pairs | 790 | 0.59630 | 0.61912 |
| nested_development | ligand_ligand_only | 3344 | 0.60624 | 0.67248 |
| lockbox_all_790 | ligand_ligand_only | 790 | 0.59421 | 0.63107 |
| nested_development | receptor_ligand_only | 3344 | 0.63109 | 0.64216 |
| lockbox_all_790 | receptor_ligand_only | 790 | 0.59644 | 0.62334 |

## Paired ligand-ligand-only minus all-pairs differences

| Dataset | Metric | Difference | 95% family-bootstrap CI | Reading |
|---|---:|---:|---:|---|
| nested_development | mae | -0.02655 | [-0.04191, -0.01137] | better without direct receptor-ligand final-pool cells |
| nested_development | spearman | 0.01876 | [0.00222, 0.03467] | better without direct receptor-ligand final-pool cells |
| lockbox_all_790 | mae | -0.00210 | [-0.02346, 0.01951] | confidence interval includes no difference |
| lockbox_all_790 | spearman | 0.01195 | [-0.02097, 0.04306] | confidence interval includes no difference |

## Evidence boundary

This intervention tests whether direct receptor-ligand cells in the final affinity pooling operation add predictive information. Ligand-ligand cells were already protein-conditioned upstream. A null result therefore does not establish that Nesso ignores protein; it only shows that this direct final-pool pathway is unnecessary for fixed-PXR prediction under this probe. The receptor-ligand-only result is diagnostic rather than the principal control.

The 790-compound lockbox is a previously opened retrospective test set. No lockbox labels were used for ridge selection or fitting.
