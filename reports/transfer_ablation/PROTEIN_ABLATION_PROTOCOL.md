# Protein and pairwise-information ablation protocol

## Evidence boundary

The published 384-dimensional vectors are outputs of Nesso's affinity pair
pathway, but PXR is constant across every row. Performance from those cached
vectors therefore establishes **transfer from a protein-conditioned
representation**, not causal use of PXR-specific protein--ligand information.
The cache contains neither token-pair tensors nor a ligand-only representation,
so this question cannot be answered by masking dimensions of the existing
384D vectors.

In the released architecture, each affinity member:

1. adds projected ESM protein embeddings to protein single-token features;
2. combines trunk pair features, single-to-pair projections, and a predicted
   distogram;
3. updates receptor--ligand and ligand--ligand cells through an affinity
   pairformer;
4. averages those pair cells and maps the result to the 384D
   `affinity_repr`;
5. applies the final regression MLP.

This makes the intervention point explicit.

## Implemented direct-pooling ablation

[`src/nesso_pxr/pairwise_ablation.py`](../../src/nesso_pxr/pairwise_ablation.py)
reproduces Nesso's final pair mask and separates it into:

- `all_pairs`: the released receptor--ligand plus ligand--ligand pool;
- `ligand_ligand_only`: excludes every receptor--ligand cell from the final
  average;
- `receptor_ligand_only`: excludes every ligand--ligand cell.

[`scripts/run_nesso_pair_pool_ablation.py`](../../scripts/run_nesso_pair_pool_ablation.py)
captures all three 384D representations for both affinity members in one
upstream Nesso pass. A 16-compound, five-recycle GPU smoke test passed with:

- 16/16 successful predictions;
- exact equality to the previously cached baseline 384D vectors
  (maximum absolute difference 0);
- exact equality when the released regression head was reapplied
  (maximum absolute difference 0);
- 62.2 seconds wall time and 1.08 GB peak allocated GPU memory.

The smoke test validates the intervention and extraction plumbing. Its sample
size is deliberately too small for a performance conclusion. The artifacts are
under the ignored path `artifacts/experiments/pair_pool_smoke/`.

A full, label-blind extraction is run inside the validated Nesso environment.
The ignored upstream support assets can be recovered at the pinned revision:

```bash
python scripts/download_nesso_checkpoint.py
hf download recursionpharma/nesso v1.0.0/hparams.json \
  --revision 1896c84c7186c506c7efd79051480809d51098bf \
  --local-dir models/nesso-1
hf download recursionpharma/nesso ccd.pkl \
  --revision 1896c84c7186c506c7efd79051480809d51098bf \
  --local-dir models/nesso-1

PYTHONPATH=src python scripts/run_nesso_pair_pool_ablation.py \
  --selection artifacts/experiments/e0/full/selection.csv \
  --processed-root artifacts/experiments/e0/full/processed \
  --checkpoint models/nesso-1/v1.0.0 \
  --ccd models/nesso-1/ccd.pkl \
  --output-dir artifacts/experiments/pair_pool_full
```

At the measured 3.89 seconds per compound, 4,134 compounds require about
4.5 GPU-hours. This replay must remain one serial, feature-row-ordered process
when exact historical parity is required. A four-worker smoke check took 67
seconds versus 62 seconds serial and changed the contemporaneous all-pairs
vectors (maximum absolute differences 0.499 for member 1 and 0.149 for member
2), showing process/order dependence rather than a safe parallel speedup.

For each representation variant, the primary analysis should
repeat the nested ridge probe with the existing five chemical-family folds,
fit the final probe on development only, and evaluate the untouched lockbox.
Report paired changes in MAE and Spearman with chemical-family bootstrap
intervals. Head capacity and optimization must be held fixed across variants.

This ablation answers a narrow question: whether **direct receptor--ligand
cells in the final pooling operation** add predictive information. It does not
create a protein-free representation, because protein information can already
have propagated into ligand--ligand cells upstream.

## Stronger causal ladder

The following controls should be predeclared and evaluated in order:

1. **Final-pool removal (implemented).** Compare `all_pairs` with
   `ligand_ligand_only`.
2. **Affinity-pairformer removal.** Zero receptor--ligand blocks immediately
   before the affinity pairformer, use a ligand--ligand-only pair mask, zero
   those blocks again before pooling, and retrain the same probe. This removes
   the affinity module's cross-pair path but still permits protein information
   inherited from the main trunk.
3. **Protein-embedding counterfactual.** Re-extract with five deterministic,
   composition-preserving permutations of the PXR residue embeddings. Run both
   a fixed-crop version, which isolates representation effects, and normal
   end-to-end recropping, which includes pocket-selection effects.
4. **Matched decoy receptor.** Replace PXR with a predeclared, length-matched
   nuclear-receptor control while leaving every ligand and conformer unchanged.
   Again separate fixed-crop and end-to-end versions.
5. **Multi-target interaction swap.** On a dataset containing the same or
   overlapping ligands against multiple receptors, compare the correct
   protein--ligand pairing with protein labels permuted across targets. This is
   the decisive test of target selectivity; a single constant-protein dataset
   cannot provide it by itself.

Zeroing and protein replacement are distribution shifts, so both must be
reported; agreement is more persuasive than either alone. A drop only in the
final-pool ablation supports use of direct pair cells. A drop only after protein
replacement supports protein use elsewhere in the trunk. No drop under either
would mean that this PXR result is adequately explained by ligand-dominated
features, without implying that Nesso ignores proteins on other targets.
