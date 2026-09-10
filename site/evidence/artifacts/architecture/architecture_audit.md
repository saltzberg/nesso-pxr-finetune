# Pinned Nesso-1 architecture audit

Static AST/source reading and safetensors JSON header only; no torch import, no GPU, no model instantiation, no forward or numerical parity validation.

## Critical corrections
- Fast distogram/affinity model, NOT coordinate-generating Boltz pipeline.
- No diffusion sampler, structure decoder, learned confidence module. DiffusionTransformer class reused ONLY within initial local atom attention.
- 48 no-sequence trunk blocks; two separate8-block affinity members.
- transformer_args.num_blocks=2,num_heads=4,token_s=384,activation_checkpointing=false do not construct an affinity head transformer. Only classification_head is read, passed via kwargs, then ignored by AffinityHeadsTransformer; binary branch always instantiated.
- Native cached-input adaptation begins BEFORE entire affinity modules; cached384-vector head-only adaptation is a distinct later boundary.

## Provenance
- Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso`
- Config: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/models/nesso-1/v1.0.0/hparams.json`
- Checkpoint (header only): `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/models/nesso-1/v1.0.0/model.safetensors`
- Header SHA256: `280e4061b9b0c274283caeb013fd12e3d9578d1402696cc743bbb0d7b9ac2394`
- Exact full tensor inventory, source SHA256s and constructor AST inventory are in `architecture_audit.json`.
- Shapes: B=batch (native crop paths assume singleton); M=padded atoms; N=full tokens; Nr=refinement tokens; Na=affinity-pocket tokens.

## Native execution order

### 1. External sequence features
Default ESM model ID facebook/esm2_t33_650M_UR50D; last hidden layer only, token-aligned s_esm [B,N,1280]. ESM is external preprocessing, not a submodule of Nesso1; no verified external ESM parameter count in this audit. esm_num_layers=34 does NOT mean 34 Nesso transformer blocks.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/data/esm.py:8–42`.

### 2. InputEmbedder
AtomEncoder -> atom_enc_proj_z -> AtomAttentionEncoder -> add res_type_encoding(33→384). s_inputs [B,N,384]. No cyclic or mol-type conditioning modules in pinned configuration. pocket_conditioning is accepted via unused kwargs.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/trunk.py:44–118`.

### 3. Initial pair representation
rel_pos linear 139→128; z_init_1 and z_init_2 384→128 broadcast-add, then relative-position, token_bonds 1→128 and bond-type embedding additions. z starts zeros; pair_mask = outer token_pad_mask.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/models/nesso1.py:201–214`.

### 4. Recycled trunk
For i in range(recycling_steps+1): z=z_init+z_recycle(z_norm(z)); ESMModule residual pair update; 48 PairformerNoSeqLayer blocks. z width128 throughout; no evolving single track. Five recycles mean six passes of the SAME 48-block stack, not 288 unique blocks.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/models/nesso1.py:222–260`.

### 5. Optional refinement crop
Only eval AND refine_protein_inference AND recycles>=1: after first full pass, distogram_head(z+z.transpose) predicts full logits; select protein/ligand indices from expected distances; crop z,z_init,s_inputs and relevant feats before remaining passes. Source fallback cutoff22 Å,budget196; CLI and pinned experiment override budget256. Bare hparams lack refine flag, so bare predict_step defaults refinement OFF; do not conflate with CLI/experiment.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/models/nesso1.py:179–260`.

### 6. Final distogram
z_sym=z+transpose(z); biased Linear128→64 gives logits [B,Nr,Nr,64]. If refined, overwrite retained submatrix of initial full logits; output pdistogram then [B,Nfull,Nfull,64], while output z remains refined [B,Nr,Nr,128]. No coordinates generated.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/models/nesso1.py:262–273`.

### 7. Affinity pocket and boundary
Compute expected distances from local final logits; retain valid NONPOLYMER tokens and protein tokens <=15 Å from any NONPOLYMER (not exclusively binder); crop s_inputs,z,feats. Softmax distogram logits to probabilities [B,Na,Na,64]. Detach s_inputs,z,probabilities for member calls. Packet boundary is HERE, before each entire AffinityModule, not after pooled g.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/models/nesso1.py:275–304`.

### 8. Independent affinity members
Call affinity_module then affinity_module2 with same cropped inputs, separate learned parameter sets and separately captured RNG. Both native calls disable CUDA autocast. Merge continuous arithmetic mean; average sigmoid binary probabilities then clamp/logit, NOT average binary logits. affinity_repr left in forward dict is MEMBER1 ONLY.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/models/nesso1.py:306–335`.

### 9. Postprocessing, not learned confidence
predict_step computes expected distances, pocket masks and normalized Shannon distogram entropy_pair, entropy_pp/pl/ll plus crop pp/pl/ll. Squeezes affinity scalars to [B]. Optional metadata z_full is actually refined z[0] in BF16 with refine_mask_full; name does not imply full N×N. No diffusion sampler, coordinate decoder, pLDDT, PAE or learned confidence head.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/models/nesso1.py:370–425`.

## Both independent affinity members: identical schematic, separate parameters

### 1. esm_proj
0 LayerNorm1280 → 1 Linear1280→384+bias → 2 ReLU → 3 Linear384→384+bias; add to s_inputs only where mol_type==0 and token valid. [B,Na,384].
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/affinity.py:30–35`.

### 2. z_norm → z_linear → single broadcasts
Cast z.float; LayerNorm128 → biasless Linear128→128; add s_to_z_prod_in1(s_inputs)_i and s_to_z_prod_in2(s_inputs)_j, each biasless384→128. These are broadcast SUMS, despite prod in names.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/affinity.py:71–80`.

### 3. distogram_proj → pairwise_conditioner
distogram_proj biasless64→128 applied to probabilities. Concatenate current z and projected probabilities (256 channels); dim_pairwise_init_proj.0 LN256 → .1 biasless Linear256→128; transitions.0 then .1, each residual gated Transition128→256→128; add conditioner result to original z.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/encoders.py:99–136`.

### 4. pairformer_stack.layers.0…7
Eight pair-only blocks; width128, triangular attention4 heads ×32 channels; transition hidden512. Mask = binder×protein + protein×binder + binder×binder using valid masks. Protein–protein excluded from mask; diagonal retained here. Residual states are not globally zeroed by this mask.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/affinity.py:85–95`.

### 5. affinity_heads pooling
Rebuild same pair mask and exclude ALL diagonal entries. g128=sum(z*mask)/(sum(mask)+1e-7), sums over both token axes. Ordered protein–binder, binder–protein, binder–binder pairs participate.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/affinity.py:133–150`.

### 6. affinity_heads.affinity_out_mlp
0 Linear128→128+bias →1 ReLU →2 Linear128→384+bias →3 ReLU. Produces affinity_repr=g [B,384]. Despite AffinityHeadsTransformer class name there is NO head transformer attention.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/affinity.py:106–111`.

### 7. affinity_heads.to_affinity_pred_value
0 Linear384→384+bias →1 ReLU →2 Linear384→384+bias →3 ReLU →4 Linear384→1+bias; affinity_pred_value [B,1].
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/affinity.py:112–118`.

### 8. affinity_heads.to_affinity_pred_score → to_affinity_logits_binary
Independent parallel MLP384→384→384→1 (Linear indices0,2,4 biased, ReLU1,3) → biased Linear1→1 produces affinity_logits_binary [B,1]. Intermediate pred_score is NOT returned. Executed after continuous branch.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/affinity.py:119–160`.

## Repeated-block leaves and input details

### 1. PairformerNoSeqLayer forward
Residual tri_mul_out → residual tri_mul_in → residual tri_att_start → residual tri_att_end → residual transition_z. First4 updates have structured dropout; configured trunk0.25/affinity0.1, eval effective p0. No post norm is executed even though constructor stores post_layer_norm.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/layers/pairformer.py:22–81`.

### 2. Triangle multiplication leaves
Each direction: norm_in LN128; p_in and g_in biasless128→256; p_in(norm)*sigmoid(g_in(norm))*mask; split2×128; outgoing einsum bikd,bjkd→bijd or incoming bkid,bkjd→bijd; norm_out LN128 → p_out biasless128→128 times sigmoid(g_out(norm_in_input)) with g_out biasless128→128.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/layers/triangular_mult.py:46–219`.

### 3. Triangle attention leaves
Each direction: layer_norm LN128; linear biasless128→4 gives triangle bias; mha.linear_q/k/v/g/o are each biasless128→128. q/k/v reshape4×32; scaled dot attention+mask+triangle bias, softmax, weighted v, sigmoid gate linear_g, linear_o. Ending attention transposes pair axes before and after. mha.sigmoid is parameterless.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/layers/triangular_attention/attention.py:33–189`.

### 4. Attention internal projection authority
linear_q/k/v output c_hidden*no_heads, so head width32 is verified by code despite stale contradictory comment in source.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/layers/triangular_attention/primitives.py:254–310`.

### 5. Transition leaves
norm LN128; fc1 and fc2 biasless128→H in parallel; silu(fc1(norm(x)))*fc2(norm(x)); fc3 biaslessH→128. H512 in Pairformer, H256 in PairwiseConditioning. Do not diagram as a plain two-linear ReLU MLP.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/layers/transition.py:35–68`.

### 6. AtomEncoder leaves
embed_atom_features biased390→128 on ref_pos,element,4×64 name chars,charge,chirality,hybridization. Pair features [B,M/32,32,128,16]: biasless ref_pos3→16 + reciprocal-distance1→16 + mask1→16, masked by valid same ref-space UID. c_to_p_trans_q/k: ReLU→biasless128→16; p_mlp: ReLU→16→16→ReLU→16→16→ReLU→16→16, residual. structure_prediction=False excludes s_to_c_trans/z_to_p_trans.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/encoders.py:164–314`.

### 7. Atom pair bias and pooling
atom_enc_proj_z LN16→biasless16→12, split3 layers×4 heads. Atom transformer3 layers width128 with32 queries/128 keys per window. atom_to_token_trans biasless128→384→ReLU; normalized atom_to_token matrix mean-pools atoms into token s; add residue-type33→384.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/encoders.py:317–408`.

### 8. Atom attention repeated layer leaves
AdaLN: a_norm LN128 no affine, s_norm LN128 weight/no bias, s_scale biased128→128 sigmoid, s_bias biasless128→128. AttentionPairBias proj_q biased128→128, proj_k/v/g/o biasless128→128, proj_z parameterless Rearrange; four heads32. Gate attention output by output_projection Linear128→128+bias→Sigmoid, add residual. output_projection_linear and output_projection.0 alias SAME Linear, do not count twice. Then residual ConditionedTransitionBlock; post_lnorm Identity.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/transformers.py:142–207`.

### 9. ConditionedTransitionBlock leaves
AdaLN as above; swish_gate.0 biasless128→512 then SwiGLU splits halves yielding256; multiply a_to_b biasless128→256; b_to_a biasless256→128; multiply output_projection biased128→128→Sigmoid on conditioning. This unusual extra multiplicative a_to_b is present.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/transformers.py:19–67`.

### 10. ESMModule leaves
esm_mlp LN1280→Linear1280→384+bias→ReLU→Dropout0.05→Linear384→384+bias; add s_inputs_proj biased384→384; esm_z_1/2 biasless384→128 broadcast SUM, multiply pair mask, add to z. No outer product, despite source comment. Same module reapplied each recycle. use_esm_all_layers=false; no esm_layer_weights parameter.
Source: `/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/snapshot/image/nesso/model/modules/esm_module.py:25–68`.

## Current adaptation and cached-input boundary
User-specified current targets, also matches pinned protocol_pilot.json training field; this audit does NOT inspect a fitted adaptation checkpoint.
Native trunk/input/ESM update/distogram and all affinity base parameters except continuous-head0,2,4 weights+bias; LoRA on selected affinity projections only. Shared pooled-feature MLP and binary branch frozen; binary outputs may nevertheless change due to upstream LoRA.
Capture each native member call independently with CPU/CUDA RNG and outputs. Restore member-specific RNG for differentiable FP32 eval replay; use_kernels=False in replay. No trunk recomputation per adapted optimization step.

| Trainable base tensor | Shape |
|---|---|
| `affinity_module.affinity_heads.to_affinity_pred_value.0.weight` | `[384, 384]` |
| `affinity_module.affinity_heads.to_affinity_pred_value.0.bias` | `[384]` |
| `affinity_module.affinity_heads.to_affinity_pred_value.2.weight` | `[384, 384]` |
| `affinity_module.affinity_heads.to_affinity_pred_value.2.bias` | `[384]` |
| `affinity_module.affinity_heads.to_affinity_pred_value.4.weight` | `[1, 384]` |
| `affinity_module.affinity_heads.to_affinity_pred_value.4.bias` | `[1]` |
| `affinity_module2.affinity_heads.to_affinity_pred_value.0.weight` | `[384, 384]` |
| `affinity_module2.affinity_heads.to_affinity_pred_value.0.bias` | `[384]` |
| `affinity_module2.affinity_heads.to_affinity_pred_value.2.weight` | `[384, 384]` |
| `affinity_module2.affinity_heads.to_affinity_pred_value.2.bias` | `[384]` |
| `affinity_module2.affinity_heads.to_affinity_pred_value.4.weight` | `[1, 384]` |
| `affinity_module2.affinity_heads.to_affinity_pred_value.4.bias` | `[1]` |

| LoRA target (both members explicitly listed) | Base weight | A | B |
|---|---|---|---|
| `affinity_module.esm_proj.1` | [384, 1280] | [4, 1280] | [384, 4] |
| `affinity_module.esm_proj.3` | [384, 384] | [4, 384] | [384, 4] |
| `affinity_module.pairformer_stack.layers.0.transition_z.fc1` | [512, 128] | [4, 128] | [512, 4] |
| `affinity_module2.esm_proj.1` | [384, 1280] | [4, 1280] | [384, 4] |
| `affinity_module2.esm_proj.3` | [384, 384] | [4, 384] | [384, 4] |
| `affinity_module2.pairformer_stack.layers.0.transition_z.fc1` | [512, 128] | [4, 128] | [512, 4] |

Rank4, alpha4, scale1. Continuous-head trainables: 592,130; LoRA trainables: 24,576; total: 616,706. These are target-derived counts, not proof of fitted state.
Protocol documents native log10(IC50/uM), converted6-native to fit cellular pEC50; that is an adapted endpoint, NOT native cellular assay support. Units are protocol-sourced, not inferable from scalar Linear.

## Output semantics

### member
- `affinity_pred_value`: [B,1] continuous.
- `affinity_repr`: [B,384] shared g.
- `affinity_logits_binary`: [B,1].

### native_forward
- `pdistogram`: [B,Nfull,Nfull,64] logits; merged full/refined when crop active.
- `z`: [B,Nr,Nr,128], or full N when not refined.
- `affinity_pred_value1`: member1 continuous [B,1].
- `affinity_pred_value2`: member2 continuous [B,1].
- `affinity_pred_value`: (v1+v2)/2.
- `affinity_probability_binary`: (sigmoid(l1)+sigmoid(l2))/2.
- `affinity_logits_binary`: logit(clamp(mean probability,1e-6,1-1e-6)).
- `affinity_repr`: MEMBER1 g only; not an ensemble representation.
- `crop_protein_tokens/keep_indices`: conditional refinement indices relative to full tokens.
- `affinity_keep_indices`: indices relative to refined token axis, not necessarily full axis.

Softmax logits weighted by64 centers: first1.5 Å,last24.5 Å, interior midpoints of63 linearly spaced boundaries2..22 Å. Not coordinate reconstruction.
-sum(p log p)/log64; PP/PL/LL off-diagonal mask means, crop variants; uncertainty proxy not learned structural confidence.

## Header inventory summary
| Top-level module | Stored tensors | Stored elements |
|---|---:|---:|
| `affinity_module` | 339 | 6,132,996 |
| `affinity_module2` | 339 | 6,132,996 |
| `distogram_head` | 2 | 8,256 |
| `esm_module` | 10 | 888,448 |
| `input_embedder` | 84 | 1,103,152 |
| `pairformer_module` | 1776 | 26,873,856 |
| `rel_pos` | 1 | 17,792 |
| `token_bonds` | 1 | 128 |
| `token_bonds_type` | 1 | 896 |
| `z_init_1` | 1 | 49,152 |
| `z_init_2` | 1 | 49,152 |
| `z_norm` | 2 | 256 |
| `z_recycle` | 1 | 16,384 |

Checkpoint entries: 2,558. Stored elements: 41,273,464. Source-proven alias-deduplicated parameter elements: 41,223,928. ESM preprocessing model excluded.
The three atom-attention layers each register the same output-gating Linear under two paths; do not count those aliases twice. Both affinity members have matching339-entry shape inventories, but separate checkpoint prefixes; no sharing between members is constructed.

## Limitations
- No model execution or fitted-checkpoint inspection; shape claims are checkpoint-header/config/source grounded.
- External ESM internal heads/layers/parameter count not independently audited from an ESM config/checkpoint.
- Optional generic encoder structure_prediction code is not a native active branch in this pinned Nesso1 configuration.
- Source-derived parameter alias dedup, not a runtime parameter-storage check.
