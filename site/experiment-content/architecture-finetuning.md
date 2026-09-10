## Which layers did we fine-tune?

[Exact layer inventory (CSV)](../evidence/reports/architecture/nesso1-finetuning-layers.csv)

### Head-only versus head + LoRA

**Head-only** updates the weights and biases of the three continuous-output Linear layers in **both native ensemble members**: `affinity_module` and `affinity_module2`. The two intervening ReLUs have no parameters. This trains **592,130 parameters** across the two heads; the shared pooled-feature MLP stays frozen.

**Head + LoRA** trains those same heads and adds **24,576 parameters** at three upstream projections per member, for **616,706 trainable parameters** in total. Each adapter uses rank **4**, alpha **4**, and scale **alpha/r = 1**: the effective projection is `W_base + B A`. Only `lora_A` and `lora_B` are trained there; the original projection weights and any biases remain frozen. The external ESM model is frozen: the adapted `esm_proj` layers are small projections **inside each affinity module**, not layers of the external protein language model.

### Exact module paths and dimensions

The continuous heads map the pooled affinity representation to a scalar. The local ESM projections map protein-language-model features into affinity token features. The selected `fc1` transforms pair features inside the first gated affinity transition.

Every suffix below occurs under **both** `affinity_module.` and `affinity_module2.`. The CSV lists all full paths separately. Matrix shapes use `[output, input]` order; A and B have no adapter biases.

| Module suffix under each member | Linear input → output | Trained tensors per member |
| --- | --- | --- |
| `affinity_heads.to_affinity_pred_value.0` | 384 → 384 | weight `[384,384]`, bias `[384]` |
| `affinity_heads.to_affinity_pred_value.2` | 384 → 384 | weight `[384,384]`, bias `[384]` |
| `affinity_heads.to_affinity_pred_value.4` | 384 → 1 | weight `[1,384]`, bias `[1]` |
| `esm_proj.1` | 1280 → 384 | A `[4,1280]`, B `[384,4]` (LoRA arm only) |
| `esm_proj.3` | 384 → 384 | A `[4,384]`, B `[384,4]` (LoRA arm only) |
| `pairformer_stack.layers.0.transition_z.fc1` | 128 → 512 | A `[4,128]`, B `[512,4]` (LoRA arm only) |

Only `fc1` in the **first affinity Pairformer block** receives LoRA—not the main trunk, not the attention projections, and not the other seven affinity blocks. The transition remains gated: `SiLU(fc1(x)) × fc2(x) → fc3`, with its normalization, `fc2` and `fc3` frozen. The input encoders, main pair-only trunk, ESM preprocessing, distogram path and every other base parameter or buffer remain frozen.

The binary branch (`affinity_heads.to_affinity_pred_score` and `affinity_heads.to_affinity_logits_binary`) and shared `affinity_heads.affinity_out_mlp` also stay frozen. Binary outputs can still change when upstream adapters change their input representation. The [binary-dual side experiment](../experiments/binary-dual/index.html) is a separate comparison, **not** the training scope shown here.

### Evidence from the completed comparison

This is the scope of the **20 completed matched fits**: five reused chemical development folds × two label budgets (100 and 500) × two arms, seed 42, 40 epochs. Both member scalars are averaged and converted as `6 − mean` for adapted cellular pEC50; this is not native cellular-endpoint support. These are reused development folds, not a new untouched holdout.

[Completed-fit scope audit](../evidence/reports/architecture/nesso1-finetuning-evidence.json) · [Architecture and shapes](../evidence/artifacts/architecture/architecture_audit.md) · [Training protocol](../evidence/experiments/20260906_affinity_comparison/train_protocol.json) · [Comparison results](../chapters/transfer/index.html) · [Editable Markdown](../experiment-content/architecture-finetuning.md) · [Source hashes](../source-map.json).
