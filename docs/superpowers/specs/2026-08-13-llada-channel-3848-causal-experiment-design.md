# LLaDA Channel 3848 Protection Causal Experiment Design

## Objective

Test whether part of LLaDA-8B-Base's GSM8K collapse at 75% unstructured
sparsity is caused by pruning weights directly connected to residual activation
channel 3848. This is a causal ablation of mask selection, not a new pruning
criterion: Wanda and Sink-Aware importance scores remain unchanged.

## Fixed Model Architecture

The official LLaDA-8B-Base configuration and a meta-device instance of the
repository model agree on the following architecture:

- 32 LLaMA-style transformer blocks
- residual width `d_model = 4096`
- gated MLP width `mlp_hidden_size = 12288`
- 32 query heads and 32 key/value heads
- seven prunable `nn.Linear` weights per block

PyTorch linear weights have shape `[out_features, in_features]`. The block
forward path determines the protected axes below.

| Layers | Module | Weight shape | Protected indices for channel `c` | Weights per layer | Reason |
|---|---|---:|---|---:|---|
| 0-31 | `q_proj` | `[4096, 4096]` | `[:, c]` | 4,096 | Reads normalized residual channel `c` into queries |
| 0-31 | `k_proj` | `[4096, 4096]` | `[:, c]` | 4,096 | Reads normalized residual channel `c` into keys |
| 0-31 | `v_proj` | `[4096, 4096]` | `[:, c]` | 4,096 | Reads normalized residual channel `c` into values |
| 0-31 | `attn_out` | `[4096, 4096]` | `[c, :]` | 4,096 | Writes attention output to residual channel `c` |
| 0-31 | `ff_proj` | `[12288, 4096]` | `[:, c]` | 12,288 | Reads normalized residual channel `c` into the gated MLP |
| 0-31 | `up_proj` | `[12288, 4096]` | `[:, c]` | 12,288 | Reads normalized residual channel `c` into the other MLP gate |
| 0-31 | `ff_out` | `[4096, 12288]` | `[c, :]` | 12,288 | Writes MLP output to residual channel `c` |

Rows of `q_proj`, `k_proj`, and `v_proj` and columns of `attn_out` are
attention-internal coordinates, not residual channel coordinates. Rows of the
MLP input projections and columns of `ff_out` are MLP-internal coordinates.
They are therefore not protected. Embeddings and the vocabulary output layer
are outside the repository's pruning scope and remain excluded.

Across all blocks, the pruning scope contains 6,979,321,856 weights. A residual
channel protection set contains 1,703,936 weights, or 0.0244140625% of that
scope.

## Immutable Baselines and Independent Variants

Wanda and Sink-Aware each produce one original 75% baseline using the existing
score calculation and baseline mask selection unchanged. Every constrained
mask is derived from that method's immutable baseline mask and its score at the
time the baseline mask was selected.

For Wanda, channel 3848 and three seeded, distinct random channels are four
independent branches from the same original Wanda baseline. A branch never
starts from another constrained branch, and deltas are never accumulated.
Sink-Aware channel 3848 is independently derived from the immutable Sink-Aware
baseline. Random controls are not required for Sink-Aware.

The three random channels are sampled without replacement from `[0, 4096)`
excluding 3848, using the experiment seed. Each random channel protects the
same module axes and therefore the same number of weights as channel 3848.

## Minimal-Intervention Mask Transformation

For a module, let `B` be its immutable baseline prune mask (`True` means prune),
`S` its unchanged Wanda or Sink-Aware score, and `P_c` the protected set for
channel `c`.

Only protected weights that baseline would prune are changed:

```text
restore = B & P_c
protected_restored = count(restore)
```

Protected weights already surviving in `B` are not an intervention.

### Protected input columns

For `q_proj`, `k_proj`, `v_proj`, `ff_proj`, and `up_proj`, each restored
`[:, c]` weight is compensated within the same output row. The compensation is
the lowest-score weight satisfying all of:

- it survived in `B`;
- it is not protected;
- it is in the same output row.

Thus every affected row retains its exact baseline prune count.

### Protected output rows

For `attn_out` and `ff_out`, all baseline-pruned weights in `[c, :]` are
restored. They are compensated by the same number of lowest-score
baseline-surviving, non-protected weights from other rows of that module.
Scores are never compared across modules or layers. Thus the module retains its
exact baseline prune count even though individual row counts may change.

If a required compensation set does not contain enough candidates, execution
fails instead of expanding the scope.

For every constrained module:

```text
protected_restored == compensation_pruned
mask_difference_count == 2 * protected_restored
count(constrained_mask) == count(baseline_mask)
constrained_mask[P_c] is all False
```

Because every module preserves its exact baseline prune count and the relevant
input widths are divisible by four, every 75% model has exactly 75% global
sparsity.

## Artifact and Data Flow

During a baseline pruning run, each module's baseline mask and score are used
immediately to derive independent channel deltas. A delta records only restored
flat indices and their original dense values, compensation indices, and
diagnostic counts. Full score tensors are not persisted.

The baseline checkpoint is saved once. Each variant is materialized from that
same baseline checkpoint by restoring recorded dense values and zeroing the
recorded compensation indices. This makes delta non-accumulation explicit and
avoids recomputing scores after an intervention.

Variant checkpoints are created and evaluated one at a time because the
workspace has insufficient free disk for all dense-size checkpoints at once.
Completed temporary variant checkpoints are removed after their metrics,
deltas, configuration, and diagnostics are persisted. Baseline checkpoints,
compact delta artifacts, and experiment reports are retained.

## Required Diagnostics

The report records the following globally and per module:

- `total_prunable_weights`
- `total_pruned_weights`
- `actual_global_sparsity`
- `protected_total`
- `protected_weight_fraction`
- `protected_already_survived_in_baseline`
- `protected_already_survived_in_baseline / protected_total`
- `protected_restored`
- `protected_restored / protected_total`
- `compensation_pruned`
- `mask_difference_count`
- `mask_difference_fraction`
- module sparsity before and after the constraint

The two protected-set ratios are headline results alongside downstream
accuracy because they quantify how much of the pathway baseline already
preserved and how much the intervention actually changed.

The implementation verifies both decision-mask counts and actual zero counts.
It also validates runtime module names and tensor shapes against the architecture
table before pruning.

## Evaluation Protocol and Execution Gates

Evaluation reuses the repository's `llada_dist` GSM8K task, prompt, tokenizer,
temperature-zero generation, `gen_length=1024`, `steps=1024`,
`block_length=1024`, and `remasking=low_confidence`. A small sanity evaluation
uses a fixed 64-example limit and otherwise identical settings.

Execution order:

1. Architecture validation and mask-transformation unit tests.
2. Generate one immutable Wanda 75% baseline and its channel deltas.
3. Evaluate Wanda 75% and Wanda 75% + protect-3848 on the same 64 GSM8K
   examples.
4. If protect-3848 produces more correct answers than baseline, run full
   GSM8K for Dense, Wanda 75%, Wanda + protect-3848, and three independent
   Wanda random-channel controls.
5. Apply the identical constraint helper to one immutable Sink-Aware 75%
   baseline and evaluate Sink-Aware 75% and Sink-Aware + protect-3848.
6. Extend to no other benchmark unless the GSM8K signal is clear.

Dense and existing repository results are used as protocol sanity references;
no 75% result is assumed from the paper where none is reported.

## Activation Diagnostic

Using the same eight fixed calibration sequences for every model, forward hooks
measure the post-block residual stream at blocks 0, 15, and 31:

```text
mean(abs(block_output[:, :, 3848]))
```

The report compares Dense, each reached 75% baseline, and its channel-3848
variant. Activation preservation is interpreted only together with paired
GSM8K accuracy and is not sufficient by itself for a causal conclusion.

## Result Table and Interpretation

The primary table contains:

| Method | GSM8K | Delta from method baseline | Restored / protected | Already survived / protected |
|---|---:|---:|---:|---:|
| Dense LLaDA | | | | |
| Wanda 75% | | | | |
| Wanda 75% + protect-3848 | | | | |
| Sink-Aware + Wanda 75% | | | | |
| Sink-Aware + Wanda 75% + protect-3848 | | | | |
| Wanda 75% + random protect (each and mean) | | | | |

`Delta from method baseline` is the absolute percentage-point accuracy change.
The final assessment follows the requested Cases A-D: large and random-control-
specific recovery, small recovery, negligible change, or degradation.

## Test Strategy

One focused unit-test module uses small synthetic matrices to prove:

- architecture names map to the correct row or column;
- protected weights cannot be pruned;
- input-column compensation stays within each corresponding row;
- output-row compensation stays within its module and outside the protected row;
- only baseline-pruned protected weights count as restored;
- independent channel variants equal direct transforms of the baseline and do
  not equal accumulated transforms;
- all count and sparsity invariants hold for both Wanda-like and Sink-like
  score tensors.

Tests are written and observed failing before production implementation.
