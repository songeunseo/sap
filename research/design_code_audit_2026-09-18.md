# Fixed-objective vector A+C allocator: code audit

Date: 2026-09-18  
Scope: read-only audit of `experiments/dlm_context_response50`; no model shard load, forward, new mask, pruning, or experiment was run.

## Finding

The smallest backend change is to keep the existing objective/readout, pair bank, Wanda order, and mask family, then replace the rank-to-rate map with a direct exact-budget allocator whose candidates are evaluated on the assembled hard sparse model. This isolates allocator/backend behavior. A centered-logit vector A+C objective is a separate readout/objective comparison; changing pairs at the same time would prevent attribution.

## Existing candidate generation and budget

`run.py` reads the frozen 224-entry candidate manifest in its existing order, seven projections per block: `attn_out`, `ff_out`, `q_proj`, `k_proj`, `v_proj`, `ff_proj`, `up_proj` (`run.py:44-47, 100-123`). There are 32 blocks and 224 projections. `collect_block` observes each target block on 80 calibration states, stops immediately after that block, and records the mean input squared norm for each of its seven linears. Crucially, blocks before the target remain in their already applied Uniform50 mask, so later activation caches are sparse-prefix caches (`run.py:100-123`; `experiments/dlm_ppl50/sequential.py:38-73`).

Each probe restores only the target block's seven weights from the dense snapshot, applies a Wanda row mask, scores all 80 context pairs, and restores the Uniform50 block. The row order is stable ascending Wanda score; the first `count` columns are pruned (`sequential.py:75-82`; `run.py:154-177`). Both 48% and 52% probes therefore modify one whole block in a Uniform50 sparse background, not one independent projection. These single-block probes do not themselves preserve the global50% budget; the final allocated masks do.

The current allocator computes signed finite differences per actually removed parameter, ranks the 32 block scores, maps ranks to rates in `[.45,.55]`, then calls the grouped DP (`run.py:173-185`). The DP has 64 groups: for each of 32 blocks, one group for the six 4096-column projections and one for the 12288-column `ff_out`; it permits floor-1/floor/floor+1 and minimizes weighted squared rate error (`experiments/dlm_owl65/core.py:21-63`). Existing A and AC masks each have 224 entries and exactly 3,489,660,928 pruned weights out of 6,979,321,856 (50%). Row counts range from 1,843 to 2,253 for width 4096 and from 5,530 to 6,758 for width 12288. This rounding mechanism is reusable for candidates represented by the existing32 layer-rate inputs, provided every resulting count change is recorded. It cannot be reused unchanged for independent224-projection allocation, and its global rounding corrections may touch units outside an intended two-unit trade. A truly local exchange should instead construct feasible integer counts directly.

The practical minimal backend comparison is:

1. Keep `margins`, `distortion`, `make_pairs`, the 80 pairs, the same candidate order, rates/row counts, and frozen Wanda activations. Compare current rank mapping against a bounded set of exact-budget exchanges, accepting each exchange by the measured assembled-model A+C value.
2. Only after that backend comparison, hold the backend fixed and compare scalar gold-vs-rest A+C with a vector centered-logit A+C readout.
3. Only after that, hold both backend and readout fixed and compare the current gold-reveal pair bank with an equal-count controlled content pair bank.

This keeps backend, objective/readout, and pair changes distinct. A direct hard exchange is lower implementation risk than an unvalidated local quadratic allocator: the saved-probe audit in the synthesis predicts the wrong direction when extrapolated from 48/52% to final allocations.

## Measured current costs

The table uses code-derived forward counts and existing `pipeline.log` event timestamps. Times are observed wall-clock event windows, not new benchmarks and not estimates of a future implementation.

| Current stage | Directly observed/derived workload | Existing evidence | Measured log window |
|---|---:|---|---:|
| Dense pair teacher | 80 pairs; 160 forwards, plus 2 repeated first-pair sanity forwards | `dense_pairs` events 1–80; `run.py:87-97` | 4.12 s from first to last pair event |
| Prefix calibration | 32 × 80 = 2,560 prefix forwards; 7 hooks per target block | 2,592 calibration events = 32 start markers + 2,560 completions; `run.py:100-123` | 45.89 s from first block-0 event to final block-31 completion |
| Uniform score | 80 pair scores = 160 forwards | `probe_pairs` candidate `uniform`, 80 events | 3.99 s |
| 48/52 probe sweep | 32 × 2 × 80 = 5,120 pair scores = 10,240 forwards | `probe_pairs` events for `block00_0.48` through `block31_0.52`; 5,120 condition events (the 5,440 total also includes 3 assembled candidate scores) | 281.64 s from first to last condition event |
| Final hard-mask construction | 224 masks per candidate, exact global budget check | `run.py:208-242`; manifests | No standalone timing recorded |

These event windows are historical observations, exclude time before their first event, and are not end-to-end latency estimates for a new vector objective. Prefix calls also execute fewer than32 blocks and must not be equated to full forwards when comparing compute.

The current collection therefore has 13,122 forward calls before the three assembled candidate scores (162 dense-reference + 2,560 calibration + 160 Uniform + 10,240 probes), by the code path. The 240 pair events for the three joint scores add 480 forwards. These counts exclude GSM8K generation.

## Exact centered-logit readout pulled back to final hidden

The frozen readout receipt records `head_weight=[126464,4096]`, no head bias, untied output weights, RMS final norm with epsilon `1e-5`, and `scale_logits=false` (`experiments/wanda_failure_characterization/geometry_kill_gate/readout_path.json`). The model source applies final RMS norm at `modeling_llada.py:1619-1624`, then the affine head at `1626-1633`; the actual head is `transformer.ff_out` because weights are untied.

Let `h` be the post-`ln_f` hidden at a query position and `U` the frozen vocabulary-by-hidden head matrix (`U` is 126464×4096). Let

```
P = I - 11^T / V,       U_c = P U,
Q = U^T P U = U_c^T U_c.
```

For dense/sparse hidden difference `δh`, the centered-logit difference is exactly `P U δh`, and

```
||P(U h_sparse - U h_dense)||² = δh^T Q δh.
```

For a vocabulary-mean squared metric, use `δh^T Q δh / V`. If `scale_logits` is enabled, multiply `Q` by `1/d_model`; it is false for this model. Any common head bias cancels in the dense/sparse difference, so the identity also holds with a shared affine bias. It is exact only when the hidden is post-final-norm, the same frozen head is used for dense and sparse, and the metric is squared centered-logit error. It is not an exact rewrite of the current gold-vs-rest `log_odds` plus `logsumexp` metric.

For paired A+C, retain the existing endpoint/response structure but use hidden differences:

```
δ_b = h_sparse(before) - h_dense(before)
δ_a = h_sparse(after)  - h_dense(after)
A = mean 0.5 * (δ_b^T Q δ_b + δ_a^T Q δ_a)
C = mean (δ_a - δ_b)^T Q (δ_a - δ_b)
```

This changes the readout/objective axis while leaving the pair algebra intact. It does not require materializing vocabulary logits if the forward path can return final post-norm hidden before the head. The current model wrapper exposes `output_hidden_states`, but still computes logits; an experiment-local `hidden-only` forward option or final-`ln_f` suffix helper is needed for a real head-compute reduction. A hook alone can capture exact hidden values and reduce retained memory, while the existing forward still pays the head projection.

Q is 4096² FP32 values = 67,108,864 bytes = 64 MiB. Forming it from the frozen head costs an estimated `V*d² = 2,121,713,844,224` multiply-accumulate terms (about 4.24e12 FLOPs if multiply and add are counted separately); this is a setup estimate, not a measured time. The head has 517,996,544 elements, about 0.965 GiB in BF16 or 1.93 GiB in FP32. Stream vocabulary rows in chunks to compute the centered row mean and Gram matrix; do not assume Gram formation is automatically faster than direct head application. Once Q is cached, each hidden metric is a 4096×4096 matvec (16,777,216 multiply terms), about 30.875× fewer terms than direct multiplication by U. A chunked direct-head implementation is the fallback; random sketching is an optional approximation and should not replace the primary exact Q comparison.

## Batching, storage, and cache constraints

The existing scorer runs one pair at a time and calls the model twice (`run.py:78-84, 125-131`). A safe batching change is to stack matched `before, after` states as adjacent batch rows, score a microbatch, then restore pair indices before computing C. Batch size must be measured on the target GPU: full bidirectional attention has `B*T²` activation/memory scaling, so no memory or time number is inferred here.

For an upper-bound 256-token query selection, retaining 80 pairs × 2 states × 256 positions × 4096 hidden values is about 320 MiB BF16 or 640 MiB FP32. This is an estimate; storing only query positions and streaming sparse candidates is preferable. Materializing full logits for a two-state, 256-token batch would be about 123.5 MiB BF16 or 247 MiB FP32 per output tensor before model activations, and retaining all 80 pairs would be much larger. Q plus query-only hidden capture avoids that retained-vocabulary cost.

There is no incumbent candidate cache in `dlm_context_response50`; every `score` calls a full model forward. The model has a `past_key_values` interface, but its attention implementation is explicitly noncausal (`modeling_llada.py:692-700`) and the full-sequence path uses all-token attention. A hard mask change in block `k` changes every token state at that block and therefore the entire suffix. Any valid block runner may reuse a cached prefix only when the input pair, model revision, and all weights/masks through block `k-1` are byte-identical; invalidate at the earliest changed block and recompute all suffix blocks plus final norm/head. Do not freeze unmasked-token states or reuse cross-candidate KV values. Cache keys should include pair-bank digest, input IDs/query indices, model revision/head digest, and the prefix mask digest. Late-layer exchanges can save suffix work in a block runner; early-layer exchanges save little, and the present code has no measured savings because it does not implement this runner.

## Validation constraints

For any implementation, verify dense hidden → Q metric against direct centered logits on a small batch, including endpoint and response terms, with FP32 accumulation and a tolerance recorded before scoring. Verify that the same exact mask manifest and row budget are used for all backend/readout comparisons. Record candidate count, forward count, peak memory, and wall time separately for model execution, readout projection, and allocation. Keep NELBO as the primary downstream quality metric; the existing mini100 result remains development evidence only.
