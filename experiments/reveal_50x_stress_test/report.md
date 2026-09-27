# Reveal-50x Stress Test

## Setup

The ONLY changed variable was the raw Reveal:Remain weight: **2:1 -> 50:1**. The model/revision, frozen calibration states, precomputed partition, DLM loss, ABS aggregation, 224 matrices, and exact row-wise 50% mask semantics were unchanged.

## Partition verification

- Frozen states: **80** (8 WikiText-2 sequences × 10 timesteps)
- Masked-token observations: **10,267**
- Reveal / Remain observations: **116 / 10,151**
- States with 1 / 2 Reveal tokens: **44 / 36**
- State digest: `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`
- Partition digest: `e847478e57e2f6d3e8ac2adacd7622f96d9ab7c0dddfc24588365f7383d10eb8`

## Effective weighting

All 80 states satisfy mean normalized masked-token weight = 1 (maximum absolute error `2.22e-16`).

| Quantity | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| \|M_s\| | 7 | 133.0 | 128.3375 | 245 |
| \|R_s\| | 1 | 1.0 | 1.4500 | 2 |
| \|U_s\| | 6 | 131.5 | 126.8875 | 243 |
| Normalized Reveal weight | 6.250000 | 31.435132 | 29.368723 | 41.609589 |
| Normalized Remain weight | 0.125000 | 0.628703 | 0.587374 | 0.832192 |
| Reveal loss-weight share | 17.1233% | 37.8875% | 42.1046% | 89.2857% |

## Mask diagnostic

The diagnostic classification was frozen before inspecting GSM8K: **2x WEIGHTING WAS TOO WEAK TO TEST THE SIGNAL CLEANLY**.

| Comparison | Global score Spearman | Mask XOR | Retained-set overlap | Differing decisions |
|---|---:|---:|---:|---:|
| reveal2_abs/uniform_abs | 0.999986563 | 0.1390% | 99.8610% | 9,703,638 |
| reveal50_abs/uniform_abs | 0.994463498 | 2.5806% | 97.4194% | 180,110,218 |
| reveal50_abs/reveal2_abs | 0.994785909 | 2.5049% | 97.4951% | 174,824,812 |

Historical 2x-vs-Uniform: Spearman `0.999986607`, mask XOR `0.1386%`; same-pass deltas were `4.31e-08` and `0.000454` percentage points.

### Reveal-50x vs Uniform

- Score Spearman median/min: `0.995021014` / `0.982843466`
- Mask XOR median/mean/max: `2.4993%` / `2.5426%` / `4.1559%`

| Rank | Layer | Matrix | Mask XOR | Spearman |
|---:|---:|---|---:|---:|
| 1 | 0 | k_proj | 4.1559% | 0.992652594 |
| 2 | 0 | v_proj | 3.7671% | 0.994788024 |
| 3 | 31 | ff_out | 3.4620% | 0.991656768 |
| 4 | 27 | ff_out | 3.4135% | 0.990704449 |
| 5 | 0 | q_proj | 3.4057% | 0.993128444 |
| 6 | 28 | ff_out | 3.3769% | 0.991202368 |
| 7 | 26 | ff_out | 3.3407% | 0.991269100 |
| 8 | 29 | ff_out | 3.2359% | 0.991883985 |
| 9 | 8 | attn_out | 3.2226% | 0.993350848 |
| 10 | 25 | ff_out | 3.2189% | 0.992104294 |

### Reveal-50x vs Reveal-2x

- Score Spearman median/min: `0.995301800` / `0.983908252`
- Mask XOR median/mean/max: `2.4212%` / `2.4677%` / `4.0155%`

| Rank | Layer | Matrix | Mask XOR | Spearman |
|---:|---:|---|---:|---:|
| 1 | 0 | k_proj | 4.0155% | 0.993218103 |
| 2 | 0 | v_proj | 3.6451% | 0.995145447 |
| 3 | 31 | ff_out | 3.3470% | 0.992215191 |
| 4 | 27 | ff_out | 3.3085% | 0.991290587 |
| 5 | 0 | q_proj | 3.2846% | 0.993659984 |
| 6 | 28 | ff_out | 3.2657% | 0.991793887 |
| 7 | 26 | ff_out | 3.2444% | 0.991781406 |
| 8 | 8 | attn_out | 3.1455% | 0.993661906 |
| 9 | 25 | ff_out | 3.1312% | 0.992535455 |
| 10 | 29 | ff_out | 3.1279% | 0.992420023 |


Validation: exactly 224 intended matrices; 6,979,321,856 weights; exact global and row-wise sparsity 50%; all score summaries finite; all 224 persisted Reveal-50x packed-mask files passed size and SHA-256 readback. No retraining, compensation, Fisher term, timestep reweighting, heuristic, or allocation change was introduced.

## GSM8K mini

| Method | Correct / 100 | Accuracy |
|---|---:|---:|
| Uniform-ABS | 58 / 100 | 58.0% |
| Reveal-2x-ABS | 59 / 100 | 59.0% |
| Reveal-50x-ABS | 58 / 100 | 58.0% |

Reveal-50x vs Uniform: both correct `48`, 50x only `10`, Uniform only `10`, both wrong `32`.

Reveal-50x vs Reveal-2x: both correct `49`, 50x only `9`, 2x only `10`, both wrong `32`.

This fixed mini-100 result is noisy and is not final method evidence.

## Decision

FACT: Reveal-50x materially changed the pruning decision (2.5806% XOR vs Uniform) while retaining high score correlation (0.994463498). Mini accuracy tied Uniform at 58% and trailed Reveal-2x by one item.

INTERPRETATION: Mask sensitivity is established as indicated by the diagnostic, but the 100-example downstream result is too noisy for a method claim.
