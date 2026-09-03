# EXP-004 — Decode-Aware Token Importance Direction × Gradient Aggregation

## Objective

Determine whether next-reveal tokens or remain-unresolved tokens should receive greater masked-token loss importance, and whether token weighting changes the preferred ABS versus SQUARE cross-state aggregation.

## Hypotheses

- H1: REVEAL-UP may preserve imminent commitments that condition later denoising steps.
- H2: REMAIN-UP may preserve difficult unresolved tokens requiring further refinement.
- Neither direction is assumed correct a priori.

## Exact Weighting Equations

For masked set `M_s`, reveal set `R_s`, and remain set `U_s`, UNIFORM uses raw weight 1. REVEAL-UP uses raw reveal:remain `2:1`; REMAIN-UP uses `1:2`. Every state divides raw weights by their masked-token mean, so `mean(alpha[M_s]) = 1`.

```text
L_s(alpha) = sum_j(alpha_s,j * CE_j) / p_mask(t) / sequence_length
d_i,s = -w_i * dL_s/dw_i
ABS_i = mean_s(abs(d_i,s))
SQUARE_i = mean_s(d_i,s ** 2)
```

## Calibration Details

- Model: `GSAI-ML/LLaDA-8B-Base`, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, BF16
- Frozen EXP-001 run: `20260828T175010-2484545`
- Calibration: 8 WikiText-2 samples × 10 timesteps = 80 states, sequence length 256
- Partition: `steps_remaining=max(1, ceil(p_mask*256))`; next count is the first value from the repository `get_num_transfer_tokens`; selection uses the repository low-confidence remasking confidence rule.
- Pruning: exact 50% unstructured row-wise over 224 Linear matrices.

## Sanity Checks

```json
{
  "all_alpha_means_one": true,
  "evaluation_config_hash": "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add",
  "exact_rowwise_half": true,
  "gradient_isolation": "torch.autograd.grad with parameter .grad always None",
  "no_parameter_updates": true,
  "partition_frozen_across_aggregation": true,
  "same_80_states": true,
  "uniform_mask_xor": {
    "abs": 0.0005331704249748817,
    "square": 0.0006445480080751273
  },
  "uniform_masks_exact": false,
  "uniform_reproduction_gate_passed": true
}
```

## Mask Diagnostics

| Pair | Spearman | Mask XOR | Mask IoU | Top-50% overlap |
|---|---:|---:|---:|---:|
| remain_abs/remain_square | 0.992612 | 0.028641 | 0.944365 | 0.971359 |
| remain_abs/uniform_abs | 0.999994 | 0.000949 | 0.998103 | 0.999051 |
| remain_square/uniform_square | 0.999977 | 0.001654 | 0.996699 | 0.998346 |
| reveal_abs/remain_abs | 0.999971 | 0.001957 | 0.996093 | 0.998043 |
| reveal_abs/reveal_square | 0.992851 | 0.028207 | 0.945180 | 0.971793 |
| reveal_abs/uniform_abs | 0.999987 | 0.001386 | 0.997232 | 0.998614 |
| reveal_square/remain_square | 0.999847 | 0.003920 | 0.992194 | 0.996080 |
| reveal_square/uniform_square | 0.999935 | 0.002629 | 0.994757 | 0.997371 |

Full matrix, layer, and module-type rows are in `mask_diagnostics.json`.

## GSM8K Results

| Method | Correct / N | Accuracy |
|---|---:|---:|
| UNIFORM-ABS | 745 / 1319 | 56.4822% |
| UNIFORM-SQUARE | 701 / 1319 | 53.1463% |
| REVEAL-ABS | 767 / 1319 | 58.1501% |
| REVEAL-SQUARE | 717 / 1319 | 54.3594% |
| REMAIN-ABS | 747 / 1319 | 56.6338% |
| REMAIN-SQUARE | 709 / 1319 | 53.7528% |

Protocol: full 1,319-example GSM8K, 5-shot, strict exact match, temperature 0, generation length 256, block length 256, and 256 denoising steps. UNIFORM results are frozen EXP-002 results; all reuse passed the exact evaluation-config hash assertion.

## Paired Counts and Statistical Tests

| Comparison | Both correct | A only | B only | Both wrong | Difference (pp) | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| REVEAL-ABS vs UNIFORM-ABS | 685 | 82 | 60 | 492 | 1.6679 | 0.0776532 | 0.388266 |
| REMAIN-ABS vs UNIFORM-ABS | 684 | 63 | 61 | 511 | 0.1516 | 0.928492 | 1 |
| REVEAL-ABS vs REMAIN-ABS | 694 | 73 | 53 | 499 | 1.5163 | 0.0901229 | primary |
| REVEAL-SQUARE vs UNIFORM-SQUARE | 646 | 71 | 55 | 547 | 1.2130 | 0.181224 | 0.724896 |
| REMAIN-SQUARE vs UNIFORM-SQUARE | 638 | 71 | 63 | 547 | 0.6065 | 0.545534 | 1 |
| REVEAL-SQUARE vs REMAIN-SQUARE | 641 | 76 | 68 | 534 | 0.6065 | 0.559821 | 1 |
| REVEAL-ABS vs REVEAL-SQUARE | 633 | 134 | 84 | 468 | 3.7908 | 0.000863365 | 0.00604355 |
| REMAIN-ABS vs REMAIN-SQUARE | 615 | 132 | 94 | 478 | 2.8810 | 0.0136694 | 0.0820165 |

`REVEAL-ABS vs REMAIN-ABS` is the preregistered primary comparison. The other seven exact two-sided binomial McNemar tests receive Holm adjustment. Discordant counts are reported directly; p-values are not interpreted causally.

## Interpretation

- ABS: Outcome A ordering: evidence favors imminent commitment / transition preservation.
- SQUARE: Outcome A ordering: evidence favors imminent commitment / transition preservation.

### Aggregation interaction

- REVEAL: ABS had the higher observed exact-match accuracy.
- REMAIN: ABS had the higher observed exact-match accuracy.

The preregistered primary comparison had exact p=0.0901229 and did not reach alpha=0.05. After Holm adjustment of the seven exploratory comparisons, REVEAL-ABS vs REVEAL-SQUARE remained significant at alpha=0.05.

These statements describe observed orderings under the frozen protocol and do not establish a universal confidence or causal mechanism.

## Implementation Deviations

- Full score tensors were processed module-wise and discarded because only 5 GiB disk was available. `*_scores.json` retains per-module shape, SHA-256, statistics, sparsity, and linked mask hashes.
- UNIFORM recomputation bit-exact status, global mask XOR, and the calibrated nondeterminism gate result are reported under Sanity Checks; evaluation started only after that gate passed.
- No permutation/random-token control, confidence-continuous weighting, ratio sweep, or additional benchmark was added.

## Decision

EXP-004 is complete. No follow-up experiment was launched automatically.
