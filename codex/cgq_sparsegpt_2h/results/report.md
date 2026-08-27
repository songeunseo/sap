# LLaDA CGQ-SparseGPT Two-Hour Decision Report

## Decision

**Outcome A — Promising transfer.** At 50% unstructured sparsity, CGQ changed
8.06% of SparseGPT mask decisions and improved every held-out preservation
metric at all four tested timesteps. This is evidence for a larger controlled
downstream experiment, not evidence for a new pruning method.

## Implementation

- Model: `GSAI-ML/LLaDA-8B-Base` at revision
  `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, loaded through the repository's
  existing `get_llm` path in BF16.
- Pruned modules: the existing seven Linear modules in each of 32 blocks:
  `q_proj`, `k_proj`, `v_proj`, `attn_out`, `ff_proj`, `up_proj`, and `ff_out`.
- Plain called the original `SparseGPT.add_batch(inp, out)` path. CGQ passed one
  optional `[batch, token]` tensor into the same accumulator.
- The module hook receives `inp` as `[B, T, C]`. CGQ computes
  `r = where(input_ids == mask_id, 1.0, 0.7) + sqrt(max(softmax(logits_fp32)))`,
  detaches it, broadcasts it as `[B, T, 1]`, and forms `(inp.float() * r)` once.
  Existing SparseGPT then reshapes to `[B*T, C]`, transposes to `[C, B*T]`,
  applies its unchanged normalization, and accumulates `H += X_weighted @
  X_weighted.T`. No coefficient is applied after this product.
- Damping (`0.01`), block size (`128`), sequential compensation, layer/module
  order, sparsity allocation, and activation propagation were unchanged.
- Dense confidence and all 16 held-out reference logits were computed before
  any pruning. Held-out logits were cached only at masked positions as FP16 CPU
  tensors; both sparse models used this same cache, identified by SHA-256
  `ee4d8b7b0508cdf2cd00dba5d79a67613638328772ea56991d2242e709570575`.
  KL was computed in FP32.
- Full pruned models were evaluated in memory and then released; no full
  checkpoints were written.

Files changed for the experiment: `lib/sparsegpt.py`, `lib/prune_llada.py`,
`main_llada.py`, `cgq_sparsegpt.py`, `tests/test_sparsegpt.py`,
`tests/test_cgq_sparsegpt.py`, `tests/test_main_llada.py`, and
`codex/cgq_sparsegpt_2h/config.json`.

## Controls

- Calibration: 4 WikiText-2 train sequences × 4 timesteps = 16 states,
  sequence length 256, seed 0.
- Held-out: 4 different WikiText-2 train sequences × the same 4 timesteps = 16
  states. Calibration and held-out clean-sequence hashes did not overlap.
- Calibration state SHA-256:
  `b69643776e06bdba6b1928d18108fde5b4b6e1bfb7558ceda195b753c40a364e`.
- Held-out state SHA-256:
  `c8ee525da65b79bce69fa1b65e01c770864526b6381a3a4d751d36c28ba0c500`.
- The real first-block all-ones path reproduced Plain exactly: zero Hessian
  diagonal disagreement and zero mask XOR across 218,103,808 weights.
- The tensor-shape test uses `[B=2, T=2, C=2]` inputs with distinct weights per
  sample and token and checks a hand-derived Hessian. It detects feature-axis
  broadcasting or applying `r` more than once.

## Calibration signal

There were 4,096 calibration tokens: 2,069 masked and 2,027 unmasked.
Values below are mean ± population standard deviation, followed by `[min, max]`
for `r`.

| t | Masked ratio | Masked confidence | Unmasked confidence | Masked r | Unmasked r |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 22.07% (226) | 0.7683 ± 0.2735 | 0.8901 ± 0.1897 | 1.8561 ± 0.1883 [1.1225, 2.0000] | 1.6352 ± 0.1248 [0.8430, 1.7000] |
| 0.4 | 39.55% (405) | 0.6782 ± 0.3134 | 0.8707 ± 0.2000 | 1.7928 ± 0.2227 [1.1541, 2.0000] | 1.6236 ± 0.1328 [0.8305, 1.7000] |
| 0.6 | 61.52% (630) | 0.4920 ± 0.3279 | 0.8380 ± 0.2167 | 1.6548 ± 0.2515 [1.0784, 2.0000] | 1.6029 ± 0.1509 [0.8208, 1.7000] |
| 0.8 | 78.91% (808) | 0.3096 ± 0.2972 | 0.7449 ± 0.2599 | 1.4973 ± 0.2494 [1.1567, 2.0000] | 1.5440 ± 0.1805 [0.9715, 1.6999] |

All confidence values were finite and in `[0, 1]`. Masked positions received
mask weight `1.0`; unmasked positions received `0.7`.

## Pruning diagnostics

| Metric | Plain SparseGPT | CGQ-SparseGPT | Delta |
|---|---:|---:|---:|
| Actual sparsity | 50.000153% | 50.000154% | +0.000001 pp |
| Exact zeros / 6,979,321,856 | 3,489,671,637 | 3,489,671,694 | +57 |
| Mask XOR vs Plain | 0 | 562,287,443 (8.0565%) | +8.0565 pp |
| Mean layer-wise XOR | 0 | 8.0565% | +8.0565 pp |
| Mean module-wise XOR | 0 | 7.8520% | +7.8520 pp |

The global mask XOR is well above the 0.5% negligible-effect threshold.

Top ten layer/module cells by mask XOR:

| Rank | Block | Module | XOR |
|---:|---:|---|---:|
| 1 | 25 | attn_out | 13.2648% |
| 2 | 29 | attn_out | 12.6852% |
| 3 | 26 | attn_out | 12.3773% |
| 4 | 27 | attn_out | 12.3638% |
| 5 | 20 | attn_out | 12.1705% |
| 6 | 24 | attn_out | 12.0991% |
| 7 | 19 | attn_out | 11.8744% |
| 8 | 21 | attn_out | 11.8059% |
| 9 | 23 | attn_out | 11.6923% |
| 10 | 22 | attn_out | 11.5988% |

Across 224 modules, the mean Hessian-diagonal cosine similarity was `0.998055`
and the mean relative L2 difference was `1.694175`. Thus CGQ mostly retained
the diagonal direction while materially changing its scale and the resulting
SparseGPT decisions. The largest relative diagonal changes were block 10
`ff_out` (`2.0102`), block 8 `ff_out` (`1.9157`), block 11 `ff_out` (`1.8584`),
block 7 `ff_out` (`1.8533`), and block 27 `ff_out` (`1.8379`).

## Held-out dense-prediction preservation

All metrics use 2,034 masked tokens from the 16 held-out states.

| Metric | Plain SparseGPT | CGQ-SparseGPT | Delta |
|---|---:|---:|---:|
| Sparsity | 50.000153% | 50.000154% | +0.000001 pp |
| Mask XOR vs Plain | 0 | 8.0565% | +8.0565 pp |
| Held-out logits KL, Dense ‖ Sparse | 0.305456 | 0.279045 | -0.026411 (-8.65%) |
| Dense top-1 agreement | 69.6657% | 72.3206% | +2.6549 pp |
| Top-1 confidence MAE | 0.081035 | 0.076679 | -0.004356 (-5.38%) |
| WinoGrande-128 | Not run | Not run | Excluded by approved scope |

Per-timestep primary metrics:

| t | Masked tokens | Plain KL | CGQ KL | KL delta | Plain agreement | CGQ agreement | Agreement delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 204 | 0.217584 | 0.208667 | -0.008917 | 86.2745% | 86.7647% | +0.4902 pp |
| 0.4 | 419 | 0.271240 | 0.258018 | -0.013222 | 76.1337% | 77.5656% | +1.4320 pp |
| 0.6 | 598 | 0.363858 | 0.307628 | -0.056230 | 71.2375% | 74.2475% | +3.0100 pp |
| 0.8 | 813 | 0.302182 | 0.286516 | -0.015666 | 61.0086% | 64.5756% | +3.5670 pp |

CGQ also reduced confidence MAE at every timestep: by `0.004239`, `0.008064`,
`0.006147`, and `0.001157` at `t = 0.2, 0.4, 0.6, 0.8`, respectively.

## Conclusion

1. **Did CGQ materially change the pruning mask?** Yes. Global XOR was 8.0565%.
2. **Did CGQ better preserve dense DLM predictions?** Yes. KL fell 8.65%,
   agreement rose 2.65 percentage points, and confidence MAE fell 5.38%.
3. **Is the effect consistent across timestep?** Yes. KL, agreement, and
   confidence MAE all improved at all four timesteps.
4. **Is a larger downstream experiment justified?** Yes. The next justified
   experiment is the preregistered larger-calibration Plain-vs-CGQ comparison
   with downstream WinoGrande and GSM8K evaluation; no CGQ formula changes or
   extra pruning heuristics are justified by this run.

Machine-readable results are in `results.json`; the exact cached corrupted
states are in `states.pt`. Total wall time was 183.45 seconds. No WinoGrande,
GSM8K, extra sparsity, ablation, or modified CGQ formula was run.
