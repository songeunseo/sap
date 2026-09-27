# Group-Balanced Reveal/Remain DLM-ABS

## Motivation

Fixed 2× and 50× multipliers make the Reveal-group loss mass depend on each state's group sizes. This experiment replaces that heuristic with a frozen modeling assumption: Reveal and Remain each receive half of the total masked-token weight mass. The 50:50 ratio was not tuned.

## Frozen Setup

Model `GSAI-ML/LLaDA-8B-Base` at revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`; the historical 80 states, persisted Reveal/Remain partition, DLM outer normalization, ABS aggregation, 224 Linear matrices, and exact row-wise 50% sparsity were reused unchanged. State digest: `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`. Partition digest: `e847478e57e2f6d3e8ac2adacd7622f96d9ab7c0dddfc24588365f7383d10eb8`.

## Group-Balanced Objective

`L_group(s) = |M_s| × [0.5 mean_(j in R_s)(CE_sj) + 0.5 mean_(j in U_s)(CE_sj)] / [p_mask(t) × 256]`.

The single factor `|M_s|` preserves the historical masked-token loss scale. Equivalently, `alpha_R=|M|/(2|R|)` and `alpha_U=|M|/(2|U|)`; the weighted token sum is then divided by `p_mask(t)×256` exactly once.

## Weighting Verification

All 80 states have Reveal share 50% and Remain share 50%. The effective `alpha_R/alpha_U=|U|/|R|` range is `6.0`–`242.0`. The complete state-level table is in `group_balance_all_states.json`.

## Mask Diagnostics

| Pair | Score Spearman | Mask XOR | Retained overlap | Differing decisions |
|---|---:|---:|---:|---:|
| group_balanced_abs/reveal2_abs | 0.985072568 | 4.2767% | 95.7233% | 298,482,894 |
| group_balanced_abs/reveal50_abs | 0.994770909 | 2.6442% | 97.3558% | 184,546,946 |
| group_balanced_abs/uniform_abs | 0.984832392 | 4.3068% | 95.6932% | 300,587,198 |

The diagnostic report was frozen before full GSM8K evaluation. Per-matrix median/mean/extrema and top-10 XOR matrices are in `mask_diagnostic_report_frozen.json`.

## Full GSM8K

| Method | Correct / 1319 | Accuracy |
|---|---:|---:|
| Uniform-ABS | 745 | 56.48% |
| Reveal-2×-ABS | 767 | 58.15% |
| Group-Balanced-ABS | 697 | 52.84% |
| DLM-SQUARE | 701 | 53.15% |
| Wanda | 677 | 51.33% |
| SparseGPT | 584 | 44.28% |

All rows use the same frozen 5-shot, 256-step, strict exact-match protocol.

## Paired Analysis

| Comparison | Both correct | GB only | Baseline only | Both wrong | Δ accuracy | Exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| GB vs Uniform | 592 | 105 | 153 | 469 | -3.639 pp | 0.00335353 |
| GB vs Reveal-2× | 605 | 92 | 162 | 460 | -5.307 pp | 1.32856e-05 |

Exact p-values use only discordant examples. A non-significant result is not evidence of equivalence.

## Conclusion

**FACT:** Group-Balanced-ABS scored `697/1319` (52.84%) under the frozen protocol; paired counts and mask diagnostics are reported above.

**INTERPRETATION — Outcome C:** Strongly balancing the small Reveal group against the much larger Remain group does not improve downstream performance. The historical Reveal-2x gain should therefore not be interpreted as evidence that stronger Reveal emphasis is generally better.

No new weighting ratio is proposed from this result.
