# EXP-002 — GSM8K Aggregation Evaluation

## Question

Does SUM, ABS, or SQUARE aggregation of the exact same official DLM gradients produce better 50% pruned LLaDA generation performance?

## Fixed Protocol

- Model: `GSAI-ML/LLaDA-8B-Base` revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2` in BF16
- Task: GSM8K, 5-shot, repository `LLaDAEvalHarness` and lm-eval `gsm8k` strict-match exact match
- Generation: temperature 0, length 256, block length 256, denoising steps 256
- Pruning: 50% unstructured over the same 224 Linear matrices. DLM and Wanda masks are exactly 50% per row. SparseGPT reuses the repository's standard intrinsic 128-column-block global threshold, so its overall sparsity is approximately 50% but individual rows need not be exactly 50%.
- DLM methods: the frozen EXP-001 masks from run `20260828T175010-2484545`
- Wanda/SparseGPT: standard reference baselines using eight clean WikiText-2 256-token spans, seed 0
- Baseline caveat: Wanda/SparseGPT are not calibration-compute-matched to the DLM methods' 80 corrupted states.

## EXP-001 Context

| Pair | Spearman | Mask XOR |
|---|---:|---:|
| SUM / ABS | -0.000048 | 0.500045 |
| SUM / SQUARE | -0.000045 | 0.500046 |
| ABS / SQUARE | 0.992708 | 0.028463 |

- Sign consistency mean: 0.127214
- Spike ratio mean: 1.249317

## GSM8K Results

| Method | Sparsity | Accuracy | Correct / N | Eval seconds |
|---|---:|---:|---:|---:|
| Dense | 0.0% | 0.711145 | 938 / 1319 | 40135.0 |
| Wanda | 50.0% | 0.513268 | 677 / 1319 | 40092.1 |
| SparseGPT | 50.0% | 0.442760 | 584 / 1319 | 40141.3 |
| DLM-SUM | 50.0% | 0.000000 | 0 / 1319 | 40085.5 |
| DLM-ABS | 50.0% | 0.564822 | 745 / 1319 | 40092.1 |
| DLM-SQUARE | 50.0% | 0.531463 | 701 / 1319 | 40092.1 |

## Paired Correctness

| Pair | Both correct | A correct / B wrong | A wrong / B correct | Both wrong |
|---|---:|---:|---:|---:|
| DLM-SUM vs DLM-ABS | 0 | 0 | 745 | 574 |
| DLM-SUM vs DLM-SQUARE | 0 | 0 | 701 | 618 |
| DLM-ABS vs DLM-SQUARE | 611 | 134 | 90 | 484 |
| DLM-ABS vs Wanda | 549 | 196 | 128 | 446 |
| DLM-ABS vs SparseGPT | 481 | 264 | 103 | 471 |
| DLM-SQUARE vs Wanda | 534 | 167 | 143 | 475 |
| DLM-SQUARE vs SparseGPT | 464 | 237 | 120 | 498 |

## Interpretation

SUM versus ABS directly tests whether retaining signed directionality helps or whether cross-state cancellation harms the ranking. ABS versus SQUARE tests whether the 2.85% mask disagreement induced by emphasizing large state-wise effects changes downstream exact match. These are observed performance comparisons on one benchmark and do not establish causality. Wanda and SparseGPT are standard reference baselines, not calibration-compute-matched controls.

## Decision

**square not justified.** This label follows the observed ordering only; paired counts should be considered before treating a small accuracy difference as robust.

## Next Step

No additional experiment is launched automatically.
