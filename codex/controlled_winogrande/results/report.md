# Controlled WinoGrande comparison

All direct runs use LLaDA-8B-Base, unstructured pruning, WikiText-2 calibration
with 8 sequences of length 256 and seed 0, and the full 1,267-example
WinoGrande evaluation with 5-shot, CFG 0, 128 Monte Carlo samples, and batch
size 8. Accuracy and standard error are percentage points.

| Sparsity | Method | Direct acc. ± SE | Sink-Aware paper | Direct − paper |
|---:|:---|---:|---:|---:|
| Dense | Base | 74.59 ± 1.22 | 69.30 | +5.29 |
| 25% | **Wanda** | **74.51 ± 1.22** | 68.59 | +5.92 |
| 25% | Sink-Aware + Wanda | 74.35 ± 1.23 | 68.59 | +5.76 |
| 25% | SparseGPT | 73.88 ± 1.23 | 67.56 | +6.32 |
| 25% | Sink-Aware + SparseGPT | 73.95 ± 1.23 | 69.53 | +4.42 |
| 25% | Mean DLM | 74.43 ± 1.23 | — | — |
| 50% | Wanda | 69.85 ± 1.29 | 64.56 | +5.29 |
| 50% | Sink-Aware + Wanda | 70.17 ± 1.29 | 65.27 | +4.90 |
| 50% | SparseGPT | 68.90 ± 1.30 | 64.64 | +4.26 |
| 50% | Sink-Aware + SparseGPT | 67.72 ± 1.31 | 65.82 | +1.90 |
| 50% | **Mean DLM** | **70.48 ± 1.28** | — | — |
| 75% | Wanda | 50.59 ± 1.41 | 47.43 | +3.16 |
| 75% | Sink-Aware + Wanda | 50.67 ± 1.41 | 49.17 | +1.50 |
| 75% | **SparseGPT** | **51.46 ± 1.40** | 50.04 | +1.42 |
| 75% | Sink-Aware + SparseGPT | 49.96 ± 1.41 | 51.07 | −1.11 |
| 75% | Mean DLM | 50.91 ± 1.41 | — | — |

Bold marks the best pruned method in each direct-run sparsity group.

## Reading the result

- At 25%, Wanda is highest, with Mean DLM 0.08 points behind.
- At 50%, Mean DLM is highest: +0.31 over Sink-Aware + Wanda, +0.63 over
  Wanda, +1.58 over SparseGPT, and +2.76 over Sink-Aware + SparseGPT.
- At 75%, SparseGPT is highest, with Mean DLM 0.55 points behind.
- These small gaps should not be interpreted as statistically established
  method superiority from one run. All methods saw the same evaluation items,
  so a paired comparison over per-example outcomes is more appropriate than
  comparing the marginal standard errors shown above.

The paper column is a reference, not an apples-to-apples reproduction: the
paper's displayed values and the direct runs differ by as much as 6.32 points,
and the paper's exact evaluation protocol is not established by this table.
Use the direct column for the controlled method comparison.

The pruning methods share the same high-level WikiText-2 8×256, seed-0
calibration setting. Mean DLM and the baseline methods use different loader
paths, however, so byte-identical raw calibration token spans have not been
proven.
