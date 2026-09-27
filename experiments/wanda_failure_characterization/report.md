# Wanda failure characterization on LLaDA

## Scope and preregistration

This was an analysis-only study of unmodified standard Wanda. The primary targets were frozen before measurement as `Y50_KL = mean masked-token KL` and `Y50_pos = mean max(deltaL, 0)`. No CGQ, token weighting, alternative saliency, structured pruning, downstream benchmark, or new backward pass was used.

The held-out manifest was frozen before the failure map (SHA-256 `bb2cdaa6985a6ae6f75b7623a58653dcc80c73b7240cddbea551d62e7bab2b5e`): WikiText-2 loader seed 1, spans 8--15, five fixed timesteps 0.1/0.3/0.5/0.7/0.9, 40 states total. Its clean token spans have no token-identical overlap with the eight seed-0 calibration spans.

## Artifact-supported facts

### Exact Wanda implementation

For a Linear input, a 2-D tensor is first given a batch dimension; Linear inputs are reshaped to `[batch * token, feature]` and transposed to `[feature, batch * token]`. `WrappedGPT.add_batch` casts the input to FP32 and updates

`scaler_row <- scaler_row * old_nsamples/(old_nsamples + batch) + ||X_feature||_2^2/(old_nsamples + batch)`.

Thus, for the eight batch-size-one, length-256 clean sequences used here, `A_j` is exactly the mean across the eight sequences of the sum across 256 tokens of `X[token,j]^2`. Wanda uses `abs(W_ij) * sqrt(A_j)`. Sorting is row-wise and prunes exactly `int(in_features * sparsity)` entries per row. The 224 matrices are the seven repository Linear modules in each of 32 blocks.

Masks were generated at exactly 50% and 75%. Their hashes, score summaries, and threshold geometry are in `wanda_mask_manifest.json`; the FP32 `A_j` vectors are persisted in `wanda_sufficient_statistics.pt`.

### Functional and numerical validation

Masks were never written into parameters. A forward hook evaluated `F.linear(X, W.masked_fill(mask, 0), bias)`. On the validation case this matched an ordinary cloned Linear containing `W * M` exactly (max absolute difference 0), and prefix/suffix execution matched ordinary full forward exactly (logit max absolute difference 0).

Every module call used one no-ablation sham, 50%, and 75% variants in the same three-way suffix batch. Reported damage is candidate minus same-path sham. Relative to a separately computed batch-1 dense reference, sham drift was:

| Metric | median | p99 | max |
|---|---:|---:|---:|
| absolute loss drift | 0.002143 | 0.012005 | 0.029160 |
| masked-token KL | 0.000481 | 0.001456 | 0.005231 |

This confirms that batch-shape drift is material and that same-path correction was necessary. Very small raw effects should not be interpreted independently of this control.

Model parameter SHA-256 was identical before and after collection.

### 224-module failure map

At 50%, the median module mean KL was 0.001358 and the maximum was 0.022711. Median module positive loss damage was 0.002971 and the maximum was 0.020220. At 75%, the corresponding KL values were 0.005178/0.192192 and positive-loss values were 0.008332/0.191601.

The largest 50% KL modules were:

| Module | KL | positive deltaL | local reconstruction error |
|---|---:|---:|---:|
| block 31 ff_out | 0.022711 | 0.020220 | 0.006054 |
| block 31 ff_proj | 0.014430 | 0.015446 | 0.014862 |
| block 30 ff_out | 0.008965 | 0.008754 | 0.042921 |
| block 31 up_proj | 0.008823 | 0.008938 | 0.021891 |
| block 0 v_proj | 0.003349 | 0.007510 | 0.022011 |

Mean 50% KL by type was: k_proj 0.000839, q_proj 0.000827, v_proj 0.001310, attn_out 0.001449, up_proj 0.001892, ff_proj 0.002024, and ff_out 0.002653. Early/middle/late layer-region mean KL was 0.001240/0.001330/0.002190; at 75% it was 0.003981/0.005578/0.011498. The map is therefore neither uniform nor explained by module type alone: the terminal MLP modules are conspicuous within their types.

Mean dense/pruned top-1 agreement was 0.97837 at 50% and 0.95949 at 75%; mean confidence MAE was 0.00670 and 0.01242.

### Local reconstruction

Local relative reconstruction error was associated with functional damage, but did not identify the extreme functional ordering perfectly. Pooled Spearman was 0.667 for `E_rec50` versus `Y50_KL`, and 0.494 versus `Y50_pos`. For example, block-31 ff_out had the largest functional KL despite an unusually small local error of 0.00605.

### Pre-pruning distributions and calibration mismatch

All exact per-module weight moments, quantiles, kurtosis, top-energy concentration; clean and held-out activation statistics; and clean/held-out mismatch are in `per_module_statistics.csv`. Across modules, clean-versus-held-out feature-energy Spearman had mean 0.7788, median 0.8216, and range 0.4236--0.9981. Relative L2 mismatch had mean 0.3625, median 0.1423, and range 0.0178--7.6493. Activation energy's timestep CV had mean 0.0888 and median 0.0579.

### Wanda versus magnitude

Within modules, sampled Wanda-score versus magnitude Spearman averaged 0.9331 (median 0.9609); Wanda-score versus the expanded activation factor averaged 0.2623 (median 0.2214). The diagnostic Wanda/magnitude row-wise pruned-set XOR averaged 8.48% at 50% and 8.69% at 75%. Wanda is therefore strongly magnitude-dominated, although its activation term changes a nonzero minority of decisions.

The complete descriptive log-score variance decomposition is persisted. It is treated only as numerical decomposition, not causal importance attribution.

### Threshold and mask geometry

At 50%, an average 0.824% of scores lay within ±1% of their row threshold and 4.03% within ±5%. Column prune-rate CV averaged 0.2300 (range 0.0636--0.7142). Exact row threshold gaps, row-to-row variance, entropy, and top-column concentration at both sparsities are in the module table. No clustering-aware criterion was derived.

### Timestep and sequence variability

The 50% KL failure ranking was stable across held-out sequences: pairwise module-rank Spearman median 0.8275 (p10/p90 0.7702/0.8968). Across timesteps it was also stable: median 0.8405 (0.6758/0.8645). At 75%, sequence and timestep medians were 0.8788 and 0.8814.

In contrast, 50% positive-loss rankings were unstable: sequence median Spearman 0.1053 (p10/p90 -0.0165/0.1568) and timestep median 0.1490 (0.0555/0.2492). KL is the cleaner functional map; the preregistered positive-loss target carries substantial state-specific sign/tail variability.

### Statistical associations

Pooled correlations looked strong. For `Y50_KL`, `Var(log|W|)`, weight p99/median, and weight CV had Spearman -0.802, -0.775, and -0.763. After within-module-type standardization these fell to -0.301, -0.270, and -0.250. Clean/held-out activation rank match was -0.604 pooled and -0.318 after type standardization. Associations also changed sign for some individual projection types, so no scalar passed the robustness gate consistently.

For `Y50_pos`, the strongest pooled weight statistics were around -0.50, but within-type standardized associations were about -0.19 or weaker; clean/held-out relative L2 mismatch reached -0.325 but was not directionally consistent across types.

A fixed 11-variable ridge model evaluated by leave-one-layer-out cross-validation failed to generalize: `Y50_KL` R2 = -0.761 with rank correlation 0.118; `Y50_pos` R2 = -0.517 with rank correlation 0.053. This rejects the claim that the tested small scalar feature set robustly predicts held-out layer-level vulnerability.

Existing EXP-001 gradient statistics were skipped: the persisted artifacts did not contain underlying per-module score vectors with an exact mapping sufficient for this analysis, and the protocol prohibited a new backward pass.

## Statistical interpretation

**Outcome C — failure dominated by module/layer identity.** Wanda damage is real, highly nonuniform, reproducible in KL, and amplified at 75%. It concentrates especially in the final MLP path, with block-0 v_proj as a separate outlier. Local reconstruction is informative but insufficient. Several pooled pre-pruning correlations are largely confounded by type/layer scale, and the fixed interpretable model does not generalize to a held-out layer.

Therefore this study identifies a reliable *where*—notably terminal MLP modules—but does not identify a robust criterion-level *why*. The evidence does not support designing a score correction from the best pooled correlation.

## Speculation

The block-31 outliers may reflect downstream amplification or lack of opportunities for later layers to repair perturbations; the block-0 v_proj outlier may reflect a distinct early information bottleneck. These are hypotheses only. Neither follows causally from weight concentration, activation mismatch, threshold crowding, or reconstruction error in this dataset.

## Exactly one recommended next diagnostic

Repeat only the preregistered 50% module-only KL failure map on a second independently frozen held-out WikiText-2 span set, using the identical masks and same-path sham, to test whether the terminal-MLP and block-0-v_proj vulnerability pattern replicates out of sample before investigating a mechanism.
