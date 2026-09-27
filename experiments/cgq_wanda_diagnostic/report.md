# CGQ–Wanda Information-Redundancy Diagnostic

## Decision

**Outcome C — strong but highly localized redistribution.** Across all 224 Linear
modules, CGQ leaves the feature statistic highly rank-similar to Wanda (mean
Spearman **0.99738**, median **0.99794**), and the typical within-module ratio
variation is small (median CV of `A_CGQ/A` **0.01419**). The redistribution is not
uniform everywhere, however: `attn_out` and `ff_out` have mean ratio CVs of
**0.05210** and **0.05138**, respectively, with a maximum of **0.10620**. Thus CGQ
is mostly redundant globally but adds localized feature-ranking information.

No pruning mask was generated or stored, no model weight was modified, and no
downstream evaluation was run.

## Artifact-supported facts

### Frozen inputs and confidence coverage

- Model: `GSAI-ML/LLaDA-8B-Base`, revision
  `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, bfloat16.
- EXP-001 run: `20260828T175010-2484545`; frozen-state SHA-256
  `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`.
- Inputs: 8 WikiText-2 sequences × 10 timesteps = 80 states, length 256;
  20,480 token positions; 224 Linear matrices (7 × 32 blocks).
- EXP-004 stores confidence only for masked positions. Its reveal/remain partition
  was validated and reused, while confidence for all 20,480 positions was recomputed
  from the pinned dense model. Under the original PyTorch 2.8 execution environment,
  every saved masked confidence was reproduced exactly (maximum absolute error 0).
  PyTorch 2.14 did not reproduce it and was not used for the reported analysis.
- Dense parameter SHA-256 before and after analysis was identical:
  `ea78a6ff576b2ba4b6961a3fdfe00e138e8a85cdab32b7fdc47e3b25c3e5e13b`.

### Exact implemented Wanda statistic

For a hooked Linear input, repository `WrappedGPT.add_batch` first promotes a 2-D
input to `[1, tokens, features]`. For a 3-D Linear input it reshapes to
`[batch × tokens, features]`, transposes to `[features, batch × tokens]`, and only
then converts it to FP32. Let `B_k` be the batch size of hook call `k`. The code
updates:

```text
scaler_row <- scaler_row * N_old/(N_old + B_k)
N           <- N_old + B_k
scaler_row <- scaler_row + ||X_k[:, j]||_2^2 / N
```

Here every state is forwarded with batch size 1, so after 80 states the exact
equivalent statistic is:

```text
A_j = scaler_row[j] = (1/80) * sum_s sum_t FP32(X[s,t,j])^2
```

`nsamples` counts batch examples (80), not tokens (20,480). Wanda then uses
`|W_ij| * sqrt(A_j)`. The diagnostic used the identical hook input and FP32
conversion, and accumulated:

```text
A_j_CGQ = (1/80) * sum_s sum_t r[s,t]^2 * FP32(X[s,t,j])^2
```

with no additional normalization.

### Exact CGQ factor and token distributions

The frozen factor was used without tuning:

```text
r = 1.0 + sqrt(c)  if masked
r = 0.7 + sqrt(c)  if unmasked
c = max_v softmax(logits)_v
```

Across all positions, confidence had mean **0.60994**, median **0.75781**, p10
**0.05371**, p90 **0.99609**, and range **[0.00235, 1.0]**. The resulting factor
had mean **1.57118**, median **1.64786**, p10 **1.22097**, p90 **1.91216**, and
range **[0.74848, 2.0]**.

Pooled across all module-token observations, Spearman correlations were weak:

| Relationship | Pooled Spearman | State×module mean | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| confidence vs activation norm | 0.08192 | 0.30731 | 0.31102 | 0.01839 | 0.61674 |
| CGQ factor vs activation norm | 0.06336 | 0.16809 | 0.14500 | -0.10994 | 0.48955 |

The state-wise distribution contains 17,920 correlations (80 states × 224
modules). Timestep-stratified pooled confidence correlations were
`0.0236, 0.0303, 0.0356, 0.0495, 0.0472, 0.0625, 0.0765, 0.0949, 0.0921,
0.0748` for timesteps 0.05 through 0.95. The corresponding factor correlations
were `0.0165, 0.0153, 0.0106, 0.0144, 0.0141, 0.0288, 0.0492, 0.0761,
0.0820, 0.0678`.

Activation norms pooled over module-token observations were:

| Group | Count | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| masked | 2,299,808 | 29.0485 | 26.2112 | 6.3604 | 48.6366 |
| unmasked | 2,287,712 | 30.5669 | 28.5338 | 7.8104 | 50.8957 |
| masked confidence bottom 20% | 461,216 | 27.0723 | 24.6344 | 5.8674 | 44.3493 |
| masked confidence top 20% | 463,680 | 30.4028 | 27.5260 | 7.1334 | 51.1527 |
| reveal masked | 25,984 | 31.0477 | 28.0845 | 7.8553 | 51.9936 |
| remain masked | 2,273,824 | 29.0257 | 26.1889 | 6.3447 | 48.5955 |

### Confidence-decile energy redistribution

Masked tokens were split deterministically into equal-count rank deciles. Mean
activation norm is the arithmetic mean over the 224 hooked module norms; energy
fractions sum squared activation over all those modules and features.

| Decile | Tokens | Mean confidence | Mean activation norm | Uniform energy | CGQ energy |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,027 | 0.0347 | 26.969 | 9.603% | 5.508% |
| 2 | 1,027 | 0.0483 | 27.172 | 9.411% | 5.714% |
| 3 | 1,027 | 0.0649 | 27.744 | 9.711% | 6.239% |
| 4 | 1,027 | 0.1013 | 28.643 | 10.165% | 7.205% |
| 5 | 1,027 | 0.1658 | 29.292 | 10.447% | 8.437% |
| 6 | 1,027 | 0.2744 | 29.732 | 10.538% | 9.980% |
| 7 | 1,027 | 0.4468 | 29.989 | 10.468% | 11.883% |
| 8 | 1,026 | 0.6874 | 30.127 | 10.193% | 13.905% |
| 9 | 1,026 | 0.9265 | 30.365 | 9.967% | 15.657% |
| 10 | 1,026 | 0.9959 | 30.456 | 9.497% | 15.473% |

CGQ therefore shifts masked-token activation energy from low-confidence to
high-confidence regions: deciles 1–2 fall from **19.014%** to **11.222%**, while
deciles 9–10 rise from **19.463%** to **31.130%**.

There were 10,267 masked and 10,213 unmasked positions. Masked tokens contributed
**53.035%** of unweighted energy and **51.600%** of CGQ-weighted energy; unmasked
tokens moved from **46.965%** to **48.400%**. Their mean confidence/factor pairs
were masked **0.37445/1.53594** and unmasked **0.84667/1.60661**.

### `A` versus `A_CGQ`

Module-level distributions over all 224 matrices:

| Metric | Mean | Median | p10 | p90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| feature Spearman | 0.99738 | 0.99794 | 0.99446 | 0.99991 | 0.99056 | 0.99998 |
| Pearson on log statistic | 0.99872 | 0.99902 | 0.99702 | 0.99995 | 0.99314 | 0.99999 |
| cosine similarity | 0.99744 | 0.99968 | 0.99713 | 0.99998 | 0.89950 | 1.00000 |
| relative L2 difference | 1.52314 | 1.54397 | 1.40257 | 1.63720 | 0.15480 | 1.84641 |
| CV(`A_CGQ/A`) | 0.02478 | 0.01419 | 0.01186 | 0.05413 | 0.00857 | 0.10620 |

The module mean of `q=A_CGQ/A` averaged **2.6091** (range **2.3405–2.7282**),
confirming that the large relative L2 difference is dominated by scale. Across
modules, CV quantiles were p01 **0.00916**, p10 **0.01186**, p50 **0.01419**,
p90 **0.05413**, and p99 **0.07393**.

### Localization

| Module type | Mean Spearman | Mean ratio CV | Max ratio CV | Mean relative L2 diff. |
|---|---:|---:|---:|---:|
| q_proj | 0.99855 | 0.01362 | 0.02348 | 1.51405 |
| k_proj | 0.99855 | 0.01362 | 0.02348 | 1.51405 |
| v_proj | 0.99855 | 0.01362 | 0.02348 | 1.51405 |
| attn_out | 0.99691 | 0.05210 | **0.10620** | 1.59321 |
| ff_proj | 0.99591 | 0.01456 | 0.02517 | 1.50675 |
| up_proj | 0.99591 | 0.01456 | 0.02517 | 1.50675 |
| ff_out | 0.99729 | 0.05138 | 0.08108 | 1.51309 |

Early/mid/late mean CVs were **0.02664/0.02102/0.02688**; mean Spearmans were
**0.99896/0.99743/0.99558**. The strongest single CV was layer 0 `attn_out`
(**0.10620**, Spearman 0.99801), while the lowest feature Spearman was layer 27
`attn_out` (**0.99056**, CV 0.06306). Layer-mean CV was highest in layers 31
(0.03714) and 30 (0.03369), followed by layers 1 (0.03156) and 0 (0.03010).
The effect is therefore module-localized and partly late-layer-amplified, but it
does not show a simple monotonic late-layer growth pattern.

### Counterfactual score ranking (diagnostic only)

No masks or threshold decisions were computed. For 128 evenly spaced output rows
in `attn_out` at layers 0, 15, and 31:

| Layer | Mean row score Spearman | Median | Mean abs. rank displacement | Median | p90 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.999518 | 0.999518 | 25.51 | 18 | 58 | 259 |
| 15 | 0.999801 | 0.999801 | 15.82 | 10 | 38 | 202 |
| 31 | 0.999444 | 0.999442 | 28.49 | 20 | 65 | 241 |

Ranks are among 4,096 input features. The score ordering changes measurably but
remains extremely similar overall.

## Interpretation

1. Confidence is not simply encoded by token activation magnitude: pooled and
   timestep-stratified correlations are weak, although within-state/module
   correlations are heterogeneous and sometimes moderate.
2. CGQ strongly redistributes token energy by confidence decile, but most of that
   redistribution collapses to a near-common feature multiplier inside the typical
   Linear module. This is why absolute scale changes are large while feature-rank
   correlations stay near one and median ratio CV is only 0.014.
3. The residual non-redundant signal is concentrated in `attn_out` and `ff_out`,
   especially several early attention and late output modules. This supports
   Outcome C rather than globally complementary Outcome B or completely redundant
   Outcome A.
4. The representative Wanda score diagnostics agree with the activation result:
   localized q variation causes small rank movements, not a wholesale score-order
   change.

## Speculation

- The `attn_out`/`ff_out` concentration may reflect mixing operations that expose
  confidence-conditioned directions more strongly than q/k/v and expansion
  projections. This experiment does not identify that mechanism.
- Similarity to the earlier late-layer/attention CGQ-SparseGPT pattern is descriptive
  only. It does not establish a shared cause or predict downstream benefit.

## One recommended next step

Run one **targeted analysis-only decomposition** for `attn_out` and `ff_out` in
layers 0, 1, 27, 30, and 31, attributing each feature's `q_j` deviation to fixed
masked/unmasked and masked-confidence-decile energy contributions. This would test
whether the localized CV comes from a stable confidence-conditioned feature subset
before considering any CGQ-Wanda pruning experiment.
