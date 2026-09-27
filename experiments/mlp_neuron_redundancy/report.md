# Does LLaDA Have Stable MLP-Neuron Redundancy?

## Scope

This was an analysis-only exact single-unit ablation feasibility diagnostic. It used no CGQ, token weighting, mask, pruning, joint ablation, parameter update, or downstream evaluation.

## Artifact-supported facts

### Frozen model, states, and candidates

- Model: `GSAI-ML/LLaDA-8B-Base`, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, bfloat16.
- States: frozen EXP-001/EXP-005 set, 8 sequences × 10 timesteps, length 256; SHA-256 `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`.
- Candidate seed: `20260905`, NumPy `default_rng`, uniform sampling without replacement independently per layer.
- Layers: 0, 4, 8, 12, 16, 20, 26, 31; 32 neurons per layer; 256 total.
- Exact IDs were frozen before measurement and are persisted in `sampled_neurons.json` (SHA-256 `c3f8d8825e81ec398e2fb450dcdb7dd896ab536ffe0fda881b788598e678fec7`).

### Exact intervention

The intervention occurs on the actual MLP intermediate tensor

`h = silu(ff_proj(ff_norm(x))) * up_proj(ff_norm(x))`

immediately before `ff_out`. One variant sets exactly one `h[...,k]` channel to zero. No parameter tensor is touched.

Sanity checks:

- ablated channel maximum absolute value: `0`;
- every other channel: bitwise identical;
- FP32 projection identity error for `Δy = -h_k ff_out[:,k]`: max absolute `1.19e-6`, relative `1.60e-4`;
- repeated dense forward: exact logits and loss;
- repeated identical ablation: exact loss and KL;
- model weight SHA-256 before collection, after collection, and after analysis: identical (`2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc`).

### Execution strategy and numerical-noise gate

Each state's dense full forward was run once while capturing prefix hidden states and intermediate activation energies for the eight tested layers. Candidate variants reused the exact captured prefix and ran the ordinary suffix. Every candidate forward contained two rows: row 0 was a no-ablation sham and row 1 removed one neuron. Thus no two neurons were ever removed in one variant.

The requested candidate batch sizes were benchmarked. Larger candidate batches were rejected because even sham correction did not reproduce the one-candidate result:

| Candidates + sham | Time | Peak allocation | Raw sham ΔL | First-candidate corrected-ΔL difference vs batch-1 |
|---:|---:|---:|---:|---:|
| 1 + 1 | 0.0375 s | 16.66 GB | -0.000723 | 0 |
| 4 + 1 | 0.0707 s | 16.94 GB | +0.007488 | 0.015599 |
| 8 + 1 | 0.1248 s | 17.32 GB | +0.021182 | 0.015608 |
| 16 + 1 | 0.2227 s | 18.08 GB | +0.021182 | 0.015608 |

The full run therefore used one candidate plus one sham per suffix forward. Both quantities were stored:

- raw requested reference: `L_ablated(batch path) - L_dense(batch=1)`;
- primary noise-controlled causal contrast: `L_ablated - L_sham` inside the identical two-row kernel path.

The raw dense-reference measurement is measurement-limited:

- absolute sham drift median / p99 / max: `0.002354 / 0.015154 / 0.023393`;
- corrected exact effect absolute median / p99 / max: `0.002043 / 0.012055 / 0.026776`;
- 99.66% of corrected state-level effects are at or below the raw-sham p99;
- median corrected effect / raw-sham p99: `0.135`.

However, the sham was exactly repeatable across all 32 same-shape candidate forwards for every state/layer: maximum within-state/layer sham loss range `0`, KL range `0`. The same-batch subtraction therefore isolates a deterministic intervention contrast. All sensitivity results below use this corrected contrast; raw values remain persisted and are not silently interpreted as causal effects.

### Sensitivity heterogeneity

Pooled across the 256 neurons:

| Metric | min | p10 | median | p90 | max | CV | Gini | p90/p10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_pos | .000280 | .000500 | .001549 | .002044 | .002565 | .367 | .205 | 4.09 |
| D_abs | .000632 | .001094 | .002799 | .003981 | .004568 | .346 | .194 | 3.64 |

The pooled spread is partly a layer-scale effect. Within-layer heterogeneity is more modest:

| Layer | D_pos p10 | median | p90 | CV | Gini | p90/p10 | D_abs p10 | median | p90 | CV | Gini | p90/p10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .001828 | .002083 | .002248 | .099 | .054 | 1.23 | .003716 | .004008 | .004353 | .061 | .035 | 1.17 |
| 4 | .001481 | .001803 | .002043 | .128 | .071 | 1.38 | .003467 | .003807 | .004213 | .075 | .043 | 1.22 |
| 8 | .001457 | .001722 | .001946 | .117 | .066 | 1.34 | .003061 | .003284 | .003449 | .064 | .035 | 1.13 |
| 12 | .001485 | .001852 | .002022 | .116 | .065 | 1.36 | .002630 | .002866 | .003203 | .080 | .045 | 1.22 |
| 16 | .001292 | .001554 | .001989 | .156 | .088 | 1.54 | .002403 | .002646 | .003050 | .100 | .057 | 1.27 |
| 20 | .000955 | .001203 | .001475 | .159 | .089 | 1.55 | .002192 | .002398 | .002736 | .087 | .049 | 1.25 |
| 26 | .000854 | .001104 | .001269 | .200 | .101 | 1.49 | .001803 | .002086 | .002388 | .116 | .062 | 1.32 |
| 31 | .000297 | .000373 | .000594 | .311 | .159 | 2.00 | .000669 | .000833 | .001223 | .323 | .158 | 1.83 |

Sensitivity is not flat, especially in layer 31, but most layers show only about 1.1–1.5× p90/p10 separation after controlling for layer scale.

### Sequence stability

Pairwise sequence rankings are poorly stable. Pooled after converting candidates to within-layer ranks:

| Metric | Pairwise Spearman mean / median / p10–p90 | bottom-10% overlap / Jaccard | bottom-20% overlap / Jaccard |
|---|---:|---:|---:|
| D_pos | .047 / .046 / -.016–.118 | .117 / .063 | .220 / .124 |
| D_abs | .066 / .068 / .005–.127 | .107 / .057 | .217 / .122 |

Chance expectations are approximately 10% overlap and 0.053 Jaccard for a bottom decile, and 20% overlap and 0.111 Jaccard for a bottom quintile. Observed sequence stability is therefore close to chance. Per-layer median pairwise Spearman ranges from `-0.034` to `0.172` for D_pos and `-0.045` to `0.369` for D_abs. Split-half correlations are inconsistent and often negative; even layer 31's higher D_abs split-half correlation (`0.568`) has bottom-10% overlap `0/4`.

### Timestep stability

Pooled within-layer timestep rankings are likewise unstable:

| Metric | Pairwise Spearman mean / median / p10–p90 | bottom-10% overlap / Jaccard | bottom-20% overlap / Jaccard |
|---|---:|---:|---:|
| D_pos | .035 / .043 / -.041–.100 | .101 / .054 | .211 / .119 |
| D_abs | .041 / .032 / -.032–.125 | .098 / .053 | .206 / .116 |

Early/middle/late D_abs rank correlations further show no common trajectory-independent low tail:

| Layer | early–mid ρ | mid–late ρ | early–late ρ | early–late bottom-10% Jaccard |
|---:|---:|---:|---:|---:|
| 0 | -.372 | -.174 | .119 | .143 |
| 4 | .111 | -.074 | -.202 | 0 |
| 8 | -.306 | -.297 | .065 | .333 |
| 12 | .056 | .154 | -.007 | .143 |
| 16 | .304 | .201 | .013 | 0 |
| 20 | .028 | .102 | .084 | 0 |
| 26 | .074 | .159 | .233 | .333 |
| 31 | .439 | .414 | .651 | .333 |

Layer 31 has some trajectory-level ordering, but its low set is not stable across sequences and each per-layer bottom decile contains only four sampled neurons. It cannot support a global static redundancy claim.

### Masked-token KL

Mean same-batch sham-relative masked-token KL across neurons has median `0.000477`, p90 `0.000686`, and max `0.000881`. Its pooled Spearman with D_abs is `0.9555`; neurons with small exact loss effect generally also have small predictive-distribution change. No large population with low D_abs but high KL was observed.

### Held-out baseline predictability

Scores were computed on sequences 0–3 and tested against exact ablation on 4–7, then reversed. Structured-Wanda here means only `||ff_out[:,k]||₂ sqrt(A_h[k])`; the earlier three-matrix group score was not used. Uniform gradient is the pre-existing `mean |d_U|` gate score. Pooled comparisons use within-layer ranks.

| Direction | Baseline | D_pos ρ | D_abs ρ | D_pos bottom-10 overlap | D_abs bottom-10 overlap |
|---|---|---:|---:|---:|---:|
| A→B | magnitude | -.054 | -.099 | .156 | .094 |
| A→B | structured-Wanda | .099 | .025 | .063 | .156 |
| A→B | gradient ABS | .086 | .034 | .094 | .156 |
| B→A | magnitude | .063 | -.065 | .125 | .156 |
| B→A | structured-Wanda | .013 | .144 | .063 | .219 |
| B→A | gradient ABS | .028 | .109 | .063 | .188 |

Most per-layer correlations are similarly weak and sign-variable. Layer 31 is the exception for held-out D_abs: average two-direction Spearman is `0.610` for structured-Wanda and `0.562` for gradient ABS, while magnitude is `-0.502`. Full per-layer Kendall, top/bottom 10/20%, and both split directions are retained in `artifact_manifest.json`. Because exact low-sensitivity identity itself is unstable, weak baseline recovery is not criterion headroom.

### First-order fidelity

At the state level, existing uniform gate derivative versus exact same-batch removal gives:

- Spearman `0.0796`, Pearson `0.1083`;
- sign agreement `52.56%`;
- MAE `0.002751`;
- median safe relative error `0.999`.

After ABS aggregation by neuron, Spearman is `0.1478`, Kendall `0.1078`, bottom-10% overlap `18.75%`, and bottom-20% overlap `25%`. A first-order gate derivative at `z=1` is therefore a poor approximation to complete removal at `z=0` in this sample.

## Interpretation

### Primary classification: Outcome C — heterogeneous but unstable sensitivity

Exact same-batch intervention effects are measurably heterogeneous, so the landscape is not simply flat. But the lower-sensitivity identity changes almost at random across sequences and timesteps: pooled pairwise correlations are near zero and bottom-set overlaps are approximately chance. This fails the central stability condition for a static structured pruning method.

The experiment therefore finds functional specialization without a stable, timestep-independent population that can presently be called redundant. Existing baselines are weak predictors overall, but that does not constitute a useful failure mode when the exact reference itself is not stable.

Single-neuron results also say nothing about interactions among jointly removed neurons.

## Speculation

Layer 31 may contain a more structured low-sensitivity landscape than earlier layers, but 32 random candidates and four-neuron decile sets are too small to distinguish a real layer-specific population from sampling noise. The strong batch-shape drift also means future exact intervention work must preserve an identical sham/kernel path.

## Exactly one recommended next step

Repeat this analysis-only random sampling with a larger preregistered candidate count in layer 31 only, retaining identical two-row sham-controlled forwards, to determine whether its apparent trajectory stability survives sampling; do not run joint ablation or structured pruning unless that stability test passes.
