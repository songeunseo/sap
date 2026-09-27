# Fisher-Geometry Wanda Prototype 1 — Final Report

## Verdict

**SUCCESS under the preregistered rule.** Gate 1 and Gate 2 passed, and Ours-FG answered 694/1319 GSM8K examples correctly versus Standard Wanda's 677/1319. The primary improvement is +17 answers (+1.289 percentage points). The paired two-sided exact McNemar/binomial p-value is 0.303, so this is a preregistered directional success, not strong evidence that the downstream gain is precisely estimated.

## 1. Prototype definition and dense-information audit

The only modified matrix is `model.transformer.blocks[31].ff_out`, with

\[
W\in\mathbb{R}^{4096\times12288},\qquad
d_t^{(ij)}=-w_{ij}x_t[j]e_i.
\]

The implemented per-weight score is

\[
G_{ij}=\frac{1}{N}\sum_{t\in\text{masked calibration tokens}}
x_t[j]^2\kappa_t[i],\qquad
S^{FG}_{ij}=|w_{ij}|\sqrt{G_{ij}},
\]

where

\[
\kappa_t[i]=e_i^T J_t^T F_tJ_te_i.
\]

The inspected dense readout is `block31 residual -> RMSLayerNorm ln_f -> bias-free LM-head ff_out -> logits -> softmax`. RMSNorm converts its variance calculation to FP32; the loaded configuration passed the implementation guards for bias-free final norm/head and `scale_logits=False`.

For a token hidden state `h`, RMS scale `r`, norm weight `gamma`, and LM-head row distribution induced by dense `p`, the diagonal was computed without materializing a Jacobian or Fisher matrix:

\[
a_i[v]=\frac{\gamma_i}{r}W_{head}[v,i]
-\frac{h_i}{d r^3}g_v,
\qquad
g_v=r z_v,
\qquad
\kappa_i=\operatorname{Var}_{v\sim p}(a_i[v]).
\]

Vocabulary rows were streamed in chunks. Continuous logits were FP32; softmax, moment calculation, and `G` accumulation were FP64; persisted scores were FP32.

Information audit:

- Dense `h_t`, dense probabilities, final norm and LM-head weights, and `x_t` are all available before pruning.
- `kappa` is computed entirely from dense-model readout geometry.
- No pruned logits, causal KL result, fitted coefficient, mixture weight, temperature, or post-pruning quantity enters `S_FG`.
- The generated mask and Gate-2/Gate-3 outcomes are post-score evaluation artifacts only.

## 2. Calibration and numerical gates

Geometry calibration reused the frozen EXP-001 8 sequences × 10 timesteps = 80 states, with 10,267 masked-token observations. Historical state digest: `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`. Manifest SHA-256: `c494feafc0bb559eee28eb3e996040236d8a44047b3f5b9d0e54f42e78b6a694`.

The kappa formula was checked on 48 preregistered state/token/basis cases against direct directional Fisher evaluation:

- maximum relative error: `4.183e-4`;
- maximum absolute error: `4.434e-12`;
- cosine: `0.99999999999994`;
- Pearson: `0.999999999999723`;
- gate result: PASS.

Across the full collection, minimum kappa was `3.162e-14`; negative count and materially negative count were both zero. Dense model SHA-256 was unchanged before/after collection: `2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc`.

## 3. Gate 1 — score and mask change

The exact comparison used the repository's sequential layer-wise Standard Wanda calibration, not the earlier isolated-module approximation.

| Comparison | Score Spearman | Global mask XOR | Median row XOR | p10–p90 row XOR | Pruned-set Jaccard |
|---|---:|---:|---:|---:|---:|
| FG vs Standard Wanda | 0.92482 | 10.782% | 10.775% | 10.417–11.165% | 0.80534 |
| FG vs matched DLM-Wanda | 0.98896 | 3.160% | 3.158% | 2.897–3.418% | 0.93873 |
| DLM-Wanda vs Standard Wanda | 0.94806 | 9.109% | 9.115% | 8.773–9.440% | 0.83303 |

Gate-1 thresholds were global XOR ≥2% and median row XOR ≥1%. Both were exceeded. Every row pruned exactly 6,144/12,288 weights.

**Gate 1: PASS.** Fisher geometry survived aggregation into a materially different static mask.

## 4. Gate 2 — independent module-only failure correction

The independent evaluation set was frozen before functional evaluation: 8 disjoint WikiText-2 spans (sequence IDs 16–23) × timesteps 0.1/0.3/0.5/0.7/0.9 = 40 states. Its state digest is `9ac2f11686fbb2696c685f4474dabcac5edf9a34efeb0f6e8aa5cbde8c6bd4cc`; manifest SHA-256 is `f0f22b49fda3bdab9b63d1d977b1e56cb2b68af2e3f3c215740906908105bed5`.

All four variants used one ordinary full-forward batch `[sham, Wanda, DLMW, FG]`. Same-path no-op row spread, repeat-logit difference, and no-op sham difference were all exactly zero. External batch-1 versus batched sham KL was nonzero (median `7.94e-4`, p99 `4.46e-3`, max `4.91e-3`), confirming why same-path comparisons were necessary.

| Condition | Mean masked KL | Mean loss delta | Top-1 agreement | Confidence MAE | Local reconstruction |
|---|---:|---:|---:|---:|---:|
| Standard Wanda | 0.027096 | 0.025813 | 0.911912 | 0.022608 | 0.006850 |
| Matched DLM-Wanda | 0.007972 | 0.006185 | 0.941036 | 0.015059 | 0.005890 |
| Fisher-Geometry | **0.007816** | 0.006990 | **0.941675** | 0.015518 | 0.006740 |

The paired mean KL improvement over Standard Wanda was `0.019280`. Its receiver-sequence cluster-bootstrap 95% CI was `[0.011089, 0.028224]`. Improvement was positive for 5/5 timestep means and 8/8 sequence means.

FG also had a lower mean KL than matched DLM-Wanda by `0.000156`, satisfying the preregistered mean criterion, but that difference was not robustly separated from zero: bootstrap CI `[-0.000247, 0.000643]`. Therefore most of the module-only gain is attributable to moving from clean to matched DLM calibration; Fisher geometry adds only a small incremental mean improvement in this gate.

**Gate 2: PASS** under all frozen criteria. This claim is restricted to module-only `block31.ff_out`; no new 224-module aggregate failure map was measured.

## 5. Gate 3 — full-model prototype and downstream results

The full-model prototype kept the exact repository Standard Wanda 50% mask on 223 matrices and replaced only `block_31.ff_out` with the FG 50% mask. Overall mask hashes:

- Standard Wanda: `3273fbbe3aaeeb1dc66b0c436e6af7537e07d1bb461667354f3da00a879fc484` (exact EXP-002 match);
- Ours-FG: `0dfe2f975bde19170736d11f3d2f9578cefb7baadb9751ff05b3d3c4e90fc6ee`;
- changed modules: exactly `block_31.ff_out`;
- total and per-row sparsity: exactly 50% for Wanda and Ours-FG.

### GSM8K — primary

Configuration: full 1,319 examples, 5-shot, strict exact match, temperature 0, generation length/block length/steps all 256. Dense, Wanda, and SparseGPT were reused from EXP-002 only after exact protocol and per-example compatibility checks; Ours-FG was newly evaluated.

| Method | Correct | Accuracy |
|---|---:|---:|
| Dense | 938/1319 | 71.114% |
| Standard Wanda | 677/1319 | 51.327% |
| SparseGPT | 584/1319 | 44.276% |
| Ours-FG | **694/1319** | **52.616%** |

Ours-FG versus Standard Wanda:

- both correct: 565;
- FG only: 129;
- Wanda only: 112;
- both wrong: 513;
- difference: +17 correct, +1.289 percentage points;
- two-sided exact paired p-value: `0.3027`.

Ours-FG versus SparseGPT was +110 correct (+8.340 points), paired p=`3.84e-9`.

### WinoGrande — secondary

All four methods were rerun because historical records did not pin an exactly matching model revision. Configuration: full 1,267 examples, 5-shot, batch 8, MC samples 128, CFG 0.

| Method | Correct | Accuracy |
|---|---:|---:|
| Dense | 945/1267 | 74.586% |
| Standard Wanda | 885/1267 | 69.850% |
| SparseGPT | 885/1267 | 69.850% |
| Ours-FG | **890/1267** | **70.245%** |

Ours-FG versus Standard Wanda: 868 both correct, 22 FG-only, 17 Wanda-only, 360 both wrong; +5 correct (+0.395 points), paired p=`0.5224`. Versus SparseGPT: +5 correct, paired p=`0.7497`.

All persisted prediction hashes and correctness counts reproduce their summaries. Every evaluated model's prunable-weight SHA-256 was identical before and after its evaluation.

## 6. Interpretation

### Artifact-supported facts

- FG materially changed the exact repository Wanda mask at the target module.
- On independent module-only states, FG reduced masked KL strongly relative to Standard Wanda and slightly in mean relative to matched DLM-Wanda.
- In the full-model setting, changing only `block31.ff_out` improved observed accuracy over Standard Wanda on both preregistered tasks: +17 GSM8K answers and +5 WinoGrande answers.
- The frozen primary rule required strictly more correct GSM8K answers; Ours-FG met it.

### Statistical interpretation

The prototype passes the registered decision rule, but the downstream increment over Wanda is modest and the paired uncertainty is wide (`p=0.303` on primary GSM8K; `p=0.522` on WinoGrande). The result demonstrates a positive realized translation from mechanism to pruning decisions to downstream outcomes, but does not yet establish a precisely estimated or broadly general improvement.

### Speculation boundary

These data do not show that Fisher geometry will generalize to other layers, module types, model sizes, seeds, or sparsities. They also do not establish that the Fisher factor alone caused all of the Gate-2 improvement, since the matched DLM calibration control already captured most of that reduction.

## Final verdict and one next action

**Final verdict: SUCCESS.** This is success under the frozen prototype rule, with appropriately cautious statistical interpretation.

**Recommended next action (one only):** freeze this exact score, calibration, target module, and 50% mask construction unchanged, then run one preregistered independent downstream replication to estimate whether the small Ours-FG versus Wanda gain is reproducible; do not tune or introduce a second prototype.

## Key artifacts

- `preregistered.json` — calibration, precision, and Gate-1/2 rules
- `geometry_calibration_manifest.json` — frozen 80-state geometry calibration
- `kappa_validation.json` — direct numerical validation
- `collection_manifest.json` — sufficient-statistic and score hashes
- `gate1_repository_exact.json` — exact sequential-Wanda comparison
- `independent_evaluation_manifest.json` — frozen Gate-2 states
- `gate2_per_state.json`, `gate2.json` — functional outcomes and decision
- `gate3_preregistered.json` — downstream protocol and success rule
- `gate3_mask_manifest.json` — full-model mask accounting
- `gate3/gsm8k.json`, `gate3/winogrande.json` — downstream outcomes
- `verdict.json` — final preregistered verdict
