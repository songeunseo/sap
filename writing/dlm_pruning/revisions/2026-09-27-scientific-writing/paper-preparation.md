# Conditional Response Preservation for Pruning Diffusion Language Models

**Internal research draft.** Preparation copy; full results are pending.

## Abstract

Pruning a masked diffusion language model requires choosing a static set of weights that remains useful as the visible context changes. We study an allocation criterion that combines the error at individual partially revealed states with the error in the model's response between states. The response is measured for the same unresolved query using a scalar gold-versus-rest log-odds readout. A multiscale graph connects states along nested gold-reveal chains, and a fixed pruning backend converts finite block probes into an allocation with an exact global budget. We compare the endpoint-only criterion with natural-chain and cross-chain response criteria using the same states, weight rankings, surviving weights, and allocation procedure. The study distinguishes the contribution of response matching from the contribution of the state bank and tests whether the selected connections matter. **[PENDING: the full comparison is still running.]** The current scope is one fixed LLaDA-8B allocation family and GSM8K protocol; broader practical superiority and generalization remain open.

## 1. Introduction

Masked diffusion language models predict missing tokens from partially observed sequences. LLaDA implements this approach through a forward masking process and a reverse generation process parameterized by a Transformer [Nie et al., 2025](https://arxiv.org/abs/2502.09992v3). A static pruning mask must serve the model across these changing inputs. This raises a calibration question: which properties of the dense model should an allocation preserve across a family of partially revealed contexts?

An endpoint criterion measures how much a sparse model differs from the dense model at each sampled state. A response criterion additionally penalizes errors in the change between two related states. These objectives express different preferences when a sparse model cannot reproduce every endpoint exactly. The response term introduces no new information in the perfect endpoint-matching limit. Its potential value is in deciding which errors to penalize more strongly under a finite pruning budget.

Our working hypothesis is that comparing the same query across nested context reveals can help allocate sparsity. Testing this hypothesis requires more than showing that a combined objective beats a historical uniform baseline. The endpoint-only control must use the same states and backend. Controls that change the connections while retaining the endpoint bank help examine whether a particular pairing contributes to the observed effect.

We therefore study four frozen allocations: endpoint-only A, natural-chain Multi, Cross, and scale-matched CrossMatched. Their downstream comparison is accompanied by diagnostics on fresh calibration spans. The intended contribution is a tested connection between DLM calibration structure and static pruning allocation. Establishing that contribution depends on the evidence below. We do not claim a new general optimizer, a task-accuracy theorem, or an established advantage over all pruning methods.

## 2. Related work

Wanda uses weight magnitude and input activation to rank weights per output without retraining or weight updates [Sun et al., 2024](https://arxiv.org/abs/2306.11695v3). We hold a previously constructed Wanda ranking fixed and study how much sparsity to allocate to each block. This isolates the allocation criterion within the chosen weight-ordering family.

Nonuniform compression already has substantial prior work. EvoPress searches compression profiles and documents limitations of treating layer errors as independent [Sieberling et al., 2025](https://proceedings.mlr.press/v267/sieberling25a.html). Our rank-based allocation is a heuristic. Full sparse-model measurements are required to evaluate its behavior.

Preserving responses alongside values also has precedent. Sobolev Training incorporates target derivatives in function approximation and distillation [Czarnecki et al., 2017](https://arxiv.org/abs/1706.04859v3). Our response term uses finite differences between masked contexts. Its proposed role is specific to the calibration construction and pruning experiment, rather than a claim that response matching itself is new. **[Literature review incomplete: closest DLM compression and response-preservation methods must be compared before a novelty claim.]**

## 3. Method

### 3.1 States and readout

Let D be the dense model and M a static sparse model. For a clean sequence, choose a fixed query set Q before sampling context reveals. Queries remain masked throughout the chain. Each other position receives an independent uniform random variable U. At visibility probability p_j, reveal its gold token if U is at most p_j. Sharing U across phases produces nested visible sets. Visibility probability refers to eligible context positions, not the realized visible fraction of the whole sequence.

The implemented bank has eight sequences of length 256, two chains per sequence, eight queries per sequence, and eight states per chain. Visibility probabilities are equally spaced in logit space from 0.05 to 0.95. Unchanged states are retained. These are gold-context calibration chains; they are not sampled model-generated decoding trajectories.

For query q with fixed gold token y, define

\[
f_X(x,q)=\log\frac{p_X(y\mid x,q)}{1-p_X(y\mid x,q)},\qquad
e_{c,j,q}=f_M(x_{c,j},q)-f_D(x_{c,j},q).
\]

The scalar readout measures gold-versus-rest confidence. It does not preserve the ordering among incorrect tokens or the full output distribution.

### 3.2 Endpoint and response objectives

The endpoint objective A averages e² equally over queries, phases, chains, and sequences. For d in {1,2,4}, connect phase j to j+d whenever bit d of j is zero. Each scale forms four disjoint edges on the eight phases. Define

\[
C_d=\mathbb E_{\mathrm{sequence},c,q}
\left[\frac14\sum_{j:\,j\,\&\,d=0}(e_{c,j+d,q}-e_{c,j,q})^2\right],
\qquad C_{\mathrm{natural}}=\frac{C_1+C_2+C_4}{3}.
\]

The natural objective is L_Multi=A+C_natural. Every state participates once per scale, so the endpoint exposure is balanced across scales. Since (a-b)²≤2a²+2b², each matching gives C_d≤4A, hence 0≤C_natural≤4A. This bounds the penalty relative to endpoint error; it gives no downstream accuracy guarantee.

The cross control uses the same sequence, query and phase pairs, but connects different chains in both directions. It retains accidentally nested or identical states. The objectives are L_Cross=A+C_cross and L_CrossMatched=A+βC_cross, with β=0.932319907797 fixed from the uniform model's calibration ratio C_natural/C_cross. This scalar match at the uniform reference does not match every candidate's marginal response cost or isolate semantic information.

### 3.3 Static allocation

All candidates retain the same native sparse-prefix Wanda rankings and surviving weight values. In a jointly sparse uniform-50% model, each of the 32 blocks is probed at 48% and 52% sparsity. Seven projections share the block's proposed rate. For objective L, the signed finite cost is

\[
g_\ell=\frac{L(M_{\ell,52})-L(M_{\ell,48})}
{N_{\mathrm{pruned}}(M_{\ell,52})-N_{\mathrm{pruned}}(M_{\ell,48})}.
\]

Average ranks map these costs to rates between 45% and 55%. Grouped row-count rounding enforces the exact budget across 224 projections: 3,489,660,928 of 6,979,321,856 prunable weights are removed. The rounding algorithm enforces the budget; it does not globally minimize L. Probe extrapolation and interactions between simultaneous block changes remain limitations.

The same cached probe outputs support all four objectives. Calculating additional response terms from these outputs requires scalar arithmetic, but obtaining the probes and calibration bank still incurs model inference cost. We therefore distinguish reused artifacts from newly executed work.

## 4. Experimental setup

The evaluated checkpoint is GSAI-ML/LLaDA-8B-Base, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, in BF16. GSM8K uses the frozen five-shot requests, 256 denoising steps, a 256-token generation limit, block length 256, temperature zero, and strict-match grading. Seeds, prompts and physical model identities are fixed in the source config. The study does not optimize the masks on the primary evaluation answers.

All four candidates generate answers for all 1,319 questions. The 200 questions examined in the earlier development and separate100 stages are identified by frozen IDs. The primary analysis uses the remaining 1,119 questions. This exclusion does not establish that these questions were never exposed anywhere in the project's history.

The three primary contrasts are Multi−A, Multi−Cross, and Multi−CrossMatched. Exact paired McNemar tests use Holm correction across this family. Paired bootstrap intervals use 10,000 draws with seed 20260927 and remain unadjusted. Full1319 and exposed200 scores are descriptive; they do not replace the primary sample. No stopping rule is based on intermediate accuracy.

Fresh fidelity diagnostics use eight separate spans, 128 states, and the same readout definitions. Their uncertainty unit is the span. The recorded response-sign metric evaluates all 28 within-chain phase pairs with absolute dense response above 1e−6; it is not the sign metric restricted to the 12 multiscale edges or the cross-chain edges.

## 5. Results

### 5.1 Prior development and separate100 comparisons

| Method | Development100 correct | Separate100 correct |
|---|---:|---:|
| Multi | 63/100 | 59/100 |
| A | 60/100 | 58/100 |
| Uniform | 54/100 | 52/100 |

| Contrast | Gain/loss | Net | Exact p | Holm p |
|---|---:|---:|---:|---:|
| Multi−A | 6/5 | +1 | 1 | 1 |
| Multi−Uniform | 13/6 | +7 | 0.16707 | 0.50121 |
| A−Uniform | 13/7 | +6 | 0.26318 | 0.52635 |

The separate100 sample was fixed before the current method selection and is disjoint from development100. It is not certified unseen across the entire project. The Multi−A paired difference is small and does not establish an incremental C effect. Failure to reject a difference also does not establish equivalence. These observations motivate the larger fixed comparison without determining its outcome.

### 5.2 Full comparison

**[미측정 / PENDING: full-run completion and verification required.]**

**[미측정 / PENDING: full-run completion and verification required.]**

**[미측정 / PENDING: full-run completion and verification required.]**

### 5.3 Fidelity and cost

**[미측정 / PENDING: full-run completion and verification required.]**

Diagnostic fidelity and task capability measure different properties. A lower response loss is not itself evidence of improved answer accuracy or a causal explanation for any observed gain.

**[미측정 / PENDING: full-run completion and verification required.]**

## 6. Discussion and limitations

The design separates three questions: whether adding C helps relative to A, whether the chosen context connections matter relative to the tested controls, and whether the full method is competitive in practice. The four-arm study directly addresses the first two under one frozen setup. It does not by itself establish the third. Matched competitive baselines, calibration cost, and separately designed generalization checks remain necessary for a broad method claim.

The bank uses gold context and a scalar readout. Generated contexts and changes among non-gold alternatives may behave differently. The allocation procedure also extrapolates local probes to a jointly modified mask. Thus, an adverse result constrains this objective–bank–allocator configuration; it does not rule out all uses of conditional response. Conversely, a positive result does not prove a universal need for natural reveal chains.

The current study uses one calibration realization, one checkpoint, one sparsity level, and one downstream protocol. More benchmark questions improve the precision of that task comparison but do not substitute for independent calibration replications. No measured inference speedup follows from the presence of unstructured zero weights alone.

## 7. Conclusion

This study makes conditional response preservation a concrete, testable allocation criterion for static DLM pruning. Its main empirical question is the added value of response matching beyond endpoint matching under a shared bank and backend. The verified comparisons above determine the scope of the result. Future experiments should target the most consequential unresolved claim in the accompanying claim map.

## Evidence and reproducibility

All displayed empirical tables are generated from verified result files. `closeout.json` records source hashes, JSON locations, per-question verification, historical reproduction and attempt-level costs. The source experiment, prior outputs and excluded IDs are preserved. This internal draft retains visible gaps in literature coverage and generalization; it is not a submission version.
