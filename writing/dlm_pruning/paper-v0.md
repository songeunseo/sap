# Conditional Response Preservation for Pruning Diffusion Language Models

**Internal research draft.** Paper v0; full results passed the documented CPU verification. Scientific review remains open.

## Abstract

A static pruning mask for a diffusion language model must support predictions as visible context changes. Two sparse models can have the same average prediction error while distorting these changes differently. We study whether preserving the dense model's response to context reveals helps allocate a fixed sparsity budget. The criterion combines prediction error at individual masked states with error in the change of gold-token log-odds between paired states. Pairs follow nested context reveals at three scales. A fixed weight ranking and allocation procedure turn this criterion into a static mask at exactly 50% sparsity. Four allocations share the same calibration states and pruning backend: endpoint-only A, natural-chain Multi, Cross, and scale-matched CrossMatched. Their comparison tests both the added response term and the choice of connections. On the primary 1,119 GSM8K questions, Multi differs from A by -0.894 percentage points (unadjusted 95% paired bootstrap interval [-2.949, 1.162]; Holm-adjusted p=1). The three controlled comparisons are interpreted jointly in Section 5. The empirical scope is one LLaDA-8B checkpoint and GSM8K protocol.

## 1. Introduction

Pruning removes weights from a language model while seeking to retain its ability to use context. In a masked diffusion language model, that context changes as tokens become visible during generation. LLaDA predicts masked tokens from partially observed text through a Transformer-based reverse process [Nie et al., 2025](https://arxiv.org/abs/2502.09992v3). A single static pruning mask must therefore support predictions at many stages of context availability. The calibration problem is to choose which aspects of the dense model's behavior to preserve with the remaining weights.

Matching predictions at sampled states is a natural starting point. Under a limited weight budget, however, exact matching may be unattainable. An average squared error summarizes the size of the remaining discrepancies but does not distinguish all patterns of error across related states. For example, a constant confidence offset cancels when measuring a response to new context. An offset that changes with context can alter that response, even when its magnitude at every sampled state is identical.

**The central insight is that equal prediction error can conceal different errors in the response to context.** Figure 1 illustrates this with two constructed residual sequences. Both have average squared error one. The constant residual preserves every change in confidence; the changing residual distorts some of those changes. This is a distinction between error patterns, not empirical evidence that one pattern gives better generated answers. With perfect prediction matching, both kinds of error vanish.

![Constructed residuals with equal endpoint error and different response errors.](/home/tmluser1/sap/writing/dlm_pruning/figures/response-example.png)

**Figure 1. Equal prediction error can coexist with different response errors.** This constructed example uses eight increasingly revealed contexts for the same masked query. It contains no model measurements. Left: residuals are sparse minus dense gold-token log-odds, in natural-log units. A constant residual of +1 and the sequence (+1,+1,−1,−1,+1,+1,−1,−1) both give endpoint loss A=1. Right: C_d averages squared residual differences over the phase pairs defined in Section 3.2. Only the changing residual has C_2=4, so its mean response loss is 4/3 and its combined loss is 7/3. The constant residual has zero response loss and combined loss one. The example also shows why additional scales can distinguish errors that the d=1 pairs treat equally.

This observation suggests preserving how the dense model changes its prediction when more context becomes available. We keep a query masked, reveal nested subsets of its surrounding gold context, and compare the dense and sparse models' changes in confidence for that query. Pairing states at several distances measures responses to smaller and larger context updates. These gold-reveal chains provide a controlled calibration construction; their relevance to generated text is an empirical question.

Our working hypothesis is that adding this response criterion to prediction matching improves downstream accuracy under a fixed pruning backend and budget. We test it with four allocations made from the same state bank and cached block probes. A matches predictions at individual states. Multi adds responses along nested reveal chains. Cross and CrossMatched change the connections between states, with the latter matching the response-loss scale at a uniform reference mask. Multi−A tests the incremental value of the response term; the other two contrasts test the chosen connections against these controls. The resulting claim depends on paired GSM8K outcomes and uncertainty, alongside fidelity measurements on fresh calibration spans.

## 2. Related work

### 2.1 Choosing weights and allocating sparsity

Wanda ranks weights using their magnitudes and input activations, independently within each output row [Sun et al., 2024](https://arxiv.org/abs/2306.11695v3). This provides a practical ordering of candidate removals. EvoPress searches nonuniform compression profiles and emphasizes that errors from compressing different layers need not combine independently [Sieberling et al., 2025](https://proceedings.mlr.press/v267/sieberling25a.html). These works separate two relevant choices: which weights a local pruner removes, and how much compression each part of a model receives. Here, the Wanda ordering and surviving weight values are fixed. The tested criterion changes block sparsity through a shared allocation heuristic, whose final masks require full-model evaluation.

### 2.2 Calibration for diffusion language models

DLM compression already uses information about masked states and generation dynamics. [Sink-Aware Pruning](https://arxiv.org/html/2602.17664) averages soft sink scores over noised calibration inputs, downweights the corresponding activations, and supplies those activations to Wanda or SparseGPT (Section 3.2). It is a close weight-pruning baseline because the DLM-specific signal changes weight importance. [Quant-dLLM](https://proceedings.iclr.cc/paper_files/paper/2026/file/805da7ef883245cb35e012cc179a5f6f-Paper-Conference.pdf) uses partially visible calibration inputs and allocates mixed precision under an average two-bit budget (Sections 3.2–3.4). [FAIR-Calib](https://arxiv.org/html/2606.06547) probes a teacher to estimate frontier and reliability weights, then uses them in hidden-state reconstruction for quantization (Section 3). These methods establish masked calibration and behavior-informed compression as existing approaches.

The present experiment asks a narrower question: does coupling the errors of the same unresolved query across selected contexts improve a static weight allocation when the states themselves are held fixed? A uses exactly the same states as Multi, so their comparison can separate the added response criterion from changes in calibration coverage. The cross-chain controls further test the role of the connections. Comparisons with strong pruning baselines remain necessary to assess practical value beyond this controlled allocation study.

### 2.3 Preserving responses as well as predictions

[Sobolev Training](https://arxiv.org/abs/1706.04859v3) incorporates target derivatives alongside function values in approximation and distillation. In pruned image diffusion models, [2ndMatch](https://arxiv.org/html/2506.05398) uses projected measurements of the Jacobian metric JᵀJ to preserve sensitivity during finetuning (Section 4). Both precedents motivate examining behavior beyond isolated predictions. Our criterion measures finite changes in a scalar output under discrete context reveals and uses those measurements to select a static sparsity allocation. Its contribution must be assessed at that level: the pairing construction, the controlled comparison, and its observed consequences. A broader review of relational distillation and DLM compression remains open before any priority claim.

## 3. Method

The procedure has three steps: construct related masked states, score prediction and response errors on those states, and use block probes to allocate a fixed number of weight removals.

### 3.1 States and readout

Let D be the dense model and M a static sparse model. For each clean sequence, choose a fixed query set Q before sampling context reveals. Every query remains masked. Each other position receives an independent uniform random variable U. At visibility probability p_j, reveal its gold token if U≤p_j. Sharing U across phases produces nested visible sets. Visibility probability refers to eligible context positions, not the realized visible fraction of the whole sequence.

The implemented bank has eight sequences of length 256, two chains per sequence, eight queries per sequence, and eight states per chain. Visibility probabilities are equally spaced in logit space from 0.05 to 0.95. Unchanged states are retained. These are gold-context calibration chains rather than sampled decoding trajectories.

For query q with fixed gold token y, the readout f_X measures the model's confidence in y against all other tokens:

\[
f_X(x,q)=\log\frac{p_X(y\mid x,q)}{1-p_X(y\mid x,q)},\qquad
 e_{c,j,q}=f_M(x_{c,j},q)-f_D(x_{c,j},q).
\]

The residual e is the sparse model's prediction error relative to the dense model at one state. A response compares two states while keeping q and y fixed. Its error is exactly the difference of their residuals:

\[
\underbrace{[f_M(x_{c,k},q)-f_M(x_{c,j},q)]}_{\text{sparse response}}
-\underbrace{[f_D(x_{c,k},q)-f_D(x_{c,j},q)]}_{\text{dense response}}
=e_{c,k,q}-e_{c,j,q}.
\]

This identity explains Figure 1: a common offset cancels, while a changing offset can distort the response. The readout is scalar and does not identify how probability is redistributed among incorrect tokens.

### 3.2 Endpoint and response objectives

The endpoint objective A measures average prediction error. Each sequence, chain, phase, and query has equal weight within its respective average:

\[
A=\mathbb E_{\mathrm{sequence},c,q}\left[\frac18\sum_{j=0}^{7}e_{c,j,q}^{2}\right].
\]

To measure responses at three distances, form the following phase pairs within each chain:

| Scale d | Phase pairs | Interpretation |
|---|---|---|
| 1 | (0,1), (2,3), (4,5), (6,7) | Smallest selected context updates |
| 2 | (0,2), (1,3), (4,6), (5,7) | Intermediate updates |
| 4 | (0,4), (1,5), (2,6), (3,7) | Largest selected updates |

Each scale forms four disjoint pairs and uses every phase once. These distances are phase-index gaps on the logit-spaced visibility grid, not fixed numbers of revealed tokens. For d in {1,2,4}, let E_d denote the corresponding row of the table. Define

\[
C_d=\mathbb E_{\mathrm{sequence},c,q}
\left[\frac14\sum_{(j,k)\in E_d}(e_{c,k,q}-e_{c,j,q})^2\right],\qquad
C_{\mathrm{natural}}=\frac{C_1+C_2+C_4}{3},\qquad
L_{\mathrm{Multi}}=A+C_{\mathrm{natural}}.
\]

Averaging the three scales keeps their total coefficient fixed. For the changing residual in Figure 1, A=1 and (C_1,C_2,C_4)=(0,4,0). It therefore receives a larger combined loss than the constant residual, even though any objective A+λC_1 treats the two equally. This construction explains the choice of multiple scales; it does not establish their optimality.

Every state appears once at each scale. Applying (a−b)²≤2a²+2b² to the four pairs gives C_d≤4A, hence 0≤C_natural≤4A. Response matching thus changes the weighting of finite prediction errors and becomes redundant when all those errors are zero. This algebraic bound supplies no downstream accuracy guarantee.

For the cross control, keep the same sequence, query, and phase pairs, but connect the two different chains in both directions. Accidentally nested or identical states remain included. The four objectives are:

| Allocation | Objective | Question addressed |
|---|---|---|
| A | A | Reference using the same individual states |
| Multi | A+C_natural | Added value of natural-chain response matching |
| Cross | A+C_cross | Effect of changing the connections |
| CrossMatched | A+βC_cross | Connections after matching scale at the reference mask |

The coefficient β=0.932319907797 is fixed from the uniform model's calibration ratio C_natural/C_cross. Matching one reference value leaves possible differences in candidate marginal costs. The cross controls therefore support conclusions about these particular connection designs, with semantic and scale explanations considered together.

### 3.3 Static allocation

All candidates retain the same native sparse-prefix Wanda rankings and surviving weight values. In a jointly sparse uniform-50% model, each of the 32 blocks is probed at 48% and 52% sparsity. Seven projections share the block's proposed rate. The signed finite cost g_ℓ measures the change in objective per additional removed weight:

\[
g_\ell=\frac{L(M_{\ell,52})-L(M_{\ell,48})}
{N_{\mathrm{pruned}}(M_{\ell,52})-N_{\mathrm{pruned}}(M_{\ell,48})}.
\]

Average ranks map these costs to rates between 45% and 55%. Grouped row-count rounding enforces the exact budget across 224 projections: 3,489,660,928 of 6,979,321,856 prunable weights are removed. The rounding step enforces the budget. The overall allocation remains a heuristic because it extrapolates local probes and changes multiple blocks together.

The same cached probe outputs support all four objectives. Additional response terms require only scalar arithmetic once those outputs exist. Obtaining the probes and calibration bank still requires model inference, so cost accounting distinguishes reused artifacts from newly executed work.

## 4. Experimental setup

The evaluation follows the two questions in the introduction. Multi−A tests whether the response term adds value under a shared bank and backend. Multi−Cross and Multi−CrossMatched test the connections under the specified controls. Fresh-span diagnostics measure the prediction and response errors of the resulting masks.

The checkpoint is GSAI-ML/LLaDA-8B-Base, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, in BF16. GSM8K uses frozen five-shot requests, 256 denoising steps, a 256-token generation limit, block length 256, temperature zero, and strict-match grading. Seeds, prompts, and physical model identities are fixed in the source config. The masks are fixed before the primary comparison.

All four candidates generate answers for all 1,319 questions. The 200 questions examined in the earlier development and separate100 stages are identified by frozen IDs. The primary analysis uses the remaining 1,119 questions. Their exclusion from those two stages does not certify absence of exposure elsewhere in the project's history.

The three primary contrasts use exact paired McNemar tests with Holm correction across the family. Paired bootstrap intervals use 10,000 draws with seed 20260927 and remain unadjusted. Full1319 and exposed200 scores are descriptive. The primary sample and stopping rule remain fixed throughout evaluation.

Fresh fidelity diagnostics use eight separate spans, 128 states, and the same readout definitions. Their uncertainty unit is the span. The response-sign metric evaluates all 28 within-chain phase pairs with absolute dense response above 1e−6. This is a broader pair set than the 12 multiscale edges used by the objective.

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

| Arm | Full1319 | Previously examined200 | Primary1119 |
|---|---:|---:|---:|
| A | 764/1319 | 118/200 | 646/1119 |
| Multi | 758/1319 | 122/200 | 636/1119 |
| Cross | 757/1319 | 120/200 | 637/1119 |
| CrossMatched | 755/1319 | 121/200 | 634/1119 |

| Primary contrast | Gain/loss | Difference (pp) | Exact p | Holm p | Unadjusted 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| Multi-A | 64/74 | -0.894 | 0.44372 | 1 | [-2.949, 1.162] |
| Multi-Cross | 74/75 | -0.089 | 1 | 1 | [-2.234, 1.966] |
| Multi-CrossMatched | 67/65 | +0.179 | 0.93068 | 1 | [-1.787, 2.145] |

Multi-A shows a negative difference (-10 answers) and does not pass the Holm-adjusted 0.05 threshold. Multi-Cross shows a negative difference (-1 answers) and does not pass the Holm-adjusted 0.05 threshold. Multi-CrossMatched shows a positive difference (+2 answers) and does not pass the Holm-adjusted 0.05 threshold. These pairwise findings do not establish equivalence, universal necessity, or generalization. The contribution claim must be reviewed jointly with the control results and the uncertainty intervals.

### 5.3 Fidelity and cost

| Arm | A | Natural C | Cross C | Query CE | Response sign flip rate |
|---|---:|---:|---:|---:|---:|
| A | 0.732798 | 0.868318 | 1.01725 | 3.00819 | 0.0758929 |
| Multi | 0.750689 | 0.867225 | 1.00993 | 3.02124 | 0.0744978 |
| Cross | 0.724621 | 0.835103 | 0.975545 | 3.01986 | 0.0742188 |
| CrossMatched | 0.724543 | 0.83233 | 0.975925 | 3.02052 | 0.0739397 |

Diagnostic fidelity and task capability measure different properties. A lower response loss is not itself evidence of improved answer accuracy or a causal explanation for any observed gain.

The current run took 8.236 wall-clock hours. Current plus preserved first-attempt records contain 1,351,936 model forward calls. This count is complete for the recorded current and preserved first attempts. It excludes earlier construction of reused probes, calibration artifacts and masks, so it is not the end-to-end method cost.

## 6. Discussion and limitations

The central distinction is between preserving predictions and choosing which patterns of prediction error to penalize. The response term assigns more cost to errors that change across selected contexts. Whether this preference is useful for pruning depends on the generated answers. The full comparison addresses that question under one fixed bank and backend; the cross contrasts help assess the selected connections.

Three limitations shape the interpretation. First, gold-context calibration may differ from contexts generated by the sparse model, and a scalar gold-versus-rest readout leaves changes among incorrect alternatives unresolved. Second, local block probes approximate the effect of changing many blocks together. An adverse outcome therefore constrains this objective, bank, and allocator jointly. Third, one calibration realization, checkpoint, sparsity level, and benchmark cannot establish generalization. More benchmark questions improve precision within this comparison, while independent calibration replications address a different uncertainty.

Practical value also requires matched comparisons with competitive pruning methods and the full cost of calibration. Sink-Aware Pruning and stronger allocation baselines provide relevant points of comparison. The current four-arm study is designed to diagnose the response criterion. Inference speed requires separate measurement because unstructured zero weights alone do not establish a runtime improvement.

## 7. Conclusion

Equal average prediction error can correspond to different distortions of a model's response to context. This motivates a concrete pruning criterion: match both predictions and their changes across selected context reveals. The shared-bank comparison tests whether that preference improves a static allocation, and the cross-chain controls test the chosen connections. On the primary 1,119 GSM8K questions, Multi differs from A by -0.894 percentage points (unadjusted 95% paired bootstrap interval [-2.949, 1.162]; Holm-adjusted p=1). The three controlled comparisons are interpreted jointly in Section 5.

## Evidence and reproducibility

Figure 1 is a constructed algebraic example, generated by `make_response_figure.py`; its inputs and exact losses are saved in `figures/response-example.json`. Every empirical table is generated from verified result files. After full completion, `closeout.json` records source hashes, JSON locations, per-question verification, historical reproduction, and attempt-level costs. The source experiment, prior outputs, and excluded IDs are preserved. The claim map tracks the evidence still needed for broader method claims.
