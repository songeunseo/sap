# Axis 3 — DLM analysis and observations as static weight-pruning hypotheses

Date: 2026-09-18  
Scope: primary literature on what masked DLMs compute during denoising, then concrete hypotheses for **one fixed post-training weight mask**. This note does not treat cache eviction, activation sparsity, token scheduling, or feature steering as static weight pruning. No GPU, forward pass, or experiment was run.

## Evidence boundary from the project state

The current research record does not support the premise that Uniform always wins. On the available 50% WikiText validation, DSA re-search and EvoPress beat the rowwise Uniform reference, while provenance differs across methods. The native sparse-prefix GSM8K mini comparison is Uniform 54, A-only 55, and A+C 61/100; A+C versus A is inconclusive (`p=0.145996`), has no NELBO measurement, and mixes added context with a changed mask count. Historical Uniform 62 uses a different cached mask pipeline. The C1 coverage screen is 54 versus pooled 50 versus Uniform 54 on the mini screen. Role-minimax and sparse-context refresh results do not justify a new role allocator.

The useful question for this axis is narrower: which directly observed denoising property could change a *static* mask or between-unit budget, and what exact measurement would distinguish it from pooled Wanda, masked calibration, or generic output preservation?

## Primary-source observations

### 1. Denoising time is a functional axis, but not a universal importance weight

Lu, *Measuring Temporal Linguistic Emergence in Diffusion Language Models* ([arXiv:2604.23235](https://arxiv.org/abs/2604.23235)), analyzes three independent 32-step LLaDA-8B-Base runs on masked WikiText-103. Coarse semantic and POS information is more linearly recoverable than exact lexical identity; uncertainty still separates eventually correct and incorrect tokens late in denoising despite calibration drift; and re-masking sensitivity peaks around steps 15–18. At the three-seed peak, 10.92 of a 10.96 point accuracy drop is direct to the re-masked positions, with only 0.03 collateral drop. Commitment is category-dependent: numbers/content words stabilize earlier than function-heavy categories, and early commitment is not a correctness guarantee.

The direct pruning implication is a **state distribution problem**: a mask that looks safe on the pooled average may leave a high-error denoising window or a state-specific support uncovered. The paper does not measure pruning, does not establish a universal middle-step weight, and uses one model/data setting. A static allocator must therefore test uniform-state, middle-window, and held-out-state versions rather than hard-code the reported window.

### 2. DLM conditional dependence uses both sides and an implicit mask-rate signal

Catruna and Radoi, *Induction in Both Directions* ([arXiv:2607.15893](https://arxiv.org/abs/2607.15893)), use matched small attention-only AR and absorbing-mask DLMs. The DLM learns previous-token and next-token pathways feeding later induction heads, works approximately symmetrically from past and future source context, and is stronger than its AR counterpart only when both sides of the masked token are visible. A single residual state also carries the global masked-token fraction; patching this direction changes prediction entropy. The mechanistic study is causal, but it is on depth 1–3 small models rather than LLaDA-8B.

The static-mask implication is that endpoint reconstruction under independent states may miss the **conditional update** produced when a token becomes visible while the global mask rate is held fixed. It is a hypothesis about a measurable DLM function, not a claim that attention-head or induction circuits transfer directly to the 8B checkpoint.

### 3. Sparse features and decoding order expose state-dependent hidden dynamics

Wang et al., *DLM-Scope* ([arXiv:2602.05859](https://arxiv.org/abs/2602.05859)), train Top-K SAEs on DLM activations sampled over denoising and corrupted positions. SAE insertion can reduce masked-token cross-entropy in some early DLM layers, an effect absent or much weaker in the AR comparison. Their decoding-order analysis reports different feature trajectories: confidence-based orders show larger early changes on still-masked positions and continued deep-layer changes after tokens are decoded, while random-order decoding is quieter. On Dream-7B GSM8K, their Origin order scores 8%, versus 56% for TopK-margin and 59% for Entropy under the reported 128-step setup.

This supports using feature/support turnover as a diagnostic for a static mask. It does not make SAE latents causal importance scores, and it does not show that a feature-preserving weight mask improves NELBO. The proposed allocator below can initially use raw hidden responses and use an SAE only as a held-out diagnostic.

### 4. Activation sparsity is high but its support drifts across DLM steps

Szatkowski et al., *Universal Properties of Activation Sparsity in Modern Large Language Models* ([arXiv:2509.00454](https://arxiv.org/abs/2509.00454)), evaluate LLaDA-8B with separate activation masks at every forward pass. LLaDA has high critical activation sparsity and slightly more favorable sparsity–performance curves than its AR counterpart. Across diffusion steps, consecutive activation-mask Jaccard similarity is relatively stable, but similarity to the initial mask declines rapidly; applying sparsification to both inputs and intermediate activations makes patterns less stable. The authors conclude that activation sparsity is promising for acceleration, while warning that dynamic patterns do not simply mirror AR behavior.

This is evidence for a static-weight **coverage/turnover** question, not evidence for copying an activation mask into a weight mask. The paper explicitly measures a changing activation mask and excludes a static post-training weight allocation objective.

### 5. Context is local and mask count can be a distractor

Piskorz et al., *Masks Can Be Distracting* ([arXiv:2511.21338](https://arxiv.org/abs/2511.21338), v2), find strong locality bias in MDLMs and an inverse scaling effect: appending more mask tokens can substantially hurt context comprehension, especially in long contexts. Mask-agnostic fine-tuning reduces this effect. Their result means a context perturbation that also changes the number of masks is not a clean conditional-response measurement. It also supplies a useful stress test for a static mask: preserve behavior under relevant context changes while holding mask count and position distribution fixed, then separately test extra-mask robustness.

### 6. Attention sinks move, and DLMs can reroute around them

Rulli et al., *Attention Sinks in Diffusion Language Models* ([arXiv:2510.15731](https://arxiv.org/abs/2510.15731), v2), analyze LLaDA-8B, Dream-7B, and MMaDA-8B. DLM sink positions move, vanish, or split between masked and unmasked tokens; LLaDA sinks often occur on punctuation, whitespace, and structural tokens. Masking one sink decreases DLM task performance by less than 1% in their tests, unlike severe AR degradation. The authors attribute the robustness to bidirectional attention and iterative unmasking, which provide alternative paths.

This is a counterexample to a tempting static rule: do not protect weights solely because they contribute to a high-attention sink, and do not equate attention mass with irreplaceable weight capacity. A useful allocation signal should measure residual conditional function after the model can reroute.

### 7. Temporal reuse papers are adjacent mechanisms, not static-weight evidence

Frumkin et al., *DARE* ([arXiv:2605.08134](https://arxiv.org/abs/2605.08134)), report token-wise redundancy in bidirectional self-attention: temporal query changes predict redundancy in corresponding key, value, and output activations. DARE-KV and DARE-O reuse cached activations and report up to 87% activation reuse with small benchmark degradation. Sparse-dLLM ([AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40586)) similarly evicts low-relevance cache entries. These observations motivate state-conditioned measurements, but their intervention is cache/activation reuse. They do not establish that a permanently removed weight is safe across all future denoising states.

### 8. Progress subspaces are real but should not be promoted directly to pruning scores

Rulli et al., *Subliminal Clocks* ([arXiv:2607.01774](https://arxiv.org/abs/2607.01774)), report that denoising progress is decodable across LLaDA and Dream residual streams, including on unmasked positions; low-dimensional mean-vector directions can be steered to change confidence, entropy, and output KL. The project’s prior audit correctly treats this as a representation/steering observation. It is not evidence that preserving a clock direction in a pruned weight matrix improves downstream quality. A clock-derived score is a diagnostic or stratification variable unless a static-mask rescue experiment establishes utility.

## Candidate 1: trajectory-conditioned state-coverage allocation (TC-Wanda)

### Observed fact

DLM activation supports remain sparse but drift across steps, and temporal analysis finds a non-initial sensitivity window. Therefore, a single pooled activation-energy statistic can hide a module that is safe on average but fails on a subset of denoising states.

### Unproven hypothesis

At an equal global weight budget, modules/projections whose **one fixed support** has high worst-state or high state-to-state variation in response error need more surviving capacity than modules with the same pooled Wanda score but stable response across states. A DLM-specific state-coverage scalar can improve allocation even when the within-unit weight ranking stays Wanda.

### Measurable objective

For calibration state (s=(x_t,t)), projection (u), and one static candidate mask (M_u(k)) at local retained/pruned count (k), compute a response-loss proxy

\[
 e_u(s,k)=\frac{\|(W_u\odot(1-M_u(k)))X_{u,s}\|_F^2}
 {\|W_uX_{u,s}\|_F^2+\epsilon}.
\]

Use the same mask (M_u(k)) for all states. Define a state-coverage cost

\[
 C_u(k)=\sum_s w_s e_u(s,k)+\lambda\,Q_{0.90,s}[e_u(s,k)],
\]

where the first screen uses (w_s) uniform, then compares a pre-registered middle-window weighting suggested by the sensitivity curve. A second diagnostic reports the state turnover (Q_{0.90}(e)-Q_{0.50}(e)) and support Jaccard across state groups; it is not silently added to the primary score.

Use an existing exact-budget discrete allocator to select (k_u) minimizing (sum_u C_u(k_u)), while retaining the current static within-projection ranking and enforcing the same total number of pruned weights. This is a criterion plus an existing allocator, not a new solver or a dynamic mask.

### Mask and budget decision

1. Build one pooled static mask family per projection at candidate local budgets.
2. Evaluate (e_u) on disjoint state groups: early, middle, late, masked-heavy, and masked-light; keep sequence documents disjoint between development and evaluation.
3. Allocate the exact global budget using (C_u(k)); the comparison is rowwise Wanda allocation, state-mean only ((\lambda=0)), state-quantile only, and Uniform under the same mask-construction pipeline.
4. Validate on held-out WikiText NELBO/PPL bound and GSM8K only after the proxy screen. Report state-wise NELBO, not only a global mean, to test whether any gain is confined to the targeted window.

The decisive ablation is **pooled state mean versus state quantile/coverage at identical masks and allocator cost**. If quantile allocation does not improve held-out state-wise NELBO or is unstable under state resampling, stop the candidate. An SAE can be added afterward as a diagnostic of feature-support turnover; it is not part of the primary method.

### Closest prior and counterevidence

This candidate is adjacent to the project’s C1 active-set coverage proposal, DLM-Scope’s SAE dynamics, and DLM activation-sparsity analysis. It must claim the added information precisely: state-conditioned error of one fixed static support, not “activation sparsity implies weight sparsity.” The project’s earlier role/reconstruction work found local improvements that often failed to predict functional KL; in V2, 40.52% of bundles improving both role-local reconstructions still had worse KL. That is a strong reason to require held-out full-vocabulary NELBO and to avoid calling the proxy causal. Sink-Aware Pruning ([arXiv:2602.17664](https://arxiv.org/abs/2602.17664)) already uses masked calibration and average soft sink reweighting in Wanda/SparseGPT, so masked-state calibration alone is not novelty.

## Candidate 2: mask-rate-matched conditional-update allocation (MCU-Wanda)

### Observed fact

The DLM induction study finds useful computation from both past and future context and an implicit global mask-fraction signal. The mask-distractor study shows that changing mask count itself changes context comprehension. Hence, independent endpoint states are insufficient: a clean test needs two aligned states with a content/context change and the same mask count.

### Unproven hypothesis

A static support selected to preserve the **conditional update** caused by revealing or changing a context token, while holding global mask rate fixed, can preserve parallel denoising behavior better than a support selected only from endpoint responses. The hypothesis is functional and finite-context; it does not assert that a recovered small-model induction circuit is the 8B mechanism.

### Measurable objective

For an aligned masked query, construct (x^-) and (x^+) with the same sequence length and the same number of masked positions. Change one context condition by swapping one masked position with one visible context position, or use a matched reveal/unreveal swap. For projection input matrices (X^-_u,X^+_u), define

\[
\bar X_u=\left[\frac{X^-_u}{\sqrt 2},\frac{X^+_u}{\sqrt 2},
\sqrt{\lambda}(X^+_u-X^-_u)\right],
\]

and the corresponding dense targets

\[
\bar Y_u=\left[\frac{W_uX^-_u}{\sqrt 2},\frac{W_uX^+_u}{\sqrt 2},
\sqrt{\lambda}W_u(X^+_u-X^-_u)\right].
\]

For a fixed mask (M_u), score the endpoint-plus-update response error

\[
 L_u(M_u)=\|(W_u\odot M_u)\bar X_u-\bar Y_u\|_F^2.
\]

The third block is the DLM-specific term. It measures preservation of the change in a projection’s response under a controlled context update; it is not a token-level loss, an attention score, or an AR next-token objective. The same criterion can be implemented as a scalar Wanda-style column score from the augmented Gram matrix and fed to the existing exact-budget allocator.

### Mask and budget decision

1. Keep query positions, sequence length, and mask count fixed across (x^-) and (x^+); separately record a mask-count-shift condition rather than mixing it into the primary pair.
2. Use (\lambda=0) endpoint-only, (\lambda>0) paired update, and a shuffled-pair control that preserves the endpoint marginals but breaks correspondence.
3. Build a single static exact-budget mask from the augmented pairs. Do not give each state a separate mask and do not update weights.
4. Evaluate (a) full-vocabulary query KL/NLL, (b) direct conditional-update error (\|\Delta r_{\text{sparse}}-\Delta r_{\text{dense}}\|^2), (c) held-out WikiText NELBO, and (d) GSM8K. Include a matched generic response-preservation or DSA/EvoPress control at the same calibration-forward budget.

The candidate earns support only if paired-update allocation beats endpoint-only on held-out conditional-update error **and** improves a downstream objective under the exact same global budget, while the shuffled-pair ablation removes most of the gain. A direct-update proxy gain without NELBO/GSM8K gain is a diagnostic result, not a method result.

### Closest prior and counterevidence

The project’s existing A+C candidate is related but not equivalent: its endpoint and conditional gold-logodds terms currently mix additional gold context with a changed mask count, and its mini gain is inconclusive. FAIR-Calib ([arXiv:2606.06547](https://arxiv.org/abs/2606.06547)) already studies irreversible frontier flips and weighted calibration for DLM quantization; Quant-dLLM ([arXiv:2510.03274](https://arxiv.org/abs/2510.03274)) already uses masked calibration and sensitivity-based mixed precision; COPSD and OPTD already use future-context/on-policy transition signals. Generic response preservation and Jacobian sensitivity also have prior art, including 2ndMatch ([arXiv:2506.05398](https://arxiv.org/abs/2506.05398)). The narrow candidate contribution would be a static weight mask whose allocation objective explicitly preserves matched conditional updates under a fixed mask rate, with a shuffled-pair control.

The main failure mode is known: local representation/reconstruction error need not predict final functional KL in a jointly sparse network. The candidate should therefore be screened first on a small held-out state/action set and promoted only after full-vocabulary and downstream confirmation. It should not be combined automatically with clock directions, sink scores, SAE features, or role minimax.

## What should not be inferred from these papers

- A moving attention sink is a token/cache phenomenon; it does not identify permanently removable weights.
- High or drifting activation sparsity does not imply a corresponding static weight support.
- A latent clock or progress subspace is useful for stratifying calibration states, but preserving it is not yet a validated pruning objective.
- Bidirectional induction in a small matched model is evidence for a conditional-dependence diagnostic, not proof of an 8B circuit-level pruning rule.
- Attention-guided decoding order and cache reuse can be strong DLM methods while remaining outside static weight pruning.

## Decision for axis 3

Keep both candidates as hypotheses, with different risk profiles. TC-Wanda is the lower-cost route: it reuses existing within-unit ranking and asks whether a state-coverage/quantile allocation adds information over pooled Wanda. MCU-Wanda is the more DLM-functional route: it asks whether a static mask preserves context-conditioned updates after controlling the implicit mask-rate variable, but it has higher calibration and evaluation cost and stronger prior-art overlap.

If only one next diagnostic is allowed, use the MCU-Wanda pair construction on a small held-out set with (\lambda=0), paired-update, shuffled-pair, and mask-count-shift controls. If the paired term is not predictive of full-vocabulary KL or held-out NELBO, stop it. If a cheaper allocator screen is required first, run TC-Wanda with state-quantile versus pooled-state allocation and no SAE or clock feature.

The parent project’s current saved-probe audit reports that naive response linearization does not predict the observed A+C/Uniform ordering and that cross-layer probe cosines are near zero. This argues for direct matched-state measurements and exact-budget mask evaluation, rather than extrapolating small perturbation probes into a universal interaction or allocation rule.

## Sources checked

1. Lu, *Measuring Temporal Linguistic Emergence in Diffusion Language Models*, arXiv:2604.23235.
2. Catruna & Radoi, *Induction in Both Directions*, arXiv:2607.15893v2.
3. Wang et al., *DLM-Scope*, arXiv:2602.05859.
4. Szatkowski et al., *Universal Properties of Activation Sparsity in Modern Large Language Models*, arXiv:2509.00454.
5. Piskorz et al., *Masks Can Be Distracting*, arXiv:2511.21338v2.
6. Rulli et al., *Attention Sinks in Diffusion Language Models*, arXiv:2510.15731v2.
7. Frumkin et al., *DARE*, arXiv:2605.08134.
8. Rulli et al., *Subliminal Clocks*, arXiv:2607.01774v2.


## Parent review: accepted scope and priority corrections
The quantile/coverage variant is lower priority: close to C1/role-minimax without new pruning evidence. Matched-count response preservation continues an existing candidate. Equal count does not imply identical mask locations or isolate all semantic/progress factors. A column-score replacement of full augmented reconstruction is a diagonal approximation, not equivalence. Current256-step/256-slot evaluation is roughly one commitment per step; parallel factorization error cannot explain its mini scores. Proxy utility prediction is a useful diagnostic, not a universal development gate. Quality/cost tradeoffs also count. No proposal is accepted for execution automatically.
