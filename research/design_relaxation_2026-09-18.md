# Allocation design for fixed conditional-response objective

Date: 2026-09-18  
Status: design review only; no model forward, new mask, training, or experiment was run.

## Decision

Use a hard, parameter-weighted boundary-swap allocator as the primary backend for the fixed vector conditional-response objective. Start the controlled backend at 32 layer units, retaining the within-row Wanda candidate order and original surviving weights, and evaluate every accepted candidate with the assembled hard sparse model. Treat 224 projection units as a separate granularity axis and row thresholds as a higher-capacity secondary control.

Lua/BESA-style threshold learning should be implemented as a control or proposal generator, not silently treated as the main method. A soft mask has a useful optimization signal, but it optimizes a fractional model and a count penalty before hardening. A hard-forward STE removes that soft-forward gap but still supplies a biased derivative through an integer top-k decision. Neither gives a mathematically valid gradient of the deployed mask count. Direct boundary swaps have no gradient, yet their finite differences are the exact objective changes for the tested hard masks. This is the better first test because the research question is fixed-budget behavior of the hard DLM, and the previous (J)-extrapolation already failed when a small local probe was mapped to a wider allocation.

This recommendation is a backend decision, not a claim that direct search is always faster. It trades optimizer steps for sparse-model forwards. The comparison must report wall-clock, forward count, peak memory, soft-to-hard objective gap, and final exact-count compliance.

## Fixed problem and constraints

For each common masked query (q), let (z_D^0(q),z_D^1(q)) be dense centered-vocabulary logits at two context states and (z_M^0(q),z_M^1(q)) the corresponding sparse logits. Centering each vocabulary vector removes an arbitrary common logit offset. With

\[
e_a(q)=z_M^a(q)-z_D^a(q),\qquad
\Delta e(q)=(e_1-e_0)(q),
\]

the fixed objective is

\[
L(M)=\mathbb E_q\left[\frac12\big(\|e_0(q)\|_2^2+\|e_1(q)\|_2^2\big)
              +\lambda\|\Delta e(q)\|_2^2\right].
\]

The first term is endpoint fidelity and the second preserves the change induced by the controlled context change. This is an allocation objective, not a task-CE replacement or a guarantee of NELBO/GSM8K quality. Keep the pair bank, query positions, mask count, sequence length, timestep construction, model revision, and λ fixed while comparing allocators.

For unit (u), sort each row once using the existing Wanda order. An integer (k_u) or row-boundary vector then determines how many lowest-ranked candidates are removed. The deployed mask is (M(k)), with the original values of all retained weights unchanged. The budget is an integer constraint:

\[
\sum_{u,r} n_{u,r}(k_{u,r})=P_{\mathrm{remove}}.
\]

Here (n_{u,r}) is the number of removed weights in row (r) of unit (u); it must be counted in parameters, rather than averaging percentages across projections. If a move removes different numbers of weights, accept only a bundle of row quanta whose net parameter delta is exactly zero. A final rounding repair is part of the algorithm and must be reported, not hidden in a percentage.

## What the literature actually provides

| Method | Published algorithm | What it would mean here | Main mismatch or risk |
|---|---|---|---|
| BESA | Sorts Wanda scores, represents a layer rate as a simplex mixture of candidate rates, generates binary masks from monotone pruning probabilities, and uses an STE for the binary mask. It prunes one transformer block at a time with block reconstruction plus an L2 sparsity penalty while freezing the original weights. The row-wise implementation uses a custom CUDA operator; the paper also reports a lightweight layer-wise version. ([paper](https://arxiv.org/html/2402.16880v2), [official code](https://github.com/OpenGVLab/LLMPrune-BESA)) | Replace block reconstruction by (L(M)), keep the fixed order, and learn a small rate vector. | The penalty targets a rate, not an exact global count; simplex interpolation and STE are surrogates. Sequential block inputs also differ from a whole-model static-mask objective. |
| Lua-LLM | Maps Wanda scores to ([0,1]), gives each row a learnable threshold (t), uses σ(τ(s-t)) as a soft Top-K mask, freezes weights, optimizes next-token CE plus a piecewise count regularizer, then extracts hard masks. The paper sets τ to the row width and trains thresholds for 500 iterations. ([NeurIPS paper](https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf), [abstract](https://papers.nips.cc/paper_files/paper/2025/hash/b2c39fe6ce838440faf03a0f780e7a63-Abstract-Conference.html)) | Use the same threshold parameterization with endpoint-plus-response reconstruction in place of CE. | The optimization forward is fractional and the regularizer only encourages the target count; hard extraction can change (L). The paper's successful allocation is not evidence that this DLM objective or exact-budget adaptation works. |
| Hard Concrete (L_0) | Uses stochastic hard-concrete gates and a differentiable expected (L_0) term; gates can become exactly zero and are optimized jointly with weights. ([paper](https://arxiv.org/abs/1712.01312)) | Gate a unit or boundary and freeze (W), then project the sampled gates to an exact static mask. | Expected count is not exact count, stochastic masks add variance, and per-weight gates discard the fixed-order allocation constraint. A unit-level adaptation is possible but is a new adaptation, not the published algorithm. |
| MaskLLM / Gumbel-Softmax | Learns a categorical distribution over local N:M candidate masks, uses Gumbel-Softmax, and forms a weighted average mask; weights are frozen during mask learning. ([paper](https://arxiv.org/abs/2409.17481), [official code](https://github.com/NVlabs/MaskLLM)) | Use categorical choices over adjacent boundary masks or row quanta. | The published structure is local N:M, while this project needs unstructured fixed Wanda order and a model-wide weighted count. Global exact-budget sampling and hardening would be additional design work. |
| DSA / EvoPress | DSA searches an allocation function with evolutionary operators; EvoPress mutates a compression-level vector under a budget and evaluates whole-model fidelity. ([DSA code](https://github.com/lliai/DSA), [EvoPress paper](https://arxiv.org/html/2410.14649)) | Use as discrete-search controls or reuse their search style with (L(M)) as fitness. | Search and exact-budget allocation are established machinery. Any contribution must be the conditional-response signal and its measured value, not a claim of a new optimizer. |

The BESA paper explicitly says that its binary mask derivative is estimated with STE and that only a small number of rate coefficients are learned, while its official repository requires a custom CUDA operator for the row-wise implementation. Lua instead makes the mask sigmoid-soft during optimization and hard only in the final pruning step. Calling both “differentiable pruning” hides a meaningful soft-hard difference.

## Backend comparison

### 1. Lua/BESA-style soft thresholds

For a fixed row score (s_{u,r,j}), a direct adaptation is

\[
\tilde m_{u,r,j}=\sigma\big(\tau(s_{u,r,j}-t_u)\big),
\qquad
\tilde P=\sum_{u,r,j}(1-\tilde m_{u,r,j}).
\]

Use one threshold per layer for the 32-unit controlled backend, one threshold per projection for the separate 224-unit axis, or one threshold per row for the row control. Optimize (L(\tilde M)) with a declared budget mechanism, preferably a global dual variable or an equality-constrained rate parameterization. A squared count penalty is a reproducible control, but it should not be described as exact budget. Initialize at the uniform 50% boundary, anneal τ only under a held-out hard-mask check, and at every checkpoint extract a hard mask, repair the integer count, and record (L(M_{hard})).

The derivative with respect to a threshold contains

\[
\frac{\partial \tilde m}{\partial t}=-\tau\tilde m(1-\tilde m).
\]

For a large τ, almost every score is saturated and only scores near the boundary contribute. For a small τ, many weights are fractional, gradients are less sparse but the soft model can be far from any hard mask. Thus the method has a real temperature/support-saturation tradeoff. A frozen weight tensor does not remove the backward graph or activation memory; two endpoint states per pair at each update roughly double the objective-side forward work, and backpropagation stores intermediate activations unless checkpointing is used. BESA's blockwise schedule reduces memory, but applying a whole-model conditional response objective to one block at a time would be an adaptation whose validity must be checked.

The useful role of this backend is diagnostic: it tests whether a low-dimensional threshold parameterization can find a good basin and supplies a candidate list for hard evaluation. The hard objective, not the soft loss, decides acceptance.

### 2. Hard-forward STE

Use an exact forward mask and a surrogate backward path:

```text
m_h = (s >= t).float()                 # exact binary forward
m_s = sigmoid(tau * (s - t))            # backward surrogate
m   = m_h + (m_s - stop_gradient(m_s)) # forward m_h, backward m_s
```

This removes the fractional-mask forward and usually narrows the soft-hard gap. It does not create a valid derivative of the deployed objective: away from a boundary, changing (t) can leave the integer mask unchanged while the surrogate reports a nonzero gradient; at a boundary, the true objective has a jump and no classical derivative. Different choices of identity, clipped sigmoid, or temperature therefore encode different search heuristics. The method can also oscillate when a minibatch moves many thresholds across ties.

Use hard-forward STE only as a matched control. Keep the count repair and final hard evaluation identical to the soft-threshold control. A better STE score than the direct boundary score is not evidence of a better allocator until the resulting exact mask wins on (L(M_{hard})) and held-out NELBO.

### 2a. Hard-forward exact-budget projected gradient

A useful middle ground is to keep a real rate vector `a`, project it onto the weighted budget and rate box, construct an exact hard mask from the projected vector for the forward pass, and use the STE only to propose the next rate vector. After each update, harden and count-repair the proposal, then accept it only when the true `L(M)` decreases. This gives a concrete control for gradient proposal plus exact hard acceptance.

It must start from the same exact Uniform50 mask. Starting at the dense model gives zero endpoint reconstruction residual, so the gradient of this squared reconstruction objective with respect to deletion gates is degenerate at the starting point. A dense start can therefore make the optimizer look inactive for a mathematical reason rather than because the allocation signal is absent.

Projected gradient has a different cost profile from finite exchange. One backward pass can propose changes across many units, but the surrogate gradient is not guaranteed to match any feasible hard exchange, and a projected step can cross many interacting boundaries at once. True hard acceptance then requires extra forwards and rejected proposals can erase the expected speed advantage. Finite exchange tests fewer directions but measures the exact hard objective of every tested move and needs no backward activation graph. Compare them by equal candidate-evaluation budget and report accepted proposals, rejected proposals, hard forward count, and peak memory; do not call the projected method faster from iteration count alone.

### 3. Score-space relaxation

There are two distinct ideas here.

First, learn a real rate or threshold in the already-sorted Wanda score space. This is the minimal adaptation: one (t_u) changes only the boundary and preserves the candidate order. It is essentially Lua's parameterization with the present objective.

Second, relax the ordering itself with SoftSort or a related differentiable ranking operator. Such relaxations are designed because argsort has zero gradients almost everywhere ([SoftSort](https://proceedings.mlr.press/v119/prillo20a.html)). They would let scores exchange order, but that violates the allocation-only primary constraint and confounds importance ranking with budget allocation. Do not use a differentiable sorter in the primary method.

A narrower relaxation that is worth considering as a diagnostic is linear interpolation between adjacent hard boundary masks:

\[
\tilde M_u(\alpha)=(1-\alpha)M_u(k)+\alpha M_u(k+1),
\quad 0\leq\alpha\leq1.
\]

It has no sigmoid saturation and changes only one known boundary quantum, but it still represents a fractional weight mask. Randomly sampling (M_u(k)) or (M_u(k+1)) with probabilities (1-\alpha,alpha) restores hard forward behavior at the cost of a high-variance estimator. This is a proposed adaptation, not a result of SoftSort or Lua.

### 4. Boundary swaps and direct hard search

For a current exact mask (M(k)), form feasible moves by increasing one unit's removal boundary and restoring another unit's boundary. For a move or bundle (b), compute

\[
\Delta_b=L(M_b)-L(M(k)),
\qquad \sum_{u,r}\Delta n_{u,r}(b)=0.
\]

Rank candidates by the measured Δ on the hard sparse model, accept a bounded best move or beam, and re-evaluate the assembled mask after each accepted step. Cache dense centered logits and use the same pair bank for every candidate. If a single projection or row quantum cannot be paired exactly, enumerate small donor/receiver bundles or solve the integer repair problem; never divide a cost by an unweighted percentage and silently round the final mask.

This method has exact support and exact budget at every accepted state, no STE bias, and no soft-hard objective gap. Its limitation is compute: each hard candidate may require two sparse forwards for every context pair, and local costs are not additive when several boundaries move. That limitation is preferable to extrapolating a local surrogate here because the saved CPU audit predicted (L=1.349005) for the measured A allocation and (1.338577) for A+C while the actual values were (1.041629) and (1.025982). The audit does not disprove every local approximation, but it is enough to reject unchecked (J^\top J)-style extrapolation as the main allocator.

Direct search is therefore the primary backend, with a small candidate budget and hard objective acceptance. It should be called a measured exact-budget search, not a new optimizer.

## Granularity choice

| Granularity | Variables and mask | Benefit | Cost/risk | Role |
|---|---|---|---|---|
| 32 layers | One rate or boundary schedule per transformer layer; projection types inside a layer share the allocation. | Very small search space, stable estimates, low metadata and threshold memory. | A layer mixes Q/K/V/O and MLP projections with different parameter counts and response roles; one move is coarse and can hide opposite effects. | Required coarse control; useful for checking whether any gain is only a depth schedule. |
| 224 projections | One rate per projection, with the existing row-wise Wanda order inside each projection. | Captures projection/type differences while keeping 224 variables; boundary moves and exact weighted repair are tractable. | More calibration forwards and higher variance than 32; still cannot distinguish rows within a projection. | Separate granularity axis after the 32-unit control. Use projection parameter counts in the global budget. |
| Rows | One threshold/boundary per output row, or an explicitly declared row group. | Matches the actual fixed Wanda decision boundary and can represent intra-projection heterogeneity. | Hundreds of thousands to millions of variables are model-dependent; per-row response estimates are noisy, mask metadata grows, and soft full-model backward memory becomes the dominant cost. | Secondary stress test or proposal generator; do not make it the first claim. |

The 224 choice is a compromise rather than a claim that projections are the true causal units. If the 224 method only reproduces a depth/type schedule, the 32 control will show it. If it loses because within-projection row heterogeneity matters, the row control can reveal that failure at matched exact budget.

## Implementation skeleton

1. Freeze the dense model, Wanda row order, pair bank, centered dense logits, and allowed rate interval (for example, the existing bounded 45--55% neighborhood). Define each admissible move by its integer removed-parameter count.
2. Construct the initial exact Uniform50 mask. For each allocation unit, store row boundary indices and its parameter count (N_u); do not store only a percentage.
3. For direct search, generate feasible one-quantum donor/receiver swaps and a small set of exact bundles at 32-layer granularity first; repeat the same protocol at 224-projection granularity as a separate axis. Evaluate the assembled hard mask with the fixed (L), retain the best accepted candidate, and repeat only for a prespecified number of moves. Recompute the full objective after every accepted move; never sum independent marginal costs as the final score.
4. For the soft control, initialize thresholds to the Uniform50 boundaries, optimize the same (L) with a declared count constraint, and checkpoint hard masks. Count-repair each checkpoint and log the soft loss, hard loss, gap, and exact parameter count. For hard STE, use the same thresholds, hard forward, and hard checkpoint evaluation.
5. Separate development and held-out documents. Report calibration forward/backward count, wall time, peak memory, mask XOR against Uniform, per-unit rates, and final exact count alongside NELBO and GSM8K. Keep the current A-only and A+C scalar controls as objective controls; do not infer that the vector objective won because a soft loss decreased.

## Cheapest falsification and controls

The smallest informative screen keeps one candidate family and one budget fixed while changing one axis at a time:

- **Objective control:** endpoint-only vector loss versus endpoint-plus-paired response loss, with a pair-shuffle null that keeps endpoint marginals but breaks correspondence.
- **Backend control:** direct hard boundary swaps versus Lua-style soft thresholds versus hard-forward STE, all selected by the final hard (L(M)), with the same candidate evaluation budget.
- **Granularity control:** 32, 224, and row units under the same fixed Wanda order and parameter-weighted exact count.
- **Interaction check:** from one common baseline, compare each single move with the measured joint bundle. Do not compare the sum of accepted sequential Δ values with the final Δ, because that telescopes by definition. A joint-versus-single discrepancy is the actual interaction diagnostic.
- **Baseline provenance:** native Uniform, the historical cached Uniform, and legitimate DSA/EvoPress adaptations must keep their actual mask engines and compute accounting. The old statement that every allocator loses to Uniform is not supported.

The primary falsifier is simple: if the measured conditional-response objective does not beat endpoint-only or a strong full-vocabulary functional control at matched exact budget and matched search cost, the response term has no demonstrated allocation value. If the direct hard backend matches a soft backend after hardening, the differentiable machinery has no demonstrated benefit and should remain an implementation control.

## Recommendation to carry forward

Implement and document the 32-unit direct hard boundary-swap allocator first. Add the Lua-style soft threshold and hard-forward STE controls against that same controlled backend, then repeat the comparison at 224-projection granularity. Keep the hard objective and count audit in place throughout. Treat BESA/Lua as established allocation-learning machinery whose objective, budget handling, and mask forward must be stated exactly. Keep score order fixed in the primary method; any learned re-ranking or stochastic gate family belongs in a separate expanded-family study.

## Sources

- Xu et al., “BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation,” [arXiv HTML](https://arxiv.org/html/2402.16880v2), [official implementation](https://github.com/OpenGVLab/LLMPrune-BESA).
- Lu et al., “Lua-LLM: Learning Unstructured-Sparsity Allocation for Large Language Models,” [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf), [paper page](https://papers.nips.cc/paper_files/paper/2025/hash/b2c39fe6ce838440faf03a0f780e7a63-Abstract-Conference.html).
- Bengio et al., “Estimating or Propagating Gradients Through Stochastic Neurons,” [arXiv](https://arxiv.org/abs/1308.3432).
- Louizos et al., “Learning Sparse Neural Networks through (L_0) Regularization,” [arXiv](https://arxiv.org/abs/1712.01312).
- Fang et al., “MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models,” [arXiv](https://arxiv.org/abs/2409.17481), [official implementation](https://github.com/NVlabs/MaskLLM).
- Prillo and Eisenschlos, “SoftSort: A Continuous Relaxation for the argsort Operator,” [PMLR](https://proceedings.mlr.press/v119/prillo20a.html).
- Li et al., “Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models,” [official implementation](https://github.com/lliai/DSA).
- Sieberling et al., “EvoPress: Accurate Dynamic Model Compression via Evolutionary Search,” [PMLR](https://proceedings.mlr.press/v267/sieberling25a.html), [official implementation](https://github.com/ist-daslab/evopress).

