# A+C 목적함수 업그레이드 검토

Date: 2026-09-22  
Status: bounded design review; no GPU forward, new mask, benchmark, or Obsidian write was performed.

## Scope and verdict

The first concrete upgrade should keep the existing scalar gold-vs-rest A+C readout, fixed Wanda support family, natural nested context pairs, and static exact parameter budget, while replacing rank interpolation with measured feasible hard-mask exchanges. The allocation objective should be tested first as an endpoint-anchored constrained problem:

\[
M_A=\operatorname*{argmin}_{M\in\mathcal C_B}^{\text{same candidate/search budget}} A(M),
\]

\[
\operatorname*{minimize}_{M\in\mathcal C_B} C(M)
\quad\text{subject to}\quad A(M)\le A(M_A)+\epsilon.
\]

Here \(\mathcal C_B\) is the fixed-support, exact-budget candidate set and \(M_A\) is the best A-only candidate found under the same finite search cap; it is not a global optimum. Use \(\epsilon=0\) plus a declared numerical tolerance for the primary comparison. A positive tolerance is a later sensitivity analysis, not an extra tuned parameter in the first test.

This removes the arbitrary relative scaling of \(A+\lambda C\) and asks the narrower question: can conditional-response preservation improve C while maintaining the endpoint fidelity that A-only can attain? It is a constrained/Pareto allocation design, so the constraint itself is borrowed generic optimization. Any contribution must come from an observed held-out quality, robustness, or compute benefit under the matched budget.

The constraint can stall: if \(M_A\) is a local candidate with no one-swap neighbor satisfying the endpoint floor, that is evidence about the discrete tradeoff, not a reason to silently relax the threshold. Record the feasible-neighbor count. A small exact-budget candidate pool with multi-swap exchanges can be a prespecified implementation extension, but changing the support family or allowing weight updates is a separate axis.

## Why the current scalar A+C can fail

The current pair is nested: extra visible context changes at the same time as the amount of masked context. Thus C mixes context-content response with mask-count/progress response. Natural nested pairs should remain the primary denoising trajectory condition, but a count-matched pair and a pair-shuffled null are needed to diagnose this mixture. A new output metric cannot remove that confound.

The scalar readout also collapses all wrong-token mass into one aggregate. It can assign zero error when the gold logit and aggregate wrong log-sum-exp are unchanged while wrong-token identities are rearranged. This is a real limitation, but a teacher-weighted vector metric only partly repairs it.

For example, let the dense endpoint distribution be \(p_D=(.98,.015,.005)\) and the sparse endpoint be \(p_S=(.98,.005,.015)\). The gold probability is unchanged, so the gold-vs-rest endpoint error is zero. The wrong-token logits differ by

\[
e=(0,-\log 3,+\log 3),
\]

while the scalar readout remains blind to this permutation. With \(q=p_D\), the vector Fisher quadratic is

\[
e^\top M_qe
=q_2q_3(2\log 3)^2 + q_1q_2(\log 3)^2 + q_1q_3(\log 3)^2
 - (q^\top e)^2,
\]

which is approximately \(0.0199(\log 3)^2\) after the common-shift/Fisher cancellation. It detects the rearrangement, but weakly because the tail probabilities are small.

A cleaner response failure does not require wrong-token swapping. Suppose the dense pair is

\[
p_D^-=(.4,.5,.1),\qquad p_D^+=(.4,.1,.5),
\]

but the sparse model predicts \(p_S^-=p_S^+=(.4,.5,.1)\). The endpoint scalar A and the scalar response C can both be zero in a gold-vs-rest construction if the gold class is the first coordinate, yet the sparse model completely misses the dense context-induced switch. This is why C must be evaluated on paired state changes, not inferred from endpoints.

## Optional vector metric: useful diagnostic, not first method

For dense and sparse logits \(z_D^k,z_S^k\), define \(e_k=z_S^k-z_D^k\), freeze

\[
q=\tfrac12(p_D^-+p_D^+),\qquad M_q=\operatorname{diag}(q)-qq^\top,
\]

and use

\[
L_q=\tfrac12(e_-^\top M_qe_-+e_+^\top M_qe_+)
 +\lambda(e_+-e_-)^\top M_q(e_+-e_-).
\]

The quadratic is exactly the teacher-weighted pairwise margin error,

\[
e^\top M_qe=\tfrac12\sum_{i,j}q_iq_j(e_i-e_j)^2,
\]

and is invariant to a common logit shift. It is also the local Fisher/second-order KL geometry around q, not a new theorem. Since q is frozen from the dense teacher, the candidate cannot manipulate its own metric.

Do not form a per-pair \(Q=U^\top M_qU\) matrix. Compute the quadratic from the full logits as \(\sum_i q_i e_i^2-(\sum_iq_i e_i)^2\). If the existing implementation already materializes the vocabulary logits, this is an additional reduction; if it previously stopped at hidden states, the LM-head computation and activation storage can be material. Top-k plus tail is an approximation and must be compared with exact full-vocabulary scoring.

The shared q has a nontrivial geometric choice:

\[
M_{(p_-+p_+)/2}
=\tfrac12[M(p_-)+M(p_+)]
 +\tfrac14(p_--p_+)(p_--p_+)^\top.
\]

Therefore it is not exactly the mean endpoint Fisher matrix. Raw \(M_q\) also downweights low-entropy, nearly one-hot states because \(\operatorname{tr}(M_q)=1-\|q\|_2^2\). That may erase high-confidence committed-token errors. Trace normalization or a uniform tail floor would add a new weighting choice and should be a diagnostic, not silently included.

The q metric should therefore be a later readout ablation: compare scalar A+C and q-weighted A+C with the same masks, search cap, pair bank, and exact budget. It earns a separate claim only if it selects different masks and improves held-out full-vocabulary KL/NELBO or robustness. A lower q loss alone is insufficient.

## Minimal discriminating comparison

Freeze model/revision, calibration documents, natural nested pair construction, Wanda within-row support, exact global budget, candidate pool, and search compute.

1. A-only hard exact-budget search, yielding \(M_A\).
2. Unconstrained scalar A+C hard search.
3. Scalar C minimization subject to \(A\le A(M_A)+\epsilon\), with \(\epsilon=0\) plus numerical tolerance.
4. Pair-shuffled version of the constrained C objective, preserving endpoint marginals but breaking the true context correspondence.

Evaluate each frozen mask on independent pair documents, full-vocabulary teacher-student KL, held-out WikiText NELBO, and the pre-registered GSM8K view. The constrained branch is informative even if it fails: no feasible one-swap improvement means C cannot improve at the endpoint floor under that support/search family. If it chooses the existing A+C mask and has no held-out advantage, it is a reformulation rather than an upgrade. If true-pair and shuffled-pair perform similarly, C is using state marginals rather than conditional response.

## Borrowed versus possible new contribution

- The Fisher quadratic \(e^\top(\operatorname{diag}q-qq^\top)e\) is standard probability/logit geometry and local KL curvature.
- Sobolev Training already combines teacher outputs with input derivatives and discusses stochastic Jacobian matching for distillation: [Czarnecki et al., Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859).
- 2ndMatch already matches dense/pruned diffusion sensitivity through directional \(J^\top J\) terms and random projections: [Zheng and Shlizerman, 2ndMatch](https://arxiv.org/html/2506.05398).
- FAIR-Calib already uses teacher-probed frontier/reliability weighting and weighted hidden-state MSE as a KL-consistent surrogate for DLM post-training quantization: [Huang et al., FAIR-Calib](https://arxiv.org/html/2606.06547).
- Existing exact-budget compression search, including level-switch/multi-stage fitness, is available in EvoPress: [EvoPress](https://arxiv.org/html/2410.14649).

The possible narrow contribution is the fixed static weight-mask allocation problem: natural DLM masked-context pairs define C, while an exact-budget hard allocator preserves endpoint A. This is a hypothesis until matched held-out NELBO/downstream evidence exists. No AR failure, DLM-specific superiority, causal commitment preservation, or optimality claim is supported yet.

## Decision

Prioritize one concrete implementation axis: scalar A+C with measured feasible hard-mask exchanges plus the endpoint-anchored C constraint. Defer the q-weighted full-vector metric to a controlled objective ablation. Do not combine it with count-coverage, clock features, SAE features, rollout values, or a new optimizer in the same method.
