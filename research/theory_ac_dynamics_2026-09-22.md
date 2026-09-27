# Dynamical-systems and model-reduction theory for the original A+C

Date: 2026-09-22

Status: literature review and design analysis only. No model forward, mask, GPU job, NELBO run, or GSM8K evaluation was started. This note is standalone; it does not modify the shared Obsidian research state.

## Question and current evidence

The current calibration readout is a scalar gold-versus-rest log-odds at the same still-masked query. For pair \(i\), \(x_{i,0}\) is the original partially masked state and \(x_{i,1}\) reveals a small subset of gold tokens. Let

\[
\delta_{i,e}(M)=f_M(x_{i,e})-f_D(x_{i,e}),\qquad e\in\{0,1\},
\]

where \(D\) is dense and \(M\) is the static sparse mask. The existing objective is

\[
A(M)=\frac12\mathbb E_i[\delta_{i,0}^2+\delta_{i,1}^2],
\qquad
C(M)=\mathbb E_i[(\delta_{i,1}-\delta_{i,0})^2].
\]

Thus \(A+C\) sees endpoint fidelity and the change in the scalar readout under a controlled visible-context change. The native 50% mini screen is Uniform 54/100, A-only 55/100, and A+C 61/100; A+C versus A has 9 rescued and 3 regressed items, with exact paired \(p=0.145996\). Full A+C and independent NELBO evidence do not exist. The existing allocation uses frozen original surviving weights and Wanda within-row ranks, layer 48/52% probes in a uniform 50% background, a 45--55% rank map, and an exact global budget.

The proposed theory question is narrower than “find a better norm.” Does A+C give equal importance to directions whose effects have unequal downstream influence? There are two distinct answers:

1. At the two measured states, \(f_M\) is already the output after the complete transformer. Downstream propagation through all blocks is therefore already present in the observed scalar \(\delta\). An observability weight computed from the same final scalar and same state adds no information; it is a re-expression of A or C.
2. A static mask can alter hidden states in ways that are quiet at these two endpoints but become large on later denoising states, other queries, or a hard token-commit branch. Those are unobserved goals. A fixed dense denoising trajectory supplies separate future input states; it does not pass the prior call's hidden state into the next call. Therefore a suffix Jacobian inside one transformer call is not a causal \(t\to h\) denoising transition map. A cross-denoising propagation Gramian would require a differentiable stateful or continuous-relaxation transition, which is absent here.

This distinction rules out an unexplained Gramian multiplier. Any useful adaptation must define (i) the state/goal horizon, (ii) how a pruning action injects an error into that state, and (iii) the exact hard-mask acceptance objective. The ideas below preserve the current A+C arm as a control.

## What the primary control literature actually supplies

### Balanced truncation and Hankel energy

For a stable discrete-time linear system

\[
z_{k+1}=Fz_k+Bu_k,\qquad y_k=Cz_k,
\]

the infinite-horizon reachability and observability Gramians are

\[
P=\sum_{k\ge0}F^kBB^\top(F^\top)^k,
\qquad
Q=\sum_{k\ge0}(F^\top)^kC^\top CF^k.
\]

They solve the discrete Lyapunov equations \(P-FPF^\top=BB^\top\) and \(Q-F^\top QF=C^\top C\). A balanced realization makes \(P=Q=\operatorname{diag}(\sigma_1,\ldots,\sigma_n)\); the \(\sigma_j\) are Hankel singular values. Balanced truncation keeps states that are both reachable from inputs and observable at outputs. For stable LTI systems, the classical bound is

\[
\lVert G-G_r\rVert_{\infty}\le 2\sum_{j>r}\sigma_j,
\]

while Glover’s Hankel-norm work characterizes optimal Hankel-norm approximations and their \(L_\infty\) error bounds. These are state-reduction results, not bounds for deleting neural-network weights.

For a finite, time-varying horizon the relevant object is instead

\[
Q_k^{(H)}=\sum_{h=k}^{H}\Phi_{h,k}^{\top}C_h^{\top}R_hC_h\Phi_{h,k},
\]

where \(\Phi_{h,k}=F_{h-1}\cdots F_k\). A perturbation \(b_k\) injected at time \(k\) has linearized future output energy \(b_k^\top Q_k^{(H)}b_k\). A finite reachability Gramian similarly sums \(\Phi_{k,j}B_j\Sigma_{u,j}B_j^\top\Phi_{k,j}^\top\). This is the concrete part that can transfer: an explicit future-output metric and finite horizon, rather than a generic \(J^\top J\) score.

The transfer boundary is decisive. A transformer pruning mask is a static parameter choice, not a reduced state realization. To use \(Q_k^{(H)}\), one must identify a state trajectory, the perturbation injected by a candidate weight change, and future outputs that matter. No Hankel singular-value theorem follows unless the model is replaced by a stable linear or locally linear input-output system with a declared norm.

### Empirical and differential gramians for nonlinear systems

Lall, Marsden, and Glavaski construct an approximately balanced realization for nonlinear control systems from simulation or experimental input-output data. The point is empirical: the data identify dynamics relevant to the observed input-output map without solving a nonlinear balancing PDE. Condon and Ivanov similarly define empirical controllability/observability constructions using a generalized linear-time-varying fundamental solution. Kawano and Scherpen later define differential gramians along a fixed nonlinear trajectory and show that the variational system can compute them along that trajectory.

The usable lesson is trajectory locality. An empirical gramian is a statement about the sampled operating region and excitation directions. It is not a global transformer importance score. In this project, the natural reveal pair can provide an empirical context excitation, but only on the calibration states. A dense generated denoising trajectory would be a different excitation distribution and must be labeled as such. If its future states are simply replayed as independent calls, they form a larger measurement bank, not a causal variational system.

### Goal-oriented error estimation and adjoints

Becker and Rannacher’s dual-weighted-residual (DWR) method starts from a nonlinear residual equation \(R(u)=0\) and a quantity of interest \(J(u)\). A linearized adjoint \(z\) satisfies

\[
R'(u_h)^\ast z=J'(u_h),
\]

and the output error has the schematic representation

\[
J(u)-J(u_h)=R(u_h)(z)+\mathcal R_2,
\]

where \(\mathcal R_2\) is a second-order linearization remainder. A local residual is weighted by the adjoint because it measures its effect on the chosen goal, rather than its size in a global norm.

For the current calibration goal one could set \(J_{AC}=(\delta_0,\delta_1,\delta_1-\delta_0)\) with \(R_{AC}=\operatorname{diag}(1/2,1/2,1)\). This gives a principled way to attribute a layer or candidate boundary perturbation to the existing objective. It does not create new information if the candidate is already fully evaluated at the final scalar output: the adjoint simply factors the same end-to-end derivative. A new signal appears only when \(J\) includes future states, additional queries, or a deployment goal absent from the current bank.

### Contraction and incremental stability

Lohmiller and Slotine’s contraction analysis studies the evolution of an infinitesimal displacement \(\xi\) under a nonlinear flow. In a state metric \(M(x,t)\succ0\), a sufficient differential inequality is

\[
\dot M+J_f^\top M+MJ_f\preceq -2\lambda M,
\]

which implies exponential contraction of nearby trajectories. In discrete time the analogous condition is

\[
J_t^\top M_{t+1}J_t\preceq \rho_t^2M_t.
\]

Tran, Rüffer, and Kellett distinguish discrete-time contraction, convergence, and incremental stability and give Lyapunov characterizations; the conditions concern all trajectories in a declared region, not one local perturbation.

If a pruned trajectory has a local forcing residual \(r_t\), the linearized error recursion gives the finite-horizon envelope

\[
\lVert e_H\rVert_{M_H}
\le
\left(\prod_{j=0}^{H-1}\rho_j\right)\lVert e_0\rVert_{M_0}
+\sum_{t=0}^{H-1}
\left(\prod_{j=t+1}^{H-1}\rho_j\right)\kappa_t\lVert r_t\rVert.
\]

This is a real derivation for a smooth branch. It says an early residual deserves more weight when later dynamics amplify it. It does not survive a hard decode branch flip: argmax/top-k/commit operations are discontinuous, and a small logit error can change the discrete trajectory. A hidden-state contraction estimate is therefore at most a local robustness diagnostic unless branch margins are separately controlled.

## Candidate 1: fixed-context future-goal A+C (conditional diagnostic)

### Definition

Call this **FG-A+C**. It is a possible diagnostic extension of the calibration bank, not a causal denoising Gramian and not a balanced-truncation theorem. It is included because it is the closest operational mapping of finite-horizon output energy; it is not selected as the main method.

For each independently materialized dense trajectory state \(s_{i,h}\), keep its query set and, where available, its paired context state with the same still-masked query. Use those states as additional fixed inputs and define future-bank scalar goals \(g_{i,h,e}\): the gold-versus-rest log-odds at the same query, plus the paired response difference. The original natural-pair goals remain mandatory. There is no hidden-state carry from \(s_{i,h}\) to \(s_{i,h+1}\), so \(h\) indexes observations, not a causal time transition. Define

\[
D_H(M)=\mathbb E_i\sum_{h\in\mathcal H_{\mathrm{future}}}
\left[
\tfrac12\bigl(\delta_{i,h,0}(M)^2+\delta_{i,h,1}(M)^2\bigr)
+\bigl(\delta_{i,h,1}(M)-\delta_{i,h,0}(M)\bigr)^2
\right].
\]

The candidate objective is

\[
L_{FG}(M)=A(M)+C(M)+\eta D_H(M),
\]

with \(\eta\) fixed before reading the development task results. The first useful screen should set \(\eta=1\) in the declared same-unit squared-output metric, rather than tune a continuous weight. This adds measured states to the bank; it does not claim that errors at \(h\) propagate causally to another denoising call.

Within each independent call, one can still form a layer-to-final-readout observability factor. Let \(b_{i,h,\ell,u,e}\) be the hidden-state residual injected at layer \(\ell\) by changing candidate unit \(u\), measured around the current incumbent. Let \(\Phi_{i,h,\mathrm{out}\leftarrow\ell}\) be the suffix tangent map inside that same call and \(C_{i,h}\) the derivative of the scalar goal readout. Then

\[
Q_{i,h,\ell}=
\Phi_{i,h,\mathrm{out}\leftarrow\ell}^{\top}
C_{i,h}^{\top}R_{i,h}C_{i,h}
\Phi_{i,h,\mathrm{out}\leftarrow\ell},
\]

with \(C_{i,h},R_{i,h}\) replaced by the corresponding endpoint/readout quantities. The local proposal energy is

\[
\widehat s_u=
\mathbb E_{i,h,e}
\left[
b_{i,h,\ell,u,e}^{\top}Q_{i,h,\ell}b_{i,h,\ell,u,e}
+(b_{i,h,\ell,u,1}-b_{i,h,\ell,u,0})^{\top}
Q_{i,h,\ell,C}
(b_{i,h,\ell,u,1}-b_{i,h,\ell,u,0})
\right].
\]

This is not an arbitrary Gramian score within a call: \(Q\) is fixed by a declared output and state metric. It is also not a cross-denoising propagation score. Use \(\widehat s_u\) only to select a bounded set of exact count-preserving Wanda-boundary exchanges. Materialize each complete jointly sparse candidate in the current sparse model and accept using \(L_{FG}\) on the full calibration bank. Cache incumbent per-state values; never add old 48/52% marginal costs as if they were additive.

If the only goal is the final scalar \(f\), \(Q=C^\top R C\) and FG-A+C reduces to a first-order factorization of the existing per-state output error. Any claimed benefit must come from the additional fixed-context states, not from relabeling \(J^\top J\). The added state bank plus exact exchange backend overlaps the project’s existing trajectory-bank and exact-exchange direction; it is not a distinct control-theory contribution.

### Cost and assumptions

The cost is substantial. The bank must contain held-out or development dense states and paired states; each state needs its own sparse forward evaluation and, for the proposal factor, a tangent/JVP or adjoint/VJP sweep through that call’s suffix. There is no legitimate saving from reusing a cross-call hidden state. A scalar goal can use one reverse sweep per state; multiple goals require multiple right-hand sides or a trace estimator. Every candidate still needs complete sparse forward evaluation on the optimization bank. The exact wall-time multiplier depends on batching and cache reuse and must be measured; the current saved scalar probes cannot supply these quantities.

The assumptions are: (a) local differentiability on a fixed decode branch within each call, (b) the sampled future states represent deployment, (c) a hidden residual from a boundary trade is meaningful before the complete nonlinear forward is re-evaluated, and (d) the chosen scalar goals are relevant to the independent quality metric. None supplies a causal cross-denoising theorem or a guarantee for hard token decisions.

### Falsifier and closest prior

Before any downstream claim, test whether the proposal score predicts exact candidate changes on held-out fixed-context states. If the Spearman rank of \(\widehat s_u\) against measured \(\Delta L_{FG}\) is no better than the existing layer 48/52 cost rank, or if accepted FG-A+C masks do not beat the A-only and natural A+C controls on an independent task metric at equal sparsity and comparable total compute, reject the extension. If FG-A+C improves only \(D_H\) while worsening natural A+C or task quality, call it a state-coverage trade-off. It must not be interpreted as evidence for a causal propagation Gramian.

Closest prior is DWR (goal-oriented residual weighting), together with empirical/differential gramians from Lall--Marsden--Glavaski, Condon--Ivanov, and Kawano--Scherpen. The combination is borrowed methodology; it is not a new claim about model reduction or pruning. It should not be described as novel merely because it is applied to DLM masks. In this project it largely duplicates the existing trajectory-bank plus exact-exchange direction, so it is not the root method choice.

## Candidate 2: contraction-envelope A+C (diagnostic, not selected)

### Definition

Call this **CE-A+C**. To make this a cross-denoising candidate, one would first need a differentiable stateful transition or continuous relaxation whose Jacobian maps the state at denoising step \(t\) to step \(t+1\). Along that declared transition, estimate a local metric \(M_t\succ0\) and induced gain \(\rho_{i,t}\) satisfying, approximately,

\[
J_{i,t}^{\top}M_{i,t+1}J_{i,t}\preceq \rho_{i,t}^2M_{i,t}.
\]

For a residual injected at \(t\), define its finite-horizon amplification envelope

\[
\kappa_{i,t}=\sum_{h=t}^{H}
\left(\prod_{j=t+1}^{h}\rho_{i,j}\right)^2.
\]

Use \(\kappa_{i,t}\) to weight the A+C residual for a candidate exchange, or equivalently to form a proposal score from its local residual \(r_{i,t}\):

\[
L_{CE}(M)=A(M)+C(M)+\eta\,
\mathbb E_{i,t}[\kappa_{i,t}\lVert r_{i,t}(M)\rVert_{M_{i,t}}^2].
\]

The complete hard mask remains the unit of acceptance. No branch-margin or short-commit rollout term is introduced here; those are already explored proposal families and cannot be counted as new.

### Cost, assumptions, and falsifier

Computing \(\rho\) requires Jacobian-vector norm estimates of the relaxed cross-denoising transition and a metric normalization. A cheap power estimate can be noisy; a reliable metric search is more expensive. The assumptions are stronger than FG-A+C: one smooth relaxed branch, a metric valid over the trajectory region, and residual forcing that is small enough for the linearized envelope. The current discrete sampler supplies no such transition Jacobian, and hard commits violate the smooth branch precisely where capability may change.

The first falsifier is structural: report the distribution of \(\rho_{i,t}\), the tightness of the one-step inequality, and the rate of branch flips under the exact candidate exchanges. If \(\rho\) is mostly above one, varies wildly across states, or the envelope fails to predict measured propagation, CE-A+C has no interpretable control meaning. Even if the envelope is locally valid, it is rejected as a method if CE-A+C gives no independent quality gain over A+C at equal budget and compute.

Closest prior is contraction analysis and discrete-time incremental stability (Lohmiller--Slotine; Tran--Rüffer--Kellett). The method provides a conditional robustness diagnostic, not a pruning guarantee. A fixed mask cannot be claimed to preserve generation decisions from a hidden-state contraction estimate.

## Decision

Select no control-theoretic replacement for the root method. FG-A+C is the only operationally measurable adaptation, but it is a fixed-context state-bank diagnostic that overlaps the existing trajectory-bank plus exact-exchange direction; it is not a causal propagation Gramian and is not selected as the root method. It directly tests state-coverage sensitivity only if an independently held-out bank is approved.

Do not select CE-A+C as the main method. The contraction assumptions are least credible for autorepeated nonlinear transformer dynamics and hard decode branches; it is useful only as a diagnostic for whether a local robustness envelope exists. Do not use balanced-truncation Hankel singular values directly for the static mask: the required input/state/output realization is absent, and the LTI error bound does not transfer.

Because the current project has no new causal transition bank or approval for new forward work, neither candidate is implemented or experimentally supported in this note. If a fixed-context future bank is not separately justified, reject both and retain original A+C as the control rather than inventing a Gramian multiplier from the existing layer probes. A genuine cross-denoising adjoint would require a continuous/stateful relaxation and is out of scope.

## Primary sources read: access and depth ledger

The ledger distinguishes body-level reading from abstract/metadata access. “Theorem-level” means that the stated theorem or derivation was inspected in the source body; it does not mean that every proof was independently rederived.

| Source | Access/read depth | Used here | Boundary |
|---|---|---|---|
| Moore (1981), *Principal Component Analysis in Linear Systems* | Author-hosted PDF found; abstract, introductory body, and Gramian/balancing equations inspected. No complete theorem-proof audit. | LTI reachability/observability and balanced-coordinate interpretation. | The transfer to a neural weight mask is an inference, not a Moore theorem. |
| Glover (1984), *All Optimal Hankel-Norm Approximations* | Publisher record/abstract and DOI only; full article was paywalled/inaccessible in this pass. | Only the abstract-level claim that Hankel-norm approximation and \(L_\infty\) bounds are characterized. | No equation or theorem from the full paper is relied on beyond the standard bounded claim; no derivation is claimed. |
| Lall, Marsden, Glavaski (2002), *A Subspace Approach to Balanced Truncation for Nonlinear Control Systems* | Author publication page and abstract only; full body not obtained. | Data/simulation-based approximate balancing for nonlinear input-output maps. | No detailed algorithm or theorem attribution beyond the abstract. |
| Condon and Ivanov (2004), *On the Empirical Balanced Truncation for Nonlinear Systems* | Bibliographic/abstract record plus searchable equation excerpts; not a complete author-PDF theorem audit. | Empirical/LTV-fundamental-solution motivation and trajectory-local limitation. | Detailed definitions and proofs were not used as if fully checked. |
| Kawano and Scherpen (2021), *Empirical Differential Gramians for Nonlinear Model Reduction* | Open arXiv record/abstract and accessible article snippets; core theorem statement was inspected, not every proof. | Fixed-trajectory variational-system construction and its local scope. | No global nonlinear guarantee is claimed. |
| Becker and Rannacher (2001), *An Optimal Control Approach to A Posteriori Error Estimation* | Publisher abstract plus accessible PDF text excerpts containing the DWR setup and adjoint/error-representation equations; full article proof audit was not completed. | Schematic goal-oriented residual weighting and adjoint interpretation. | Glover/Becker publisher access limitations mean this is not a claim of complete theorem verification. |
| Lohmiller and Slotine (1998), *On Contraction Analysis for Non-linear Systems* | Author-hosted MIT preprint body/abstract excerpts; contraction differential inequality and interpretation inspected, not every proof. | Smooth-branch contraction condition and finite-horizon amplification derivation. | No claim that a transformer satisfies a contraction metric. |
| Tran, Rüffer, Kellett (2019), *Convergence Properties for Discrete-Time Nonlinear Systems* | Open arXiv/preprint text excerpts; discrete-time definitions and Lyapunov characterization were inspected, not every proof. | Distinction between contraction/incremental stability and trajectory-region assumptions. | No transfer theorem to the DLM sampler is claimed. |

The report therefore uses full derivations only at the level explicitly marked above. In particular, the Glover and Becker publisher records alone would not justify a detailed theorem claim. Supplementary finite-horizon LTV model-reduction work was checked for the finite-window distinction, but it is not used to imply cross-denoising causality.
