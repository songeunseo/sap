# Non-evolution exact-budget allocation design

Date: 2026-09-18  
Status: design recommendation only; no model forward, mask generation, benchmark, or Obsidian write

## Decision

Use a bounded hard-mask exchange search as the main non-evolution alternative to Lua-style sigmoid allocation and EvoPress evolution. The search backend is a standard discrete direct-search/coordinate-exchange construction; it is not an optimizer novelty claim. The possible project contribution is narrower: whether the already proposed vector endpoint plus paired conditional-response objective gives useful allocation decisions when it is measured on the assembled sparse DLM at the final 50% budget.

Keep the objective fixed. For paired contexts \((x_0,x_1)\), dense reference \(F_D\), and candidate sparse mask \(M\), measure the existing vector loss

\[
 L(M)=\operatorname{mean}\left[
 \tfrac12\|F_M(x_0)-F_D(x_0)\|^2+
 \tfrac12\|F_M(x_1)-F_D(x_1)\|^2+
 \lambda\|[F_M(x_1)-F_M(x_0)]-[F_D(x_1)-F_D(x_0)]\|^2
 \right].
\]

The readout, scaling, pair construction, and \(\lambda\) are frozen before search. There is no task-loss term, clock proxy, coverage term, role requirement, weight update, or dynamic denoising mask. Original weights and the within-row Wanda ranking remain frozen. The negative role-context refresh result and weak C1 coverage result remain controls to report, not ingredients to reintroduce. Every candidate uses one static mask for every denoising state and exactly the same weighted 50% budget.

## Exact candidate representation

Let projection \(p\) range over the 224 projections in the 32 layers. Let \(k_p\) be the number of low-ranked weights removed in each output row and \(a_p\) the number of output rows in projection \(p\). A mask is feasible only when

\[
 0\leq k_p\leq k_p^{\max},\qquad \sum_{p=1}^{224} a_p k_p=K_{50},
\]

where \(K_{50}\) is the pre-registered global 50% pruning count (currently 3,489,660,928 in the native pipeline). A move is an integer vector \(\delta\) with bounds respected and \(\sum_p a_p\delta_p=0\). Positive \(\delta_p\) removes more weights per output row; negative \(\delta_p\) restores them. Here \(k_p^{\max}\) is the projection input width. Restores and removals always occur at the current within-row Wanda boundary, so no support ranking is learned.

The smallest proposal is a pair exchange: a donor removes \(u\) additional weights per output row and a receiver restores \(v\) weights per output row with \(a_d u=a_r v\). Different projection output-row counts often make a one-step-for-one-step move impossible. Do not round percentages and repair the budget afterward, since that silently changes unrelated projections. Instead, enumerate exact integer exchanges and use a small feasibility DP for two- to four-projection bundles satisfying the equality exactly. If the least feasible bundle is larger than the current trust radius, reject that move and enlarge the radius only under the pre-declared schedule.

The existing 32-rate/64-block grouped representation and the 224-projection representation are different search spaces. A 32-coordinate run is a coarse capacity backend whose fixed template maps one layer rate to projection row counts; a 224-coordinate run searches projection counts directly. Compare them as granularity ablations or report a two-stage coarse-to-fine procedure. Do not call the finer result an optimizer improvement when the candidate family itself is larger.

## Bounded hard-mask exchange search

Call the following an implementation label, not a named algorithm claim.

1. **Initialization.** Reconstruct and hash the native Uniform50 mask, model revision, within-row ranking, state bank, pair list, and exact count. Cache all dense reference vectors. Set the incumbent to this mask. Split calibration pairs by document into a search bank and a sealed holdout bank; the latter is never used to propose, rank, or stop a move.

2. **Proposal.** At incumbent \(M_t\), form feasible integer pair exchanges and, only when needed, two-exchange bundles. Use a trust radius measured in per-row quota steps or weight quanta, plus a maximum number of changed projections. For an illustrative bounded study, poll at most 32 pair proposals and 8 bundle proposals per sweep, with fixed depth/type strata and a fixed seed; these are accounting examples, not a fixed protocol. The poll must contain some random or uniformly stratified proposals so a proposal score cannot silently define the result.

   A gradient-informed shortlist is optional. On the incumbent only, differentiate the *same fixed* \(L\) with respect to effective weights \(z_j=m_j w^{orig}_j\). Use the frozen original nonzero weight and gate derivative \(s_j=w^{orig}_j\,\partial L/\partial z_j\) for each boundary weight in the fixed per-row ranking; do not use the current zeroed parameter as the original weight. Rank feasible swaps by the summed restore-minus-prune \(s_j\) over all changed weights, then measure the top proposals plus stratified random controls. This uses gradients only to choose what to measure; it neither changes weights nor accepts a candidate from its linear prediction. Conceptually, only boundary gate derivatives are needed; a dense full gradient matrix is not required. If backward memory or compute is unavailable, use the fixed stratified poll directly. SparseGPT is precedent for calibration-aware second-order mask selection, but its weight reconstruction/update path is outside this frozen-weight setup ([SparseGPT](https://arxiv.org/abs/2301.00774)).

3. **Measurement.** Materialize every proposed hard mask in the current jointly sparse background. Evaluate the complete vector objective \(L\) on the full search bank, not a sum of projection losses or a 48/52% extrapolation. Dense vectors are cached; each candidate needs one sparse forward per unique context. With \(N\) paired contexts and no shared endpoints, this is \(Q=2N\) forward equivalents per candidate; for the existing 80-pair bank, 160 forward equivalents. Reuse exactly the same ordered pairs and any stochastic draws for all candidates. A candidate's paired difference is measured against the incumbent on the same examples.

4. **Acceptance.** Let \(\Delta(M')=L_{\mathrm{search}}(M')-L_{\mathrm{search}}(M_t)\). Accept the best feasible proposal only when \(\Delta\leq-\epsilon\), with \(\epsilon\) fixed before search (numerical tolerance for a deterministic vector bank, or a predeclared uncertainty margin if the readout is stochastic). Keep the complete candidate evaluation as the acceptance value; the gradient score and any bundle edge sum are proposal scores only. Cache the accepted incumbent and its per-pair vectors.

5. **Radius and bundle rule.** After an accepted pair move, repeat at the same radius and permit one larger radius on the next sweep, capped at the declared maximum. If no pair move improves, poll the prespecified two-exchange bundle set once using the best boundary coordinates from the pair poll and the random controls. This is the only escape attempt. If no bundle improves, shrink to the minimum radius and stop. For illustrative accounting only, four sweeps with 160 measured candidates gives at most about 25,600 forward equivalents for \(N=80\), before final holdout evaluation; these numbers are not a fixed protocol. A gradient shortlist adds the measured forward and backward costs over the 160 contexts per sweep; record \(F\) and \(B\) separately and do not assume a fixed backward-to-forward ratio. It does not replace hard-candidate measurement.

6. **Stopping and final selection.** Stop at the first of: no improving pair and no improving bundle at minimum radius; four accepted/attempted sweeps; or the fixed candidate-evaluation budget. Select the incumbent using search-bank loss only, then evaluate the initial Uniform50 mask and the selected mask once on the sealed document-held-out bank. For downstream validation, use the agreed independent WikiText NELBO protocol and shared exact-k MC draws. Never reopen the holdout after seeing its result.

This is a trust-region interpretation of discrete direct search: exchange radius is the mesh, feasible exchanges are polling directions, and a successful move expands or retains the radius while a failed poll contracts it. MADS gives the relevant granular/discrete direct-search framework ([Audet, Le Digabel & Tribes, 2019](https://doi.org/10.1137/18M1175872)). Its continuous/noisy convergence conditions should not be transferred here: with a finite hard-mask space and a fixed deterministic calibration bank, exhaustive polling at one radius gives only a local optimum for that neighborhood; MC noise or a sampled poll gives no global guarantee.

## Where DP fits, and where it does not

Candidate-level resource-allocation DP is appropriate when every unit has a small discrete option set \(q\), an additive resource \(c_{p,q}\), and an additive surrogate cost \(e_{p,q}\). The recurrence then returns the best option combination at an exact budget. SPDY states these assumptions explicitly: layer execution cost and model error are treated as additive, with a discrete option set; its DP is exact for that surrogate and its learned local search supplies the scores ([SPDY](https://proceedings.mlr.press/v162/frantar22a/frantar22a.pdf)).

Use DP here for exact integer feasibility, or as a clearly labeled proposal surrogate over pre-measured per-projection options. Do not call its result the optimum of the vector objective: the current project has direct evidence against that assumption. In the corrected role-exchange analysis, both local reconstruction terms improved while functional KL worsened in 1,226 of 3,026 bundle-state cases (40.52%), and the saved \(\pm2\) percentage-point local-response probe predicted the final \(\pm5\) allocation in the wrong direction. The latter rejects that extrapolation, not every local model; it does justify measuring assembled candidates. Any DP recurrence that repairs an infeasible proposal by modifying unselected projections must be treated as a different candidate and logged explicitly.

## Greedy failure, bundles, and beam size

A one-exchange greedy method can fail when two changes are jointly useful but each is harmful alone, when a donor/receiver interaction reverses sign in the sparse background, or when exact quota arithmetic makes the useful move unavailable at radius one. The first case is a generic failure mode, not a finding from the saved audit. The project has evidence of sparse-background sign changes and poor local extrapolation, but not a direct finding that a jointly helpful move is individually harmful. A small measured bundle poll is therefore a conditional diagnostic if one-exchange polls plateau or quota arithmetic blocks useful moves; it is not a default claim. It tests non-additivity while keeping the search interpretable and bounded.

A broad beam is not the default. A beam of 2–4 incumbents at one extra depth is defensible only if the one-exchange poll repeatedly plateaus and the budget allows each beam member's complete hard-model evaluations. It multiplies the same-bank selection burden and is still a local search heuristic. Prefer a measured two-exchange bundle over a broad beam; if a beam is used, fix its width, depth, proposal strata, and evaluation budget before looking at results. EvoPress already establishes global candidate-vector evolutionary search as existing machinery ([EvoPress](https://arxiv.org/abs/2410.14649)); beam or bundles must not be presented as a new optimizer.

## Selection bias and paired reuse

The minimum of many candidate losses on one bank is optimistically biased, and adaptive sweeps compound that bias. A fixed proposal bank does not remove it. The same bank is valuable for paired comparisons because common examples cancel variance, but it is not independent evidence after the search has selected a mask. Report the number of proposed and measured candidates, sweeps, strata, and rejected proposals. Use document-cluster bootstrap for paired differences; do not treat individual tokens, contexts, or candidates as independent replicates.

The sealed document holdout is the protection against candidate/subset selection bias. If a gradient shortlist is used, include random/stratified proposals and compare shortlist coverage, but do not claim an unbiased gradient estimate of hard-mask utility. The final NELBO bank must remain separate from both proposal and acceptance data. A positive search-bank reduction with no holdout reduction may reflect selection overfit, bank mismatch, or objective-to-downstream mismatch; it is not evidence against the vector objective itself.

## Recommended comparison matrix

Keep one objective, one native support family, one exact 50% count, one state/pair bank, and one forward accounting across these arms:

| Arm | Allocation mechanism | What it tests |
|---|---|---|
| Frozen Uniform50 | existing fixed allocation | reference mask |
| Hard exchange, 32 coordinates | bounded direct search over fixed layer/block template | optimizer under coarse granularity |
| Hard exchange, 224 coordinates | bounded direct search over projection row counts | finer allocation capacity |
| Hard exchange + gradient shortlist | same as above, gradient only for proposal ranking | proposal efficiency, not a new objective |
| DP surrogate + hard verification | additive option-cost DP, then complete-model measurement | whether additive allocation is adequate |
| Lua sigmoid or EvoPress reference | existing methods with provenance and compute recorded | external allocation baseline |

The primary claim, if supported by held-out NELBO, should be: the fixed DLM conditional-response objective provides useful information for exact-budget static allocation under the native Wanda support family. The search procedure itself is borrowed discrete direct search, and DP is exact only for its stated additive surrogate.

