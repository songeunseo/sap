# A+C framework upgrade review — 2026-09-22

Status: research design and frozen-artifact CPU analysis only. No new model forward, pruning mask, GPU job, GSM8K evaluation or NELBO run. The parent verified Obsidian with get_sync_status and Research-State read. Three gpt-5.6-luna/xhigh agents independently reviewed evidence, objectives/prior art, and the solver; parent corrected factual/mathematical errors rather than treating agreement as evidence.

## Decision

Keep conditional-response preservation as the working hypothesis. Make actual hard-mask evaluation the common allocator, and introduce **endpoint-anchored response minimization as a competing candidate**, not an already superior replacement for A+C. Its question is: at a fixed static weight budget and an explicitly preserved endpoint-fidelity floor, does spending the remaining allocation freedom on context response improve unseen denoising quality?

The first comparison retains the existing scalar readout and natural nested context pairs. Vector readout, probability weighting, equal-count pairs, independent projection allocation, gradients, and bundles are separate axes. The user asked for an idea upgrade, not execution; no new experiment was started.

The earlier conversational forecasts 'A+C should beat Role at50, Role should beat A+C at65' are not supported by matched head-to-head evidence. In particular A+C65 has not been measured. No cross-sparsity method ranking or numeric success probability follows from the current observations.

## What the three agents contributed

| Reviewer | Main contribution | Parent disposition |
|---|---|---|
| Evidence | Mini61 vs55 is inconclusive; original pairs confound added evidence and mask-count progress; require independent quality evaluation | Accept limitations, but reject replacing natural pairs only in the C arm and reject calling count control pure semantics |
| Objective/prior art | Endpoint floor provides a concrete alternative to arbitrary A+lambda C weighting; teacher-weighted vector geometry has blind spots | Adopt constrained C as challenger; keep ordinary A+C; defer probability weighting |
| Solver | Direct feasible hard-mask exchanges, bounded candidate screening and common full-bank acceptance | Use as shared backend; account for exact mask family, all evaluations and adaptive-selection limits |

## Observations and a new small CPU audit

Existing same-native-family mini100: Uniform54, A55, A+C61; ACvsA has9 rescued/3 regressed, exact p=.145996. Historical cached Uniform62 has different ranking/mask construction. A/AC full evaluation was stopped and A/AC NELBO remains unmeasured.

Parent reanalyzed only frozen JSON files with Python standard library (one CPU process). 33 referenced receipt hashes match; stored probe costs and ideal rates are reproduced. No tensor, model or new mask is evaluated. See `ac_probe_stability_2026-09-22.py` and `.json`.

- A vs A+C marginal-cost Spearman:0.942815; A vs C:0.153592.
- Adding C changes27/32 ideal layer rates, mean absolute change0.7863 percentage points, max2.2581 points. These are ideal rates, not evidence of224 independent decisions.
- On the same eight calibration spans, the assembled AC mask has lower C than the A mask on7/8, and lower A+C on5/8.
- Leave-one-span recomputation gives AC rank correlation0.923–0.987 versus all spans. The sign of the A-to-AC rate change agrees on66.7–85.2% of the27 originally changed layers. The resulting leave-one-span masks are NOT evaluated; this is not holdout performance.
- Uniform-to-AC total surrogate decrease splits into93.45% decrease of its A term and6.55% decrease of its C term. This is only arithmetic accounting. It is NOT causal attribution: C can change the selected allocation and thereby reduce A.

These findings keep C plausible as useful allocation information, but establish neither its downstream value nor the optimal weighting/solver.

## First complete algorithm to compare

Let M encode one static removal mask (1 means removed), W0 be the original weights, and B be the exact target number removed. A(M),C(M) are the existing means over paired endpoint errors, with lambda1 when A+C is used. The scored query positions are already identical at both endpoints in the current code. The reveal operation changes the surrounding visible content and mask count.

Obtain/freeze an A-only candidate M_A using a declared finite calibration budget. It is a found reference, not a global argmin. All three continuation arms start from this same M_A, with the same frozen within-row Wanda ordering, candidate domain, original surviving weights, state bank and additional compute cap:

1. **A continuation:** minimize A(M), to make the endpoint-only control strong.
2. **A+C continuation:** minimize A(M)+C(M).
3. **Anchored C continuation:** minimize C(M) subject to A(M)<=A(M_A)+epsilon_A and sum(M)=B.

Use epsilon_A=0 in mathematical definition and a fixed, separately measured numerical comparison tolerance in implementation. A positive quality tolerance would be another declared hyperparameter; do not tune it on mini100. The anchor is fixed at M_A, not reset at every accepted step. A fixed anchor permits some A tradeoff relative to an improved intermediate mask while staying below the original floor; a per-step Pareto guard is more restrictive and a different algorithm.

The anchor construction cost counts toward every arm. If a stored A mask is reused, report its original construction cost plus the new continuation cost; no free warm-start comparison against methods charged for initialization. The three arms have the same starting candidate domain; their later trajectories may differ. Equal evaluation budget does not imply equal wall time if cache reuse differs, so record both.

### Actual exchange and acceptance

- Propose a bounded set of count-preserving trades: restore original weights at one unit's Wanda boundary and remove exactly the same number at another unit's boundary.
- Materialize each candidate's complete hard mask. Preserve the exact count before evaluating. Score candidates in the current jointly sparse model, never by merely adding old Uniform-background costs.
- A small, shared, stratified subset may prioritize candidates. Keep both endpoints and the same query set together. Every accepted candidate is compared with its incumbent on the SAME complete optimization bank.
- In A and AC arms, accept only a decrease in that arm's scalar objective beyond numerical tolerance. In anchored C, require lower C and the fixed A bound. Retain the incumbent otherwise.
- After acceptance, cache the new incumbent's per-pair A/C values; old marginal estimates are priorities only. Any candidate accepted later must still be evaluated in the new model. A rotating exploration component prevents a stale predictor permanently excluding directions.
- Shrink the integer trade size after an unsuccessful bounded poll. Stop at the declared state-evaluation budget, or a failed minimum-size poll. This establishes no improvement among fully evaluated candidates; it does not establish local/global optimality of the full space.

### Exact count family: a subtle implementation constraint

For a projection, removing k entries per output row removes out*k weights. Actual equal-count changes, not equal percentages, define a trade.

In this model a layer contains218,103,808 prunable weights. The input widths are4096 and12288. A tied layer step adds d to each4096-width row-prefix count and3d to each12288-width count; it changes that layer's removed count by53,248*d. Applying opposite steps to two layers preserves global sparsity exactly. Starting from M_A preserves any existing rounding offsets and defines an explicit restricted lattice. This lattice must be shared by the three continuation arms.

The historical rank allocator's64(block,input-width)-group DP may produce counts not in the zero-offset32-template lattice. Therefore do not claim that a Uniform-start pure-template solver can reproduce every historical A/AC mask. Use the common anchored lattice for the continuation comparison, or separately define a shared quantized-template rank baseline. Historical61 is a descriptive reference unless its exact mask belongs to the compared domain. A proposed alternative that calls the old DP after changing rates must record and evaluate ALL groups changed by rounding; it cannot describe the result as a two-layer-only trade.

The first design does not require224-projection freedom, bundles, STE or a new sparsity schedule. Those enlarge capacity or compute and need their own comparison.

### Illustrative compute schedule, not an executed configuration

With the existing80 pairs, a full candidate costs160 sparse state evaluations before batching/prefix reuse. A poll of32 candidates on16 pairs costs1,024 evaluations. Completing four finalists on the remaining64 pairs costs512 more if screen values are cached, for1,536 per poll, plus160 to initialize the incumbent. If the finalists are recomputed instead, cost is1,664 per poll, with the incumbent still cached. Count all rejected candidates, refreshes, teacher/setup work and any backward passes.

A backward pass is not part of the proposed default. Prefix reuse is valid only before the first changed block; the entire bidirectional suffix must be recomputed. Candidate count screening costs are not independent-generalization evidence. No numerical poll cap or radius in this design is a claim of optimality or an already launched setting.

## Why the anchored candidate might help, and why it might fail

The motivation is explicit: prevent a C gain from being purchased by an unbounded loss of endpoint fidelity, while testing whether response can guide allocation beyond an endpoint-only baseline. It also makes the reference fidelity level interpretable instead of treating lambda1 as a derived law.

It may fail because the finite A-only anchor lies at a local endpoint minimum and no one-trade candidate satisfies the A bound. It may also miss a useful tradeoff that ordinary A+C accepts. The scalar readout remains blind to wrong-token rearrangements, and both A and C can improve on calibration while NELBO/GSM8K worsen. Thus 'A floor' is a calibration-fidelity constraint, NOT a safety guarantee for actual task accuracy. Existing AC already improves both A and C on the old bank, so a new constraint may be redundant. If it merely reproduces AC or stalls, keep the simpler unconstrained AC arm.

## Objective refinements kept separate

A concrete scalar blind spot: dense predictions over [gold,b,c] change from(.4,.5,.1) to(.4,.1,.5), while sparse predictions stay(.4,.5,.1). Gold-vs-rest log-odds is unchanged everywhere, so scalar A=C=0 although the most likely token and its context response differ. This is a constructed example, not a measured failure frequency.

Centered full-vector logits detect this, but uniform weighting over the vocabulary can emphasize an irrelevant tail. A teacher-weighted alternative uses q=(p_D0+p_D1)/2 and M_q=diag(q)-q q^T, with e^T M_q e=sum(q e^2)-(sum(q e))^2. It measures teacher-weighted logit-margin distortion and can be evaluated with O(V) reductions after logits. However M_q can downweight consequential high-confidence errors; it is not an automatic fix. Also M_meanp = mean(M_p) + .25(p_D0-p_D1)(p_D0-p_D1)^T, so the shared metric is not exactly the mean endpoint Fisher. Do not call it a free constant-Q hidden-space shortcut.

Use vector A-only, vector A+C and full-distribution KL with the SAME backend if the scalar continuation study motivates changing readout. Exact fixed-head Q from the Sept18 report is available for unweighted centered-logit squared error only, subject to full cost accounting.

Keep natural nested reveal pairs in the first solver/constraint comparison. Count-matched alternative visible contexts are a separate diagnostic, using the same common masked queries and the same endpoints across A and AC arms. Count matching does not isolate pure semantics. A broken-pair null needs a frozen query-compatible common coordinate system and endpoint marginals; blindly shuffling current variable-length gold-logodds vectors across spans is invalid. If the null only establishes that pairing matters, do not reinterpret that as a complete content/progress mechanism proof.

## Success criteria and novelty

Freeze masks after the common search budget. Measure pair losses and NELBO on documents excluded from calibration/selection, using matched draws and actual hard masks. GSM8K mini100 remains development; full or independent task evaluation provides the next capability check. NELBO and task accuracy can disagree and should be reported separately. Neither65/100 nor a better calibration C is a search stopping criterion.

A meaningful positive is anchored C or ordinary AC beating the strengthened A continuation on independent quality at equal sparsity and comparable total compute; a conditional-response claim additionally needs a valid pairing control. If all methods improve similarly with the new solver, the gain belongs to optimization, not demonstrated C-specific information. If constrained C is no better or repeatedly infeasible, do not add further guards automatically.

Output-plus-derivative matching is established in [Sobolev Training](https://arxiv.org/abs/1706.04859). [2ndMatch](https://arxiv.org/html/2506.05398) matches sensitivity in pruned image diffusion models and reports first-order matching failing to improve its output-KD baseline (FID5.05 to5.14), while its second-order variant helps. [EvoPress](https://arxiv.org/html/2410.14649) already uses compression-preserving exchanges and staged fitness evaluation. Constraints, Fisher geometry, vector outputs, and deterministic search are not independently novel. The possible contribution is demonstrated additional quality or efficiency from controlled DLM context-response preservation in one fixed static-mask allocation framework.

## Artifacts

- `upgrade_evidence_2026-09-22.md`
- `upgrade_objective_2026-09-22.md`
- `upgrade_solver_2026-09-22.md`
- `ac_probe_stability_2026-09-22.py`
- `ac_probe_stability_2026-09-22.json`

Parent synthesis takes precedence over draft agent recommendations, especially unsupported cross-sparsity forecasts, proposed pair-bank changes, and candidate-domain equivalence claims.

