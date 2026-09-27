# Allocation design review with the conditional-response objective fixed

Date: 2026-09-18. Status: design exploration, not implementation or an experiment.
The parent read the connected Obsidian Research-State and latest four-axis note before work. Three luna/xhigh agents independently reviewed continuous relaxations, discrete allocation, and the existing implementation. No GPU, model forward, new pruning mask, or benchmark was executed for this review.

## Scope

The user provisionally accepts the endpoint-plus-conditional-response objective but questions its optimizer and implementation. Hold the functional readout, pair bank, coefficient, aggregation, candidate ranking family, and compression target fixed when comparing optimizers. A change from the existing scalar gold-logodds to vector logits is a separate objective experiment. New same-mask-count context pairs are another separate axis. Neither is silently part of an optimizer comparison.

One static mask is used at all denoising states. Original surviving weights are frozen. The primary allocation-only comparison keeps the native within-row Wanda ordering fixed. Relaxing this ordering, adding weight compensation, or learning individual supports enlarges the method scope and needs a separately labeled comparison.

## What is actually established

- Existing native mini100 at 50%: Uniform54, A55, A+C61; A+C vs A is not conclusive (paired p=.145996). Historical cached Uniform62 belongs to a different ranking/mask construction. No full A/AC result and no A/AC NELBO measurement establish the new objective's benefit.
- Existing collection uses 32 blocks, each probed at48/52% in Uniform50; signed finite costs are converted to ranks and mapped to45–55%.
- `experiments/dlm_owl65/core.py:exact_row_counts` minimizes weighted rounding deviation under an exact count. This DP does not optimize A+C and is not evidence that the final mask is functionally optimal. It groups by block/input width, so a future224-projection implementation cannot reuse it unchanged and still claim independent projection allocation.
- Saved scalar response vectors do not reconstruct a vocabulary-vector objective. They can document old behavior, not serve as new vector-loss measurements.
- The saved local linear-response extrapolation predicts AC distortion1.338577 where actual AC is1.025982, against Uniform1.202569. This rules against trusting that particular unchecked extrapolation. It does not establish that all gradients, local models, or small-step methods fail.

## Candidate designs and the assumptions they add

| Design | What it estimates or optimizes | Material assumption / limitation |
|---|---|---|
| Current finite cost then rank mapping | One-time ordering of layer costs | Loses cost magnitude; final combination not optimized; remains useful as the existing control |
| Independent cost curves plus resource-allocation DP | Exact minimum of an additive surrogate on its allowed grid | True model distortion need not be additive; DP exactness applies to surrogate and constraint, not functional optimum |
| Lua-style soft threshold descent | The objective of a continuously masked network | Soft attenuation can improve while hardened deletion does not; backward activation memory and temperature choice matter |
| Hard-forward surrogate-gradient proposals | True hard model in forward, approximate derivative of its discrete selection in backward | STE is biased; budget projection and rounding can reverse a predicted improvement |
| Feasible hard-mask direct search | Measured objective at actual count-preserving candidates | Expensive evaluations and limited-neighborhood local traps; candidate schedule matters |

SPDY is an explicit primary precedent for DP plus globally informed local search: its layer-error additivity is an assumption, not a consequence of the solver. Lua-LLM provides row thresholds with Wanda ordering and a soft selection model. BESA provides block-reconstruction-driven allocation. EvoPress already provides feasible compression exchanges and multistage fitness evaluation; making a deterministic exchange variant is not by itself a new pruning contribution.

Sources:
- Lua-LLM: https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf
- BESA: https://arxiv.org/abs/2402.16880
- SPDY: https://proceedings.mlr.press/v162/frantar22a/frantar22a.pdf
- EvoPress: https://arxiv.org/html/2410.14649

## A concrete hard-mask reference design

This is the implementation reference to test first, not a claim that it wins on quality or time.

1. Freeze original weights, within-row order, exact integer budget, pair bank and starting native Uniform50 mask. A matched warm-start experiment using old AC is separate.
2. Represent a mask by integer row-prefix removal counts and record the actual mask hash. A layer-level control may tie count profiles across the seven projections using identical templates across layers. Projection-level allocation uses independent projection counts. Do not silently let a global rounding repair alter unrelated layers.
3. Propose bounded feasible trades: restore q actual parameters in one unit and remove q in another. With row-uniform counts in projections a,b, `out_a * delta_k_a = out_b * delta_k_b = q`; choose q from attainable count quanta and check bounds. Equal percentage changes are not generally feasible trades. For tied layer templates, verify the template parameter-count increments match.
4. Use a declared rotating candidate schedule, or a cheap predictor only to prioritize candidates. Do not require a predictor-positive score as a correctness condition or filter every unexplored direction forever. An exhaustive directed neighborhood contains G(G-1) pairs per step size, so cap and record how much of it was searched.
5. Evaluate actual candidates with the unchanged endpoint+response objective. Cheap stages can use a shared document-stratified subset; finalist and incumbent must be compared on the same full optimization bank. Both endpoints and the common query set remain paired at every stage. Do not use independent per-endpoint minibatches that destroy C.
6. Accept a measured improvement exceeding a fixed numerical tolerance; otherwise retain the incumbent. Reduce the declared exchange size after unsuccessful coverage, and stop on evaluation budget or minimum step. A capped candidate schedule establishes only 'no improvement among evaluated candidates,' not local or global optimality. Ordinary confidence intervals after adaptive candidate selection are not confirmatory tests.
7. Rebuild candidate priorities and invalidate affected caches when a mask is accepted. Stop with a frozen hard mask, then measure untouched-document objective, NELBO, and separately downstream quality. The optimization bank is calibration, even when repeatedly used as a finalist acceptance bank.

Single trades can miss jointly useful moves. A bounded bundle/beam is a later search-capacity ablation; adding it now is not justified by the old cross-term audit, which did not show interaction dominance. Repeated measured exchanges are not equivalent to the earlier unsuccessful 'refresh local reconstruction once' method, but they still have to demonstrate benefit.

## How a gradient alternative should compete

A hard-forward threshold/count proposal is the principal challenger, rather than treating a soft loss curve as the result. Use the same feasible candidate family and initialize at the same sparse mask. A common-budget projection plus a declared tie rule produces actual integer counts; the hard candidate is then evaluated with the same acceptance rule. Preserve gradients only where the surrogate needs them; frozen original weights do not eliminate activations needed for backward. A pure teacher reconstruction objective has zero error at the dense teacher itself, so dense initialization supplies no initial reconstruction gradient; it is not an equivalent starting point to Uniform50.

A restoration proposal must use the original nonzero weight value and a derivative through the effective weight or boundary gate. Multiplying a gradient by the already-zeroed deployed weight incorrectly erases restoration utility. Boundary gates can expose this derivative without updating the original weights.

No claim is made that the challenger is faster: count forward work, backward work, hard validation, rank/mask construction, and memory. If gradient proposals consistently beat uniform/scheduled feasible proposals per measured wall time under the same objective, promote the helper. Otherwise a simpler discrete optimizer can be the method backend. Whole-model finite probes are also expensive and are not required for every unit at every iteration.

## Exact implementation opportunity: compute the same logit objective in hidden coordinates

The repository model applies final normalization before a linear vocabulary head (`model/modeling_llada.py`, lines1621–1633). Let h be this post-normalization state, U the frozen head, alpha its optional logit scale, V vocabulary size, and P=I-11^T/V. The common head bias cancels between dense and sparse. Then for delta_h=h_sparse-h_dense:

    ||P (z_sparse-z_dense)||^2 = delta_h^T Q delta_h
    Q = alpha^2 U^T P U

If the objective averages vocabulary coordinates, Q includes an additional1/V. Apply the same Q to endpoint errors and to the difference of endpoint hidden errors. This is an exact algebraic re-expression in real arithmetic, not plain hidden MSE, not a local network linearization, and not the failed allocation Jacobian approximation. The transformer and final norm remain nonlinear and are evaluated for each candidate.

Conditions: both models share the same head, vocabulary, scale and affine output path; obtain h after final norm; preserve declared token/pair/vocabulary averaging. A nonlinear logit transformation, changed head, KL, or gold log-odds would not admit this same constant Q identity. Floating-point equality still needs numerical checking before use.

Q needs d^2 storage, while its one-time formation costs O(V*d^2); therefore do not call it free or automatically faster. A chunked head calculation avoids materializing P. For short studies, chunked direct logit scoring may be cheaper than building Q. A random sketch or low-rank truncation is an approximation and a separate implementation choice.

## Exact prefix reuse without causal attention assumptions

For a hard candidate whose first modified block is b, the incumbent's full-sequence hidden state entering b is unchanged. It may be cached and the complete suffix recomputed. This is valid for bidirectional attention because every token within the changed suffix is still recomputed. It is not permission to freeze visible-token states or to reuse changed-layer KV tensors. Keys need model/revision, mask-prefix identity, pair endpoint/input, position and attention configuration. Accepting an earlier-layer edit invalidates later prefix caches. Late edits may benefit most; cache rebuilding and storage count toward the search budget.

## Comparison that resolves the design question

Use one fixed objective, initial hard mask, rate range, candidate domain, documents and total measured budget:

1. Current rank allocator, to establish the existing behavior.
2. Feasible hard-mask direct search.
3. Hard-forward gradient-proposed feasible search; soft Lua is a separate relaxation control if affordable.

Measure best **hard-mask** optimization loss versus elapsed compute, unseen-document loss, NELBO, peak memory, forward/backward count, number of accepted exchanges, and rejected predicted improvements. A lower calibration objective is not the downstream success criterion. Optimize on disjoint articles from final quality evaluation; reused mini100 is development.

Start the controlled backend comparison at the existing layer granularity. Comparing32-layer discrete search against row-wise Lua changes both optimizer and hypothesis class. A224-projection extension is useful but is a separate capacity/compute comparison. No final step size, coefficient, subset size, or run budget is authorized or silently selected by this design document.

## Code-audit cost findings

The model readout receipt has U shape126464x4096, untied head, no head bias and scale_logits=false. Q therefore occupies64MiB inFP32, with a one-time construction estimate of2.122e12 multiply-accumulates. The present output_hidden_states path still runs the vocabulary head; a hidden-only path is required to save its compute. Merely adding a capture hook saves retained storage but not head multiplication.

Historical scalar collection has64 probe candidates x80 pairs x2 endpoints =10,240 full sparse forward calls. The corresponding first-to-last probe event window is281.64s; the Uniform score's event window is3.99s. These are historical event windows, omit preparation/first-event work, and are not promises for vector scoring or future hardware. A fresh full-bank hard candidate at80 pairs requires160 state evaluations before valid batching/prefix reuse. Exhaustive32-unit directed pairs require992 candidates per exchange size (158,720 state evaluations);224 units require49,952 candidates (7,992,320 evaluations). This is why bounded candidate scheduling and measured cost comparisons are necessary.

## Decision

Do not equate a plausible objective with a justified soft-threshold optimizer. Keep a true-hard-mask evaluator and exact integer budget as common infrastructure. Compare a bounded discrete method with a gradient-proposal method; use exact readout algebra and valid prefix reuse to reduce evaluation costs before increasing algorithm complexity. Existing optimizer ideas are reused openly. Any scientific contribution must come from demonstrated incremental value of DLM conditional-response preservation, quality at a lower calibration cost, or a clearly tested design improvement, not from renaming the search.

## Agent reports and parent review

- [Continuous relaxation review](design_relaxation_2026-09-18.md)
- [Discrete design review](design_discrete_2026-09-18.md)
- [Code and cost audit](design_code_audit_2026-09-18.md)

The parent corrected several draft recommendations rather than treating agent agreement as evidence: primary224 vs existing32 was separated into a granularity comparison; weight-per-row prefix counts must not be confused with deleting rows; the sum of sequential accepted loss differences telescopes and cannot diagnose non-additivity (use independent edits from a common baseline versus their joint edit); and old local-surrogate failures do not establish universal gradient failure or justify a beam/bundle by themselves. Proposed numerical search caps in subordinate reports are illustrative, not an approved experiment setup.
