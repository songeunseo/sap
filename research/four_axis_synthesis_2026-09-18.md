# Four-axis DLM pruning method exploration

Date: 2026-09-18
Status: research hypotheses and existing-artifact analysis; no GPU, model forward, new pruning mask, training, or benchmark execution.

## Decision

The leading research direction remains conditional-response preservation, but the existing gold-logodds A+C criterion and its rank-based allocator should not be treated as an indivisible method. A concrete extension is to preserve the vector-valued response to controlled changes of visible context, then optimize a static, exact-budget allocation without discarding measured cost magnitude. This is a continuation of prior proposals, not a newly discovered concept or a demonstrated improvement.

Direct task-CE search is a useful baseline, but an earlier explicit user preference was not to put task loss directly into the main proxy. Later approval of A+C means functional reconstruction objectives are within the explored scope; it does not automatically approve a new task-loss-driven method. No implementation or experiment is launched by this review.

## Four evidence streams and historical provenance

1. Conversation review: parent retrieved actual messages from `Proxy 검증 실험 설계`, `role-wanda 결과 표로 정리`, `DLM proxy와 search 방향 결정`, and `DLM 맞춤 allocation 기준 설계`. Compact source receipts are `four_axis_history_extract_2026-09-18.json` and `four_axis_history_additional_2026-09-18.json`. Agent1's first draft relied on notes because its catalog did not expose app-history tools; parent supplied the actual messages for revision. These are selected relevant conversations, not every historical turn.
2. Methodology: `four_axis_methods_2026-09-18.md`. DSA/EvoPress/BESA/Lua/LSA/2ndMatch and DLM calibration/distillation priors constrain novelty; new solver invention is unnecessary.
3. DLM analysis: `four_axis_dlm_analysis_2026-09-18.md`. Dynamic activation supports, mask-count effects, temporal sensitivity, bidirectional context and sinks are observations; none alone establishes weight-pruning sensitivity.
4. Experiments: original A+C/coverage results, role-exchange V2, context-refresh and existing baseline audit, plus independent `four_axis_experiments_2026-09-18.md` review. Axis4 favors direct whole-sparse-model functional exchanges; parent treats ordinary KL exchange as a strong baseline/backend rather than the DLM-specific main contribution. Parent also computed a small CPU-only audit from saved per-layer response vectors.

Important historical corrections:
- Context-response preservation and joint Q/K or gated-MLP function preservation were already proposed in `DLM proxy와 search 방향 결정`; do not present them as new discoveries.
- User relaxed mandatory masked/unmasked role separation. Do not restore it as a requirement.
- DSA originally uses performance-guided evolutionary expression search. Local masked-CE search is a DLM adaptation of that mechanism, not an invented search stage; custom controller/budget differ from full paper reproduction.
- The old claim that all allocation methods lose to Uniform is not supported across current protocols.

## Existing observations relevant to method design

Matched native 50% mini: Uniform54, A55, A+C61. Historical cached Uniform62 remains a descriptive end-to-end reference; masks/ranking differ. A+C vs A is not conclusive at mini100, no A/AC NELBO measurement exists, and full jobs were stopped.

Current calibration distortion:

| Mask | A | C | A+C |
|---|---:|---:|---:|
| Uniform | .824001 | .378568 | 1.202569 |
| A allocation | .665290 | .376338 | 1.041629 |
| A+C allocation | .658982 | .367000 | 1.025982 |

These show the assembled A+C mask reduces its calibration surrogate; they do not show that conditional-response preservation caused the mini accuracy gain.

Current implementation probes one layer at48/52% in Uniform50, divides the difference in distortion by actual extra removed parameters, then maps ONLY the 32 ranks into45–55%. All candidates share one frozen Wanda ordering. Consequently magnitude is discarded, budget changes extend beyond the measured interval, and the final combination is evaluated after allocation rather than optimized directly. These are design choices, not implementation errors.

Negative evidence remains active:
- C1 coverage54, pooled50, Uniform54 on mini; small16-article NELBO gain vs pooled, Uniform comparison inconclusive. A renamed quantile/worst-state allocator is not supported as a new main method.
- Role-context refresh19/dense-target12 vs Role24 did not establish refresh as a remedy.
- In corrected role-exchange V2, both local reconstruction costs improved but KL worsened in1226/3026 bundle-state cases (40.52%). This is not a universal probability, but a concrete counterexample to assuming additive local losses imply functional improvement.
- Pairwise interaction prediction and blind timestep weighting have not earned positive support merely by sounding DLM-specific.

## New CPU-only check: do saved response vectors justify joint quadratic allocation?

Input: Uniform errors and32 pairs of48/52% errors,80 existing context pairs, no new model evaluations. Each per-state vector was embedded so squared norm exactly reproduces the original A or A+C averaging. J is a central finite difference per actually removed weight; delta uses final manifest counts. Prediction: r(s)=r0+J delta. See `four_axis_saved_probe_audit_2026-09-18.py` and `.json`.

| A+C objective | Uniform | A allocation | A+C allocation |
|---|---:|---:|---:|
| Saved actual | 1.202569 | 1.041629 | 1.025982 |
| Linear-response prediction | 1.202569 | 1.349005 | 1.338577 |

The naive extrapolation predicts degradation against Uniform where measured distortion improves. This does NOT prove that all local approximations fail: the probes are about±2percentage points whereas final allocations reach±5points, and hard-mask effects need not be smooth. It specifically prevents promoting this unvalidated J-transpose-J allocator from its algebra alone.

For A+C, off-diagonal response cosines average .000164, range[-.12849,.11014]. At the existing AC allocation, cross-layer quadratic contribution .023169 is smaller than diagonal .335134. These statistics do not establish dominant cross-layer interactions; neither do they measure all nonlinear interactions.

## Literature observations that change priorities

- [Masks Can Be Distracting](https://arxiv.org/html/2511.21338) finds mask-count effects and context locality. This motivates controlling count when identifying content response; it does not show that pruning amplifies these effects. Its reported quantization comparisons also caution against assuming every compression intervention worsens them.
- [Subliminal Clocks](https://arxiv.org/html/2607.01774v2) supports progress-associated internal directions, not a pruning importance score. Keep progress-related endpoint behavior rather than deleting its subspace by default.
- [Measuring Temporal Linguistic Emergence](https://arxiv.org/html/2604.23235) finds mid-trajectory re-masking sensitivity mostly local to perturbed positions in one LLaDA/WikiText setting. This is counterevidence to a universal long-range error-amplification story, not a refutation of weight-pruning effects.
- [Attention Sinks in DLMs](https://arxiv.org/html/2510.15731) finds limited damage from masking a single sink in tested DLMs. High attention mass need not be irreplaceable static weight capacity.
- [Generation Order and Parallel Decoding](https://arxiv.org/html/2602.00286) separates order-sensitive model error from factorized parallel-sampling error. Our256steps/256slots protocol is roughly one commitment per step; parallel error cannot simply explain its current results.
- [BESA](https://arxiv.org/abs/2402.16880) and [Lua-LLM](https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf) make allocation/threshold optimization existing machinery. [2ndMatch](https://arxiv.org/html/2506.05398) already preserves sensitivity in pruned image diffusion models; generic response matching is not new.
- [FAIR-Calib](https://arxiv.org/html/2606.06547) already handles frontier instability and amplification in DLM quantization. Commitment-cost rollout remains an expensive alternative, not the default most-novel route.

## Leading candidate: vector conditional-response preservation with direct allocation

### Hypothesis and measurement

A fixed sparse DLM should preserve how its token hypotheses change when available context changes. Current gold-vs-rest A+C collapses all incorrect alternatives; a vector response can retain those distinctions without using a gold-token CE as the allocation score.

Let F denote one fixed functional readout. A concrete main version uses vocabulary-centered logits on common masked queries (subtract each query's vocabulary mean); endpoint and transition distortions then do not depend on an arbitrary common logit offset. A smaller final-normalized-hidden-state readout is a separate cost/representation choice, not silently the same method. The surviving original weights stay frozen.

For dense D, sparse S and paired contexts x0,x1:

    Eend = 0.5 * (||FS(x0)-FD(x0)||^2 + ||FS(x1)-FD(x1)||^2)
    Eresponse = ||[FS(x1)-FS(x0)]-[FD(x1)-FD(x0)]||^2
    L = mean_pairs(Eend + lambda * Eresponse)

Use a fixed shared scaling and declared lambda; normalizing each tiny response by its own energy can amplify noise and is not automatically principled. This is an endpoint metric with a cross-state weighting structure; response information is not magically absent from sufficiently accurate endpoints.

Construct some pairs from the same natural clean span and equal-size alternative visible-token sets, keeping query positions masked and total mask count/sequence length equal. These compare WHICH evidence is available, not only how much has been revealed. Mask locations can still differ: equal count does not fully isolate pure semantic content or remove all progress representation effects. Preserve original/nested-reveal endpoints too; controlling progress in one diagnostic is not a proposal to discard progress behavior. A fixed-mask-pattern content substitution is a separate, potentially off-distribution control.

### Allocation algorithm options

Keep the current within-row Wanda order and original surviving weights. Learn or search only how many weights remain, using one static mask for every denoising state. Parameter-weighted budgets matter when projections have different sizes.

1. **Research version without evolution:** differentiable threshold/rate optimization using an existing BESA/Lua-style relaxation. Optimize L with the global count constraint, harden the mask with exact count correction, then evaluate the hard model. The backward graph and soft-to-hard gap are real costs; freezing weights does not eliminate backward memory. This is a concrete option, not already implemented or proven faster.
2. **Lower implementation-risk version:** use measured marginal costs to propose a bounded set of exact-budget exchanges, but accept/rank them using L measured on the whole hard sparse model. This is direct search, with no novel optimizer claim. Avoid extrapolating unchecked local curves or the failed response-J model.

These backends are alternatives. Do not add both, state-quantile weighting, clock projection, SAE features, and rollout Q into one method.

### What would be new versus existing A+C?

- Vector-valued conditional behavior rather than one gold-vs-rest scalar.
- Controlled context-pair construction separates at least mask-count changes from evidence choice.
- Allocation retains objective magnitude and evaluates/optimizes the assembled mask rather than mapping a one-time rank into a fixed interval.

Changing all three at once would prevent attribution. First compare objective variants with ONE backend/readout/state bank. Separately compare rank allocation and direct allocation with ONE frozen objective. New solver mathematics and complete causal circuit identification are not required.

### Minimal distinguishing screen, not an execution plan

With the same candidate family, budget, allowed sparsity range, pair bank and compute accounting:
- endpoint-only vector reconstruction;
- endpoint plus correctly paired vector response;
- same endpoint marginals with broken pair correspondence.

Then isolate the backend change on the same objective. Keep nativeUniform, historicalcachedUniform and legitimateDSA/EvoPressadaptations with their actual mask engines identified. NELBO remains the agreed primary quality metric; GSM8K adds a task view. Current reused mini100 is development, not confirmation. A cheaper method matching a stronger baseline can still have value; a utility-prediction diagnostic is useful but not a mandatory gate for all methods.

## Alternatives and discarded overclaims

1. Complete-operation preservation (Q/K attention, gated MLP, block residual updates) is a valid older candidate. Paired nonlinear group outputs differ from per-Linear local MSE, but AR relevance and BESA-style reconstruction overlap are strong. Do not claim the D31 experiment already proves Q/K importance.
2. Two-token order compatibility is a higher-risk new diagnostic: compare log p(i|c)+log p(j|c,i) with the reverse order, and compare sparse-minus-dense defects. Dense defects need not be zero; reducing a defect can improve consistency while worsening correctness. No pruning evidence presently promotes this to the main method.
3. Worst-state/quantile coverage, simple log-variance, clock decodability, Bayesian replacement alone and a large commitment-rollout stack are lower priority. None becomes novel or useful solely by being more complicated.

## Synthesis limits

The agents' suggestions are research judgments, not votes establishing correctness. Parent rejects promotion of the weak coverage variant and unchecked local quadratic allocation. Static-mask optimization, response matching, joint reconstruction and direct search all have substantial prior art. The defensible research question is whether controlled DLM conditional responses add quality, efficiency or robustness to a fixed compression budget; its novelty and benefit remain unvalidated.

## Audit verification
All64 saved probe tensor SHA256 values match existing receipts. Recomputed A/C/AC losses match the summaries with maximum absolute error2.22e-16; see four_axis_saved_probe_verification_2026-09-18.json. No new model evaluation.
