# Axis 4 — Existing experiment-results audit: functional background and computation-unit gaps

Date: 2026-09-18  
Status: read-only audit and design recommendation; no model loading, forward pass, mask generation, or Obsidian write  
Scope: a static unstructured weight allocation for the current LLaDA sparse-prefix pipeline

## Finding

The most concrete gap is between a useful functional signal and the background in which it is measured. The historical capacity allocator is the strongest evidence that a DLM functional signal can allocate projection rates: its held-out mean KL was `0.452808` versus `0.562117` for Uniform, and its full GSM8K score was `250/1319` versus `139/1319`. However, it measured one projection at a time in an otherwise dense model with the historical one-shot Wanda mask family. The current native sparse-prefix pipeline changes `217/224` projection masks relative to the historical family (only block 0's seven masks match), and the role audit gives a direct warning that local effects can change sign in a sparse background.

The next allocation candidate should therefore measure a finite, full-vocabulary functional exchange in the *current jointly sparse background*, at the rates that the final allocator will actually use. This is distinct from adding another state-coverage or context-pair scalar. A second, related gap is the unit of allocation: several results suggest that independent Linear errors are the wrong abstraction for some operations, but this has not been tested against a per-Linear control under one current mask pipeline.

## Evidence from completed experiments

### 1. The historical capacity result is real, but its transfer assumptions are untested

`experiments/projection_capacity_allocation_65/report.md` reports direct masked-token KL curves for 224 projections at 50–75% sparsity. The exact-budget capacity allocation beats its historical Uniform control on held-out DLM KL (`0.4553846` versus `0.561242` in the full-model receipt) and on the full task (`250` versus `139`). Reconstruction and EIS+type controls are weaker on KL (`0.631180` and `0.498892` in the audited receipt), so this is evidence for measuring functional damage rather than treating a reconstruction proxy as sufficient.

There are two transfer boundaries. The capacity curves prune one projection in a dense background, and the saved masks use the historical Standard Wanda ranking. The corrected native sparse-prefix runner remeasures each later block after earlier blocks are sparse. The historical Uniform mini is `62/100`; the current native Uniform is `54/100`, with A-only `55/100` and A+C `61/100`. These are different mask/ranking families, so the old capacity gain cannot be applied to the current Uniform or A+C masks without a new control.

The role audit gives the second boundary. In the dense background, some local bundle changes improve KL, while in the Role-sparse background a corresponding bundle can reverse direction (the documented B2 reversal). In the corrected role-exchange analysis, among 3026 bundle-state cases where both local reconstruction costs improved, functional KL worsened in 1226 cases (40.52%). Thus a dense-background additive curve, or a local reconstruction score, is not a safe allocation objective for the current sparse model.

### 2. The 48/52 probe-to-45/55 rate map discards the evidence needed for allocation

The context-response configuration evaluates one block at 48% and 52% in a frozen Uniform50 background, then maps the signed per-weight cost by average rank to rates spanning roughly 45–55%. Rank mapping removes the cost magnitude. It also assumes that the local slope at 48/52 remains informative after moving to the final rates and after combining 224 changes. The saved linear-response audit does not support that extrapolation: for the AC background, the linear predictions were `1.349005` (A) and `1.338577` (AC) versus observed `1.041629` and `1.025982`; the recomputed Uniform baseline was `1.202569`. The diagonal quadratic term being larger than the cross term does not rescue the predictor; it shows that the finite per-unit response itself was not transported correctly.

The A+C mini result (`61` versus `55` for A and `54` for native Uniform) is a development screen with an inconclusive A+C versus A comparison (`p=0.145996`). Its scalar gold-vs-rest log-odds also omits wrong-vs-wrong ordering. It is not evidence that rank-only mapping is a valid static allocator.

### 3. Functional damage depends on computation location and direction, not only magnitude

The Wanda failure map contains a useful constraint on any cheap replacement. At 50%, `block31.ff_out` has local reconstruction error `0.006054` but the largest module KL `0.022711`; `block30.ff_out` has reconstruction error `0.042921` but KL `0.008965`. In the transplant audit, matched-norm perturbations show a location×direction interaction of `0.013907`, with sequence-cluster bootstrap interval `[0.008673, 0.020205]`; a foreign D31 perturbation at the same timestep retains only about `38.3%` of native KL despite comparable downstream perturbation magnitude. The token×feature factorization likewise reports a positive interaction (`0.004970`, interval `[0.003279, 0.006760]`).

These results do not identify a pre-pruning score. They do rule out treating local error magnitude, a single global direction cosine, or a small-response quadratic as a complete functional allocation signal. A full-vocabulary dense-versus-sparse output comparison in the current sparse background at least measures the relevant direction/readout effect directly.

### 4. Granularity itself is a confound

The completed 50% WikiText comparison gives row-quota Uniform NELBO `2.550805`, while layer-global Uniform, with the same 50% total per layer but no per-projection/row quota, is much worse at `2.663777`. In the same current pipeline, DSA layer is `2.530142` and DSA projection is `2.544038`. This does not prove that block coupling is universally better: DSA search spaces and mappings differ, and DSA's bounded masked-CE adaptation is not a new method. It does show that an allocation result can be dominated by the granularity and path constraints rather than by the named scalar.

## Candidate A — Sparse-background functional exchange allocation (SB-FEA)

### Hypothesis

At equal global sparsity, a projection's useful rate is determined by the finite change in the *whole sparse DLM's output distribution* when that projection's rate is changed, not by a dense-background module curve or by a rank-normalized 48/52 scalar. The candidate is a criterion and measurement change; it does not introduce a new optimizer.

### Frozen construction

1. Reproduce and freeze the current native sequential Uniform50 masks, row-wise Wanda order, model revision, state bank, exact global count, and block order. Do not remeasure activations separately for each candidate.
2. For a prespecified exchange `(receiver a, donor b, q)`, restore exactly `q` Wanda-ranked row-quanta in `a` and remove exactly `q` in `b`. Candidate masks must be materialized in the already sparse background, so every exchange has the same global count.
3. Evaluate dense-versus-candidate full-vocabulary masked-token KL on a held-out DLM state bank. Use the same-path dense sham and preserve signed changes. The primary quantity is the finite exchange KL, not a local Linear reconstruction error, gold-vs-rest log-odds, or task cross-entropy.
4. Keep the first search to one prespecified exchange round. If a rate curve is needed, measure the actual final grid (for example 45/50/55%) in the same sparse background rather than ranking 48/52 magnitudes and extrapolating. Any multi-exchange assembly must be evaluated as a complete sparse model; a sum of edges is only a proposal score.

The output is a static exact-budget mask. It uses the existing Wanda support family and therefore isolates the background-aware functional criterion from within-row support construction. It is a practical extension of the old capacity idea, not a claim that direct KL allocation is novel in general.

### Minimum useful experiment and controls

The smallest informative screen is a prespecified set of 6–12 exchanges spanning early/late depth, terminal MLP, early `v_proj`, attention, and ordinary MLP projections. Use the same state bank and exact q for every arm. Compare:

- dense-background capacity cost (historical curve, where available);
- current sparse-background finite full-vocabulary KL cost;
- the existing 48/52 rank-mapped cost;
- cardinality-matched random exchanges and a depth/type-matched exchange control.

The screen asks whether sparse-background functional cost predicts the held-out full-model KL change better than the rank-only and dense-background controls. If it does, assemble one exact-budget candidate from a prespecified matching/exchange rule and compare it with native Uniform on held-out WikiText NELBO using shared MC draws. Do not select by GSM8K mini. If the direct cost does not add predictive value, stop the criterion; do not repair it with a new rank transform.

The saved J audit makes a quadratic surrogate a negative control, not an additional feature: the cross-term (`~0.023`) is not evidence of interaction dominance, and the observed finite outcomes already contradict its extrapolation. Direct finite evaluation is the relevant control.

## Candidate B — Operation-group functional allocation as a structural ablation

The D31 results and the row/projection quota results suggest that the independent Linear may be too small an evaluation unit. A bounded structural candidate is to retain Wanda's within-Linear support but allocate rates to a small set of architectural computation groups, then use the same sparse-background functional output objective:

- attention relation group: Q/K (with V and output as a separate branch or a pre-registered combined attention group);
- gated MLP group: the two input branches together, with the output projection measured at the residual branch;
- residual branch output: the block-level output after the operation that actually enters the residual stream.

The group signal must be the change in the corresponding dense operation, including its composition (for example attention relation or gated product), rather than the sum of member Linear errors. The exact group partition should be fixed from the model graph before observing outcomes. No hand-coded protection of D31 or reversal of depth is justified by the current data.

The minimum control is a matched comparison under the current sparse-prefix mask family:

1. 224-independent projection-rate allocation;
2. block-coupled group-rate allocation with the same global budget;
3. the same group rates selected from the sum of member local costs;
4. a row/projection-quota control.

Evaluate the operation-group output error and the full-vocabulary held-out KL separately. A group metric earns consideration only if it predicts whole-model KL beyond the sum-of-members control and yields a held-out NELBO improvement. A group metric that merely changes granularity, or a favorable local response without NELBO gain, is a diagnostic result. DSA layer/projection and EvoPress remain strong existing baselines; their current scores do not establish this group hypothesis or make its search procedure novel.

## Decision and claim boundary

Prioritize SB-FEA as the minimum useful experiment because it repairs the clearest experimental mismatch: functional capacity worked historically, while its dense-background/single-module assumptions have never been checked under the native sparse-prefix mask family. Keep Candidate B as a structural ablation unless the exchange screen shows that per-Linear direct costs are unstable or systematically misordered.

The defensible initial claim, if the proposed control passes, is narrow:

> Under a fixed native sparse-prefix Wanda support and exact static budget, finite full-vocabulary functional exchanges measured in the jointly sparse background provide more useful projection-rate information than dense-background or rank-normalized local costs.

This does not establish DLM-versus-AR specificity, task-CЕ superiority, causal preservation of generation, or an optimal allocator. Historical Uniform62 uses a different ranking/mask-construction family; native Uniform54, A55, and AC61 share the frozen native Wanda ordering with different allocations. No conclusion here promotes coverage, role minimax, DLM-SUM, timestep weighting, a super-outlier rule, or the failed quadratic extrapolation.

## Audited sources

- `experiments/projection_capacity_allocation_65/report.md` and `experiments/dlm_capacity_predictor/existing_results_report.md`
- `experiments/dlm_ppl50_mechanism_review/report.md` and `experiments/dlm_ppl50/README.md`
- `experiments/dlm_context_response50/{config.json,report.md}`
- `research/four_axis_saved_probe_audit_2026-09-18.json`
- `experiments/dlm_role_decision_audit/report.md` and `experiments/dlm_role_exchange_prediction_v2/report.md`
- `experiments/wanda_failure_characterization/{report.md,propagation_report.md,transplant_report.md,token_feature_factorization_report.md}`
- `experiments/dlm_distribution_hypotheses/report.md` and `experiments/dlm_scale_shape50/report.md`


## Parent synthesis clarification
Direct sparse-background KL exchanges are a strong functional baseline/backend, not a DLM-specific contribution by themselves. Selection data are development/search data; reserve separate validation data. Historical65% capacity results and50% Uniform62/54 provenance are distinct settings.
