# Axis 2 — From allocation literature to concrete DLM static-pruning candidates

Date: 2026-09-18  
Status: methodology review; no new experiment, forward pass, mask, or Obsidian write  
Scope: post-training static weight pruning/allocation for a masked DLM, with a fixed global weight budget

## Decision

The best first candidate is **matched context-transition function geometry with fixed Wanda support**. It uses no CE, KL, or likelihood loss as the allocation proxy. It measures how intermediate denoising computations change when visible context changes, while holding the within-row ranking, surviving weights, model, sparsity, and allocator family fixed. Its DLM-specific claim is narrow: one static sparse support should preserve the functions used to pass newly visible information to still-masked positions.

A higher-cost validation candidate is **native paired-response exchange search**. It evaluates exact-budget mask exchanges with the whole sparse DLM on the paired state bank, rather than trusting a local reconstruction curve or a quadratic surrogate. It should be used only if the low-cost criterion appears to add information, because its compute and prior overlap are substantial.

The saved-probe audit argues against promoting a Gauss–Newton-style (J^T KJ) allocator. A naive linear extrapolation from the 48/52%-probe masks predicted A/AC objective values worse than Uniform, while the observed A/AC values were better. The sparsity steps represented by the real allocations were as large as about plus or minus 5 percentage points, whereas the probes covered about plus or minus 2 points; the observed-versus-predicted discrepancy must not be described as a 5-point loss error. The probe report gives AC objective Uniform 1.202569, predicted A 1.349005, predicted AC 1.338577, observed A 1.041629, and observed AC 1.025982. Cross-vector cosine was near zero on average (mean 0.000164, range [-0.12849, 0.11014]); the cross-quadratic contribution (about 0.023) was smaller than the diagonal contribution (about 0.335). This is not evidence that cross-unit interactions are the dominant allocation signal. The existing role audit also found that local reconstruction improved while functional KL worsened for 40.52% of bundles. Any local or intermediate-function criterion must therefore be treated as a ranking hypothesis and checked against whole-model NELBO/KL.

## What the primary literature actually makes available

The literature does not leave a missing generic optimizer to invent. It leaves a choice of what the candidate vector or threshold is supposed to preserve.

| Source | Optimized object and signal | Cost or boundary relevant here |
|---|---|---|
| [DSA, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf) | A computation graph maps element scores through preprocess/reduce/transform/postprocess operations to layer sparsity; an evolutionary search scores candidates on validation performance under a size constraint. | The contribution is allocation-function discovery, not a new scalar such as variance. The paper reports a large operation space and roughly half a day on one H800 for search, so its search can be a control but is a poor novelty claim.
| [EvoPress, ICML 2025](https://arxiv.org/html/2410.14649) | A discrete compression-level vector is searched under an exact resource constraint. Level-switch mutation changes two units in opposite directions, preserving budget; elitist multi-stage evaluation uses cheap partial samples before larger evaluations. | Direct functional fitness and exact-budget mutation are already prior. Reusing this machinery is sensible; the DLM-specific objective must carry the contribution. |
| [BESA](https://arxiv.org/html/2402.16880) | Sequential block pruning minimizes dense-versus-sparse block reconstruction plus an (L_2) sparsity penalty. Layer/block sparsity coefficients are learned with STE while the original weights remain fixed. | Differentiable allocation and reconstruction are prior. BESA's blockwise input propagation also warns that a local curve is not automatically a whole-model objective. |
| [Lua-LLM, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf) | Every row gets a learnable threshold; a sigmoid soft Top-K mask is optimized end-to-end with task loss plus a target-sparsity regularizer, yielding layer and intra-layer allocation. | It is a strong global/row allocation control. A new method should not claim differentiable thresholds, global allocation, or end-to-end mask learning as novelty. |
| [LSA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7b805585c7e249c1f65737506d4fe1e4-Abstract-Conference.html) | Minimal linear reconstruction error under hypothetical pruning is used for layer/projection budget allocation. | Covariance-aware reconstruction and finer allocation are close prior. A paired conditional term must show incremental information beyond pooled reconstruction, not merely repeat it. |
| [2ndMatch](https://arxiv.org/html/2506.05398) | A pruned diffusion model is trained to match directional (J^\top J) sensitivity of the dense model using random-vector JVPs. | “Preserve denoising response” and second-order sensitivity are already prior. A discrete, fixed-mask context-pair allocator is a narrower distinction; it is not a first response-preservation method. |
| [FAIR-Calib](https://arxiv.org/html/2606.06547v2) and [OPTD](https://arxiv.org/html/2608.02942) | FAIR-Calib weights hidden-state calibration by frontier instability and reliability; OPTD evaluates future commitment outcomes from student-visited states. | Frontier weighting and future consequences already exist in DLMs. Static weight allocation needs evidence of an additional mask-selection value, not just a port of the terminology. |
| [Automatic Pruning for Quantized Neural Networks](https://arxiv.org/abs/2002.00523) | Bayesian optimization selects layer pruning ratios for quantized CNNs. | This establishes that Bayesian ratio search is an old generic option. With 224 DLM units and expensive whole-model fitness, it is not an attractive first algorithm or a novelty claim. |

The practical implication is to fix the solver and ranking family first. A new DLM-conditioned cost can be a contribution when it changes the selected rate vector under the same budget and adds validation quality or robustness at a measured cost. “Replace evolution with Bayesian optimization” does not meet that bar.

## Candidate 1 — Matched Context-Transition Geometry (MCTG)

### Hypothesis

The repeated reveal process of a masked DLM is a sequence of conditional function changes. A sparse model may retain the magnitude of an intermediate activation while changing how a newly visible token affects the still-masked positions. Measuring this transition directly may distinguish useful information transfer from a large but harmless endpoint perturbation.

This is a hypothesis. The prior role experiments show why it may fail: better local reconstruction did not guarantee lower functional KL, and masked/unmasked role splitting did not add reliable exchange prediction in the corrected audit. MCTG changes the observable from local error magnitude to a matched state-transition function; it does not assume that this fixes the downstream mismatch.

### Calibration states and mask scope

For each calibration query, create a pair (x-, x+) with the same response span, the same response mask positions and count, and the same query. Change visible context one-for-one so that the pair differs in context content rather than simultaneously changing context content and the number of masked response positions. Align the persistent response rows used by both states. The pair construction must be frozen before allocation.

Use the existing row-wise Wanda ranking within every projection u. For a candidate retained count k, let M(u,k) be the prefix mask induced by that ranking. Do not change surviving weight values, do not retrain, and do not learn a new support in the first version. This deliberately tests a criterion-plus-existing-allocator contribution. Support learning would mix within-row support quality with between-projection budget allocation and is a separate ablation.

### Functional response criterion

Define a small fixed set of operational readouts g. These can include the Q/K attention relation, the V-to-attention-output computation, the gated MLP output, and the block residual output. The choice is an observation interface, not a claim that Q/K or gated MLP grouping is novel; those groups were already proposed in the project history. For each pair p and readout g, collect dense and candidate intermediate values Phi(D,g,p,-), Phi(D,g,p,+), Phi(M,g,p,-), and Phi(M,g,p,+). The response vectors are

    r(D,g,p) = Phi(D,g,p,+) - Phi(D,g,p,-)
    r(M,g,p) = Phi(M,g,p,+) - Phi(M,g,p,-).

No CE, KL, token correctness, or likelihood term enters the proxy. To compare direction and response magnitude without an arbitrary loss coefficient, map each response to

    h(r) = [r / max(norm(r), eps), log(max(norm(r), eps))].

The candidate score is the equal-group, equal-pair mean

    D_TR(M) = mean(g,p) norm(h(r(M,g,p)) - h(r(D,g,p)))^2.

The epsilon floor must be fixed before looking at results and sensitivity to that floor is a diagnostic. If a readout is exactly or nearly unchanged in the dense model, its transition should be recorded as a zero-response case rather than allowed to dominate the score. The score preserves a denoising function change, rather than directly minimizing a task loss or local reconstruction error.

### Exact allocation and search

The allocation vector is the projection rate vector k, with M(u,k(u)) fixed by Wanda. Search the allowed 45–55% rate grid, or another pre-registered grid, under the exact global count. Use an existing budget-preserving search, such as EvoPress level-switch mutation or the project bounded exchange routine, and make no optimizer claim. Each candidate rate vector is evaluated by the same full model on the paired states while collecting the intermediate readouts. This avoids assuming that group costs add or that a local curve predicts joint sparse behavior.

Algorithmically:

1. Freeze model revision, calibration documents, paired-state construction, Wanda rankings, rate grid and global count.
2. Run the dense model once on both states of each pair and cache the selected intermediate readouts.
3. Generate exact-budget rate-vector candidates with the existing budget-preserving search.
4. Run each candidate on both states of each pair, compute D_TR, and retain the best candidate under the fixed evaluation cap.
5. Evaluate held-out document NELBO and the pre-registered GSM8K protocol only after the allocation is frozen. Record D_TR, full-vocabulary KL, and action/commit diagnostics separately.

### Cost and expected contribution

The dense reference costs two forwards per pair. A candidate costs two full-model forwards per pair if the states are evaluated separately, although batching can reduce wall-clock overhead. With 80 pairs, this is 160 state evaluations per candidate, plus intermediate-readout storage. The total is N_eval times that cost; it must be measured rather than assumed cheap. A local projection reconstruction curve is a lower-cost diagnostic/control, not the MCTG algorithm.

The plausible contribution is a no-loss DLM-conditioned response criterion plus an existing budget-preserving allocator, scoped to fixed Wanda support and static exact-budget projection rates. It does not establish that the criterion is universally better, that AR pruning fails, or that conditional response is causally necessary.

### Minimum distinguishing ablation

Keep Wanda support, rate grid, search candidate pool, state bank, document split and exact budget identical. Compare at minimum:

1. endpoint intermediate-state geometry, which compares Phi- and Phi+ separately;
2. true-pair MCTG, which compares the transition vector;
3. pair-shuffled MCTG, which preserves endpoint marginals but breaks one-to-one context correspondence;
4. final-logit transition geometry as a readout-only control;
5. full-vocabulary KL as a strong functional-fidelity control, not as the proposed proxy.

Uniform/native Wanda remains the mask baseline, and the historical cached Uniform result remains a separate ranking family. A true-pair gain over endpoint-only and pair-shuffled at the same search budget is the minimum evidence that the transition adds information. If pair-shuffled matches true-pair, the method is using state marginals rather than conditional response. If final-logit or full-KL selects the same rate vector, the intermediate readout has not earned a separate method claim; it may still offer a cost or interpretability advantage.

### Strongest competing explanation and failure boundary

The strongest alternative is that MCTG only measures activation geometry or response magnitude, with no useful information-transfer interpretation. Matched masks, one-for-one context changes, pair shuffling, fixed group averaging and endpoint controls address this explanation. They do not address nonlinear downstream propagation; held-out whole-model NELBO remains required.

The closest negative evidence is the role reconstruction audit: local endpoint/role scores sometimes improved while functional KL worsened, and the corrected role interaction model did not generalize. Therefore MCTG must be reported as a response-ranking heuristic until full-model validation demonstrates otherwise. The candidate should be stopped or downgraded if true-pair does not beat endpoint-only/shuffled on independent documents, or if it improves D_TR without improving held-out NELBO under the matched budget.

## Candidate 2 — Native Paired-Response Exchange Search (PAX)

### Why this is a separate candidate

MCTG is the lower-cost intermediate-readout candidate. PAX asks the stronger functional question directly: does an exact-budget change in the jointly sparse model preserve the dense DLM response at the final readout? It avoids claiming that independent unit curves add correctly and avoids the failed unchecked linear extrapolation.

PAX is high-cost and has weaker novelty. EvoPress already searches exact-budget compression vectors with direct functional fitness; BESA/Lua-LLM already learn allocation parameters; 2ndMatch already matches denoising sensitivity. PAX can only claim the narrower combination of a fixed static weight mask, DLM masked-state context pairs, and an explicitly paired final-response geometry objective if that combination provides incremental predictive or downstream value.

### Fixed support and search object

Keep the within-row Wanda support fixed initially. A candidate exchange E(a->b,q) restores q Wanda-ranked row-quanta in receiver unit a and removes exactly q Wanda-ranked row-quanta from donor unit b, preserving the global count. Restrict the first candidate pool to a frozen, count-matched set of same-type/cross-depth exchanges plus cardinality-matched random exchanges. Do not use role max/mean, a learned support, or a dense local loss as a hard gate.

The search object is the exact rate/exchange vector, not a quadratic approximation. For a candidate sparse model M, evaluate dense and candidate logits on both states of each pair. Let delta z(D,p) = z(D,p,+) - z(D,p,-) and delta z(M,p) = z(M,p,+) - z(M,p,-). Use the same response map h as MCTG and define

    D_logit_TR(M) = mean(p) norm(h(delta z(M,p)) - h(delta z(D,p)))^2.

This is a final-readout function-geometry objective and contains no CE or KL. Full-vocabulary KL is an evaluation control, not the proposed proxy. The relation term is a DLM-specific hypothesis, not a new Fisher or Gauss–Newton theorem.

### Search procedure and cost cap

Use an existing exact-budget exchange search, such as EvoPress's level-switch mutation, with no new optimizer claim. Generate a fixed candidate pool from the local MCRR curve and structural/random strata, then evaluate every shortlisted whole-model exchange on the same pair bank. A safe first scope is one frozen round with a prespecified number of exchanges and no state-bank refresh. If sequential exchanges are used, each whole bundle must be evaluated directly; do not add independent edge costs and call them a jointly sparse objective.

The cost is roughly (C_{\mathrm{cand}}\times|\mathcal P|) full-model evaluations, plus dense reference evaluations. This can be orders of magnitude above MCRR and should be reported as such. A practical candidate set may be tens of exchanges, not all 224-unit pairs. If PAX cannot beat endpoint-KL with the same candidate pool and compute, it remains a diagnostic rather than a method.

### Minimum distinguishing ablation

Hold candidate masks, exchange pool, state bank, exact budget and search steps fixed while replacing fitness with:
Use an existing exact-budget exchange search, such as EvoPress level-switch mutation, with no new optimizer claim. Generate a fixed candidate pool from MCTG response geometry and structural/random strata, then evaluate every shortlisted whole-model exchange on the same pair bank. A safe first scope is one frozen round with a prespecified number of exchanges and no state-bank refresh. If sequential exchanges are used, each whole bundle must be evaluated directly; do not add independent edge costs and call them a jointly sparse objective.
1. endpoint full-vocabulary KL;
The cost is two full-model state evaluations per pair for each candidate, plus dense reference evaluations; with 80 pairs this is 160 state evaluations per candidate before batching. This can be orders of magnitude above MCTG and should be reported as such. A practical candidate set may be tens of exchanges, not all 224-unit pairs. If PAX cannot beat endpoint geometry with the same candidate pool and compute, it remains a diagnostic rather than a method.
3. pair-shuffled response term;
4. local MCRR prediction used only to rank the same candidates.

The response term earns a claim only if it selects exchanges that improve held-out NELBO or downstream relative to endpoint-KL and pair-shuffled controls. The local predictor is useful if it reduces the number of full evaluations without changing the selected exchange quality, but it is not evidence of correctness by itself.

1. endpoint final-state geometry;
2. paired final-logit transition geometry;
3. pair-shuffled response geometry;
4. full-vocabulary KL as a strong functional-fidelity control.

The response term earns a claim only if it selects exchanges that improve held-out NELBO or downstream relative to endpoint geometry and pair-shuffled controls. The endpoint and KL controls test whether the paired transition adds value; they do not make KL the proposed proxy.
## Why Bayesian optimization and unrestricted quadratic allocation are not first choices

Bayesian layer-ratio search is established in older quantized CNN pruning, but its black-box sample-efficiency does not transfer automatically to 224 DLM projection units with expensive full-model fitness. A GP or tree surrogate would also introduce a new optimizer confound while the metric is still unvalidated. Use Bayesian optimization only as a later search control if the candidate dimension is reduced to a small, prespecified schedule family; do not present it as the method contribution.

The proposed (Q=J^\top KJ) route has two separate problems. First, its mathematics is covered by Gauss–Newton/Fisher loss-aware pruning and by (J^\top J) sensitivity matching in 2ndMatch. Second, the existing saved-probe audit falsifies the unverified local linear extrapolation needed to use (Q) for this model and mask range. Cross terms are not empirically dominant in the available probes. A trust-region version would require new direct whole-model measurements and should be treated as an implementation of PAX, not as a new standalone axis.

## Scope and claim discipline

The first implementation choice should be **fixed Wanda support, changed projection rates**. This isolates the new DLM-conditioned criterion from support construction. If it fails, that does not prove paired context information is useless; it may mean Wanda's within-row support is the bottleneck. A later support ablation may replace Wanda prefixes with a paired-mask local solver while fixing the rate vector, but it must be reported as a separate axis.

The strongest defensible initial claim is:

> Under a fixed LLaDA-Base revision, calibration protocol, static unstructured budget and Wanda support family, a matched context-pair criterion can be tested for preserving conditional denoising behavior during projection-level rate allocation.

Do not claim universal DLM superiority, AR failure, causal preservation of generation decisions, or optimal allocation without independent NELBO/downstream evidence. Historical Uniform62 and native Uniform54 remain separate ranking families. A+C mini (61) versus A (55), Uniform (54), and historical Uniform (62) is descriptive evidence only; it does not establish that the response term caused the difference.

## Primary sources consulted

- DSA: https://proceedings.neurips.cc/paper_files/paper/2024/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf
- EvoPress: https://arxiv.org/html/2410.14649
- BESA: https://arxiv.org/html/2402.16880
- Lua-LLM: https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf
- LSA: https://proceedings.iclr.cc/paper_files/paper/2026/file/7b805585c7e249c1f65737506d4fe1e4-Paper-Conference.pdf
- 2ndMatch: https://arxiv.org/html/2506.05398
- FAIR-Calib: https://arxiv.org/html/2606.06547v2
- OPTD: https://arxiv.org/html/2608.02942
- Bayesian ratio allocation example: https://arxiv.org/abs/2002.00523

## Related local evidence

- `research/dlm_novelty_synthesis_2026-09-18.md`
- `research/allocation_novelty_reaudit_2026-09-18.md`
- `research/llm_allocation_methods_2026-09-16.md`
- `research/four_axis_saved_probe_audit_2026-09-18.json`
- `experiments/dlm_role_exchange_prediction_v2/report.md`
- `experiments/dlm_context_response50/`

## Parent review corrections and accepted priority
Endpoint reconstruction already uses one common mask across states; the paired term changes cross-state weighting/covariance rather than fixing independent masks. Equal target-energy normalization is itself a weighting choice and can amplify tiny-response noise; it does not remove the need to declare a scale/floor. Local224-by-grid matrix products are not established cheaper than whole-model search without measuring actual tokens/matrix sizes/reuse. A pair requires two state evaluations (or equivalent batch work), not one. Matching an endpoint-KL baseline at lower compute can be a useful contribution. Proxy prediction is diagnostic, not an obligatory gate. Candidate proposals have not been implemented or validated.
