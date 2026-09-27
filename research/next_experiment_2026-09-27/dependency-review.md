# Next-experiment dependency review

**Date:** 2026-09-27  
**Scope:** literature-to-design review only. No GPU/model run, mask regeneration, source mutation, or Obsidian write was performed for this review.

## Decision

The controller's selected next experiment is **Dense/A50 × 256/64/32 denoising steps**, on the frozen exposed200 development questions. The authoritative design is [proposal.md](/home/tmluser1/sap/research/next_experiment_2026-09-27/proposal.md).

This agent initially proposed an A/Multi schedule interaction. That remains a possible secondary comparison, but it cannot identify additional damage relative to an unpruned model. The selected design includes dense generation at every schedule and uses one fixed A mask first. No new confidence/attention allocation score is introduced.

## Research-state and project evidence

I freshly called Obsidian `get_sync_status` and read `Research/DLM-Pruning/Research-State.md`; both returned valid responses, so the Obsidian connection is connected. I also read the relevant confidence, role, and latest full-crosschain notes. The original experiment role document, project experiment-role adaptation, and current contract were read before this review.

The local completed-result evidence constrains the design:

* EXP-004's confidence/reveal weighting was only suggestive: `REVEAL-ABS` was 767/1319 versus `REMAIN-ABS` 747/1319 (+1.5163 percentage points, exact p=0.0901229). No weighting-versus-UNIFORM comparison survived Holm correction; the masks differed in only 0.1957% of weights and their scores had Spearman 0.999971.
* Earlier role/aggregate allocation diagnostics did not establish a role-specific gain. Local reconstruction and downstream task behavior sometimes moved in opposite directions.
* The completed full crosschain result is A 646/1119, Multi 636/1119, Cross 637/1119, and CrossMatched 634/1119. The three planned comparisons all had Holm p=1.0 and intervals spanning zero. The added C term and natural-chain pairing therefore have no established advantage under the current protocol.
* The deployed generation setting is temperature 0, generation length 256, block length 256, 256 denoising steps, and low-confidence remasking. With one block, this schedule transfers one token per denoising forward. The current results therefore do not test the low-NFE, simultaneous-commit regime emphasized by the papers below.

These observations motivate a decoder-regime interaction test, not a restart of the rejected confidence-weighting intervention.

## Primary-source audit

### 1. The Confidence Shortcut

**Primary source:** [Kim and No, *The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models*](https://arxiv.org/abs/2605.29123), full text [§§2–5 and Appendices A/D](https://arxiv.org/html/2605.29123v1).

**What was tested.** The paper trains small task-specific masked diffusion transformers, approximately 0.4M–21M parameters, rather than LLaDA-8B. It compares uniform random masking with PAPL (confidence-weighted loss) and PUMA (teacher-forced confidence-path masking), under confidence decoding and a task-specific dependency-respecting decode when available. The main controlled lens is 32-digit addition with 33 output digits, 20,000 training instances, 300,000 iterations, batch 256, and three seeds. The extension covers 10×10 mazes rendered on 21×21 grids, ListOps, Countdown, and 9×9 Sudoku. Training iterations are reported; there is no standardized large-model NFE sweep. The addition trajectory has 33 output positions, but that is not a general NFE comparison.

**Observed claim.** On long carry chains, confidence decoding can commit a locally easy digit before its carry dependency is resolved. On the chain-length-≥28 stratum, random masking scored 0.992, PUMA 0.908, and PAPL with α=1 0.026 under confidence decoding; PUMA recovered to 1.000 under the LSB-first order, while PAPL remained near zero. The failure was highly confident and concentrated at the chain-MSB position. Across the other tasks, confidence-aligned training was harmful on some dependency-ordered problems and helpful on Sudoku, where confidence often matched constraint readiness.

**Causal status.** Within these small, synthetic/task-specific models, changing the training intervention and changing the decode order are controlled interventions. They support a causal statement about those interventions in those tasks. The result is not a causal test of weight pruning, weight importance, or a LLaDA-8B static mask.

**What does not follow for pruning.** The paper does not measure any weight-level importance score, static sparsity mask, row/projection allocation, or post-training pruning outcome. PAPL changes the training loss and PUMA changes the training-state distribution; neither is a static weight-selection rule. The result supports “confidence is not guaranteed to indicate logical readiness.” It does **not** show that confidence is anti-correlated with weight importance, that low-confidence weights should be protected, or that confidence weighting is a valid or invalid static pruning criterion for this project.

### 2. Attention-Discounted Adaptive Sampler (ADAS)

**Primary source:** [Sahin et al., *Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models*](https://arxiv.org/abs/2606.10829), full text [§§3–7 and Appendices A/B/D/E](https://arxiv.org/html/2606.10829v4).

**What was tested.** ADAS uses LLaDA-8B-Base and Dream-7B-Base on GSM8K, MATH500, HumanEval, and MBPP. It is inserted into Top-k, Fast-dLLM, and EB-Sampler. The decoder operating points sweep Top-k `k∈{1,2,4,8,16}`, Fast-dLLM budgets, and EB entropy budgets; moving toward more tokens per step lowers NFE. The main tasks use maximum generation lengths 256 for GSM8K and 512 for the other tasks, temperature 0. The real-data dependency diagnostic uses 100 examples per task/model, five trajectory states, and 3,500 selected-position pairs per task/model; it reveals one position in a counterfactual copy and reruns the denoiser. The paper reports matched-NFE curves rather than one universal NFE.

**Observed claim.** In the diagnostic, final-layer attention from a position to a selected position is positively associated with the KL change at the position after the selected token is revealed. Across all eight task/model settings, the 95% paired bootstrap interval for the attention-discounted ranking improvement is positive. The paper reports average low-NFE gains of 9.11 percentage points for LLaDA and 10.46 for Dream across the four tasks and three samplers. Across 90 matched operating points, the mean gain is +9.27 with a bootstrap interval [+7.74,+10.84], with 80 improved and 10 regressed.

**Causal status.** Revealing one token and rerunning the model is an input intervention, so it measures a conditional distribution change. The association between pre-reveal attention and that change is still a proxy relationship; the paper does not randomize attention or establish that attention itself causes the change. The end-to-end ADAS comparison is a decoder intervention, not a pruning intervention. The 90-point bootstrap is explicitly across operating configurations, not fixed-setting significance.

**What does not follow for pruning.** ADAS never selects or scores weights, changes a mask, or allocates a static parameter budget. Its attention is dynamic, state-dependent, and extracted from the final denoiser layer. The paper’s evidence validates attention as a useful proxy for a particular **token-position interaction during decoding** in two models and four tasks. It does not validate attention-weighted static pruning, cross-layer allocation, or projection-level weight importance. The paper itself lists syntax/position/formatting confounding, missing higher-order interactions, two-model/task coverage, confidence miscalibration, irreversible commits, and lack of fixed-setting significance as limitations.

### 3. Parallelism and Generation Order in Masked DLMs

**Primary source:** [ACL Anthology publication](https://aclanthology.org/2026.findings-acl.357/), with the authors’ [full text and section numbering](https://arxiv.org/html/2601.15593) used to inspect the methods, experiments, and limitations.

**What was tested.** The study evaluates eight mainstream masked DLMs, up to 100B parameters, against autoregressive baselines on 58 benchmarks across knowledge, mathematics, reasoning, language understanding, agentic tasks, and coding. It reports one inference per instance under a unified local pipeline and documents model-specific block sizes rather than imposing one common NFE. A mechanistic analysis focuses on the 100B LLaDA2-flash using Average Finalization Parallelism (AFP) and Kendall’s τ. Appendix A also reports small controlled Sudoku and MathMatrix studies and a preliminary generate-then-edit experiment on LLaDA2-mini-16B.

**Observed claim.** The MDLM update factorizes jointly updated positions as a product of marginal predictions (§3), which creates a conditional-independence/factorization gap when those positions are coupled (§4.3). Across the benchmark suite, more aggressive parallelism is associated with lower accuracy. AFP and generation order vary by task domain, decoding stage, and correctness; logic-heavy tasks often have high τ, while structural/formulaic spans can be finalized in parallel. In the small Sudoku study, LLaDA-flash-100B scored 78/100 zero-shot, and fine-tuned Dream-7B reached 80/100 after 10 epochs on 50 training puzzles. These are observations under the paper’s protocols, not evidence about static sparse masks.

**Causal status.** AFP and τ analyses are observational summaries of generated trajectories. The factorization bound is a theoretical decomposition, not a weight intervention. The Sudoku architecture/training comparison and the generate-then-edit study change model training or inference, so they do not isolate a static weight mask. The paper’s own limitations caution that unified-pipeline scores are not leaderboard-equivalent and that hardware/model implementation differences matter.

**What does not follow for pruning.** The paper does not compare dense and statically pruned versions of the same model, and it does not identify which weights preserve inter-token dependencies. It supports testing whether a fixed sparse mask is more fragile when the decoder commits multiple coupled positions. It does not support claiming that an AFP, τ, attention, or generation-order statistic is a weight-importance score, nor that Generate-then-Edit is a pruning method.

## Cross-paper conclusion

Confidence can be logically premature, attention can be a useful proxy for a revealed token's conditional influence, and parallel commit size changes the dependencies that decoding must handle. These facts motivate an experiment on the existing dense and sparse model. They do not establish a weight-importance measure.

The current decoder commits one token per step. ADAS Table 19 makes the regime boundary concrete: for LLaDA GSM8K, k=1 gives 69.98→69.98, k=4 gives 53.90→50.11, and k=8 gives 22.67→42.46. These are the paper's settings, not the project's measured dense scores. An average low-NFE gain should not be applied to every operating point.

## Selected bounded experiment

### Objective and hypothesis

Does reducing NFE increase the accuracy penalty of the existing A50 mask relative to dense? A positive interaction would justify investigating low-NFE pruning robustness. It would not identify conditional dependence as the cause.

### Setup and primary quantity

Use Dense and the frozen A exact50% mask at 256, 64, and 32 steps. Hold generation length and block length at256; these schedules select1,4,8 tokens per forward. Freeze the same revision, BF16, tokenizer, five-shot prompts, temperature0, confidence selection, seeds, and strict grader.

Use the frozen exposed200 as development data, with validated A256 predictions reused. Previously analyzed primary1119 questions are not a new confirmation set. Match prompt, tokenizer, model/mask, decoding, and grading identities before reusing any cache.

Define gap(T)=accuracy(Dense,T)−accuracy(A,T). The primary quantity is I32=gap(32)−gap(256);64steps is secondary. Resample questions with all four correctness values kept together to report a paired interval. Report all absolute accuracies and paired gains/losses as well. A performance floor at32steps can shrink the gap even when both models are badly affected.

### Shared-state reveal diagnostic

At dense32 trajectory states before steps0/8/16/24, select the highest-confidence dense position from its Top-8. In a counterfactual copy, reveal that position's dense argmax token. Feed the identical before and after inputs to D/A; keep the other seven positions masked.

Measure full-distribution response magnitude, sparse-minus-dense response error, endpoint KL before and after, and argmax changes. Use document-level aggregation and uncertainty. This avoids confusing different model-generated inputs with different model responses. It does not separate semantic reveal from every mask-progress effect or prove that a response difference causes an incorrect answer.

**Endpoint KL is not NELBO.** A later allocation comparison should measure independent NELBO separately under its specified estimator. Fixed-state fidelity measurements must not be relabeled as NELBO or downstream quality.

### Interpretation

- Positive I32 with a reasonably narrow interval supports a pruning-by-decoder interaction for this model and mask. Confidence calibration, general quality degradation, and different visited states remain competing explanations.
- An interval spanning zero is inconclusive when wide. It does not establish equivalence or automatically terminate every dependency-based method.
- Similar dense and sparse degradation provides no demonstrated additional pruning penalty at that setting. This comparison is possible because dense is actually evaluated in all three schedules.
- A response diagnostic alone does not establish useful allocation. If a subsequent method is tested, compare full-distribution endpoint fidelity and a response-aware objective under the same mask family, exact budget, search budget, and independent quality evaluation.
- No universal proxy-prediction or NFE-interaction gate is imposed on every future method. This is a current research priority, not a necessary condition for all useful pruning criteria.

### Cost and artifact boundary

The planned batch1 generation count is200×2×(256+64+32)=140800 calls. Reusing the validated A256 cache saves51200, leaving89600 new generation calls. The four-state single-reveal diagnostic costs at most200×4×2×2=3200 calls without before-state caching. These are planned call counts, not measured runtime; record preparation, model loading, token counts, instrumentation validation, and actual execution separately.

Reuse physical masks and prompt identities. Write new protocol hashes and outputs for the new NFE settings. Do not mutate old configs/results or invoke a validation command that rewrites their receipts. No GPU or model experiment was started in this review.

## Controller integration

The source audit was produced by the experiment agent using experiment.md. The controller selected the dense-controlled six-condition proposal and corrected the draft's KL/NELBO conflation and its overly strong interpretation of intervals spanning zero. The original A/Multi-only proposal is superseded by proposal.md for the current recommendation.

## References

* Kim and No, [*The Confidence Shortcut*](https://arxiv.org/abs/2605.29123), especially §§3.1–3.5, §4, §5, and Appendix A.
* Sahin et al., [*Attention-Discounted Adaptive Sampler*](https://arxiv.org/abs/2606.10829), especially §§4–7 and Appendices A/B/D/E.
* Zhong et al., [*Parallelism and Generation Order in Masked Diffusion Language Models*](https://aclanthology.org/2026.findings-acl.357/); full text [arXiv HTML](https://arxiv.org/html/2601.15593), especially §§3–5, §6, Limitations, and Appendices A.1/A.6/A.7.
