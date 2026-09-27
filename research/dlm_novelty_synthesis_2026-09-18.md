# DLM pruning: Uniform 전제와 방법론 novelty 재검토
Date: 2026-09-18
Status: research review / method candidates; no experiments launched
Scope: static post-training weight pruning/allocation. Existing artifacts and primary papers reviewed with three subagents; not an exhaustive proof that no prior exists.

## 결론과 연구 포지션
기존 allocation 방법이 모두 Uniform보다 나빠야 DLM용 방법을 제안할 수 있는 것은 아니다. 연구 목적을 'AR 방법 실패 입증'에서 'DLM의 조건부 복원 구조를 반영해 동일 압축 예산의 품질·비용·견고성을 개선'으로 수정하는 것이 현재 증거와 맞다.
DLM에 맞춘 새 기준+기존 allocator도 기여가 될 수 있다. 다만 DLM을 데이터셋/모델 이름으로만 바꾸는 적용과, mask/conditional update/decision structure가 실제 설계 선택을 바꾸는 방법은 구분한다. 새 optimizer, 회로 전체 규명, 최적성, 모든 상호작용, AR 대비 현상의 고유성은 일반적인 방법론 논문의 무조건 선행 요건이 아니다. 'AR에서는 없고 DLM만의 원인'처럼 더 강한 주장을 할 때만 추가 대조가 필요하다.
성능 검증은 novelty 자체와 구별한다. 새로운 아이디어여도 아직 유용하지 않을 수 있고, 익숙한 조합도 문제에 필요한 비자명한 수정과 효율성/품질 이득이 있으면 기여가 될 수 있다. 채택/acceptance를 보장하지 않는다.

## 관측 결과: Uniform 보편 우위는 지지되지 않음
같은 LLaDA-8B-Base revision, WikiText validation551 chunks/268163tokens, exact-k MC128, shared mask draws. NELBO/PPL-bound는 낮을수록 좋다. 최종 test가 아닌 development validation이다.

| Method | NELBO | exp(NELBO) |
|---|---:|---:|
| DSA layer, bounded DLM re-search | 2.530142 | 12.555285 |
| EvoPress DLM adaptation | 2.540893 | 12.690996 |
| DSA projection, bounded re-search | 2.544038 | 12.730979 |
| Uniform rowwise Wanda | 2.550805 | 12.817419 |
| OWL projection | 2.557026 | 12.897403 |
| OWL layer | 2.569351 | 13.057348 |
| LSA layer | 2.576876 | 13.155969 |
| LSA projection | 2.579124 | 13.185582 |
| Uniform layer-global | 2.663777 | 14.3504 (rounded) |
| AlphaPruning layer | 2.670538 | 14.447742 |
| DLP layer pinned mean path | 2.698444 | 14.856601 |

Original result JSONs: experiments/dlm_ppl50/*/validation/results.json; experiments/dlm_ppl50_projection/*/validation/results.json. Mechanism review: experiments/dlm_ppl50_mechanism_review/report.md.
Wanda-based methods exact50%; EvoPress actual50.0000257% with FastOBC weight reconstruction, so not allocation-only ablation. DSA is public operators plus custom bounded controller, not official full-search reproduction. DLP pinned implementation uses mean while paper emphasizes median. LSA layer/projection differ mapping and range too. These are important provenance, not reasons to discard stronger baselines.
Initial frozen65% mini Uniform12 vs OWL5/DLP4/DSA7/Alpha1/LSA6 was real within that setup, but does not imply all original methods fail after re-search/native integration, at50%, or on another metric.
At65% full GSM8K, same frozen ranking: Uniform139/1319, Aggregate248, EIS+type263, Role268. Role versus Aggregate p=.08498; this supports feasibility of nonuniform allocation but does not isolate a proven role-specific gain.
At50% mini: matched native sparse-prefix Uniform54/A55/A+C61; historical cached DLM-Wanda Uniform62 on same100/protocol. End-to-end descriptive comparison is valid; assigning62vs61 to allocation alone is not. A+C vsA p=.145996. A/AC NELBO unmeasured; full runs stopped with no complete scores.
C1 coverage54/pooled50/Uniform54 mini;16article NELBO improves vs pooled but Uniform comparison CI includes0. It is neither proof of state-coverage necessity nor a reason to reject all shared-support algorithms.

## 원인을 해석하기 전에 구분할 구현 축
1. Importance data/calibration and dense versus sparse-prefix inputs.
2. Within-row/projection weight support selection and any surviving-weight reconstruction.
3. Between-unit sparsity allocation, granularity, allowed range, optimizer/search compute.
4. Metric and evaluation data: calibration objective, validation NELBO, final test, mini/full task.
Same 'Uniform50' does not fix axes1–2. Same layer50% with different row/projection quotas can yield different masks and performance. The observed gaps do not determine which detail caused them.

## 문헌 비교: 필요한 변화는 복잡성이 아니라 문제에 맞는 설계
| Primary source | Actual contribution | Implication |
|---|---|---|
| [OWL](https://arxiv.org/html/2310.05175v3) | Outlier-ratio signal + bounded nonuniform rates using existing pruners | New signal+budget rule can be a method; a new solver is not compulsory. |
| [ATP, ICML2025](https://proceedings.mlr.press/v267/huang25ax.html) | Propagation analysis + one-parameter arithmetic sparsity schedule | A short algorithm can support a full methodological contribution. |
| [EvoPress](https://arxiv.org/html/2410.14649) | Budget-constrained compression-vector search with model-output fitness | Generic exact-budget joint search is prior, but reusable as our optimizer/control. |
| [BESA](https://arxiv.org/html/2402.16880), [Lua-LLM](https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf) | Learn sparsity parameters/thresholds with original weights fixed | Allocation learning itself is not unoccupied; a DLM-specific objective can still contribute. |
| [Sink-Aware Pruning](https://arxiv.org/html/2602.17664) | Average soft sink scores on noisy calibration; activation reweighting feeds Wanda/SparseGPT | Closest direct DLM weight-pruning competitor. Temporal sink variance motivates it; actual metric is average soft sink reweighting, not a variance-per-weight allocator. |
| [Layer Collapse](https://arxiv.org/html/2605.06366) | Representation/outlier analysis and EIS versus DIS schedules | Early-layer protection/reversal alone is already studied. Our oracle-derived EIS+type is not automatically the paper's pure EIS. |
| [Quant-dLLM, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/805da7ef883245cb35e012cc179a5f6f-Abstract-Conference.html) | Masked calibration, multi-binary weight representation,1/2/3-bit within-layer block allocation at avg2 | DLM-tailored compression algorithm is a legitimate contribution form. Simply using masked calibration or mixed budgets is already prior. |
| [FAIR-Calib](https://arxiv.org/html/2606.06547) | Teacher frontier/reliability prior then static weighted hidden-MSE W4A4 calibration; diagnoses commit flips and amplification | Strong overlooked predecessor for our commitment proposal; not the same weight allocator or causal action-value measurement. |
| [Sparse-dLLM](https://ojs.aaai.org/index.php/AAAI/article/view/40586), [SparseD](https://arxiv.org/html/2509.24014), [DARE](https://arxiv.org/html/2605.08134) | Cache eviction, sparse attention, temporal activation reuse; DARE allocates reuse using query drift | DLM-aware temporal algorithms already exist; these do not implement static50% weight pruning. |
| [COPSD](https://aclanthology.org/2026.findings-acl.1344/), [OPTD](https://arxiv.org/html/2608.02942) | Future-context calibration / on-policy transition distillation | Future context and rollout supervision broadly have prior art; task/operator/measurement differences matter. |
| [2ndMatch](https://arxiv.org/html/2506.05398) | Image-diffusion pruned-model finetuning matches J^T J sensitivity using projections | Generic 'preserve response to perturbations' is prior too. Discrete mask-conditioned finite differences and static mask selection are different; continuous-Gaussian claims do not directly transfer. |

Expanded agent reviews: research/dlm_literature_reaudit_2026-09-18.md; research/allocation_novelty_reaudit_2026-09-18.md. Prior14-method review: research/llm_allocation_methods_2026-09-16.md. These are research judgments, not an acceptance prediction.

## Correction to previous recommendation
The 2026-09-18 commitment-value proposal remains a well-defined untested candidate, but is no longer designated the most novel direction. FAIR-Calib already provides the irreversible-frontier-error motivation, including fragile decisions and downstream amplification. Our actual intervention/Q/readout and static budget exchange differ, yet 'quantization -> pruning' alone is not an adequate novelty argument.
Expensive continuation is not a prerequisite for a DLM pruning method. Its short-horizon expected token mismatch also does not equal the agreed NELBO objective or final task accuracy. Action mismatch is not inherently harmful, and deterministic action scores can be flat across candidates. Retain as a higher-cost alternative/diagnostic rather than automatically replacing A+C.

## Candidate paths after this review (hypotheses, not execution plans)
### A. Keep a small DLM-specific criterion with an existing allocator
A+C is eligible for this route. Its endpoint term and paired conditional-response term define a different target from pooled reconstruction, though current mini evidence is inconclusive. Full-vocabulary fidelity and shuffled-pair controls help test whether paired information contributes. Gold-logodds is not disqualified solely for being scalar; the wrong-token blindspot is a limitation, not a mathematical impossibility of useful allocation.
Evidence needed for a paper: reproducible gain or a useful cost/robustness tradeoff against strong baselines and a matched ablation of the DLM-specific component. A proxy-versus-utility correlation study is useful diagnosis, not a hard prerequisite before any downstream screen.
### B. Make conditional-response preservation an actual shared-mask optimization problem
Research question: one fixed sparse set of weights must handle changing visible context. Can selecting support and allocation using paired conditional updates preserve functions that independent-state pooling loses?
A concrete local surrogate: for a fixed measurement unit and aligned persistent-query rows, define dense targets Y_D^- and Y_D^+, sparse-prefix inputs X_S^- and X_S^+. Build augmented paired data
Xbar=[X_S^-/sqrt2, X_S^+/sqrt2, sqrt(lambda)*(X_S^+−X_S^-)]
Ybar=[Y_D^-/sqrt2, Y_D^+/sqrt2, sqrt(lambda)*(Y_D^+−Y_D^-)].
Minimize ||(W elementwise M)Xbar−Ybar||_F^2 using one static binary M, with sum pruned weights fixed globally. Original surviving weights remain unchanged in this candidate definition.
This converts endpoint+update preservation into an objective used to select actual masks; it is not the existing C1 channel-coverage-to-rate proxy or the current A+C gold-logodds rank mapping. For a common linear dense input, the effective Gram adds lambda*(X+−X-)(X+−X-)^T, including cross-state terms absent from independent endpoint pooling.
Possible algorithm: generate a small support/cost curve per unit, allocate exact budget with an existing discrete solver, then check the assembled sparse model. This is a candidate, not a solved optimizer. Updating weights, dynamic per-step masks, or full training would change scope and requires a separate setup.
Limits: local linear response may miss downstream nonlinear damage; prior role reconstruction has already shown such mismatches. BESA/Lua/SparseGPT, relational/derivative distillation and2ndMatch are necessary novelty comparisons. Do not assert paired second moments are a new mathematical invention. Shuffling pair correspondence while retaining endpoint marginals is a meaningful pairing control; merely permuting the order of an already-averaged state set is not.
### C. Decoder-consequence-aware allocation
Retain previous proposal, but compare to FAIR-Calib-style frontier weighting, full-vocabulary fidelity, and same-compute generic search. It must earn its additional compute via useful quality/robustness, not by being more elaborate.

These alternatives are not a stack of mandatory modules. Do not elevate raw logvariance, arbitrary mask curvature, simple support threshold changes, or automatic sparse-context refresh merely because an agent proposed them. Prior failures constrain confidence, but do not prove the entire conceptual family impossible.

## Recommended claim and next decision
Provisional problem statement:
'We design post-training sparsity allocation for masked DLMs that preserves conditional denoising behavior under a fixed parameter budget.'
The exact claimed behavior must follow the selected method and later evidence; do not write that existing AR methods fail, that A+C is best, or that DLM-specificity is already proven.
Retain simple criterion development as an allowed main path. If broadening algorithm design, prioritize the explicit shared-mask/conditional-update problem over collecting unrelated statistics. A+C and C1 remain existing candidates with limited evidence, not proven failures or winners. No next experiment has been scheduled.
User's Sept14 scope permits existing allocator/ranking and no longer requires masked/unmasked separation as a compulsory component; record role diagnostics as useful history, not an overriding older mandate.


## Native sequential65 재측정 기록
experiments/dlm_allocation_sequential65/{method}/results.json에서 correct 확인: Uniform12, OWL9, DLP1, Alpha0, basic LSA2, DSA 재검색17 (각100문항,65%). 별도 projection LSA는12. 초기 frozen dense-mask 표의 DSA7과 native+재검색17은 동일 후보의 단순 반복측정이 아니며, pipeline과 검색 결과가 함께 바뀌었다. 기록의 comparisons_historical p값은 historical reference 비교이므로 native Uniform과의 paired p로 재사용하지 않는다. 점수상으로는65%에서도 재검색 DSA가 Uniform보다 앞선 사례가 있고, '기존 allocation은 항상 Uniform보다 못함' 전제는 더 좁혀야 한다.
