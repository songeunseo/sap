# A+C 이후 방법론 검토: 공개 결정의 후속 손상으로 예산 교환 선택
Date: 2026-09-18
Status: conceptual proposal; no new experiments, masks, model forwards, or GPU jobs.
Request: 사용자 요청으로 gpt-5.6-luna/xhigh 서브에이전트 3개가 objective / algorithm / novelty를 나누어 검토. 최종 설계 판단은 부모 에이전트가 상충하는 제안을 교정하여 작성.

## Objective
Frozen decoder 아래 static exact50% unstructured pruning의 다음 방법론 후보를 구체화한다. 새 점수가 A+C보다 실제로 우수하다는 결론이 아니라, 놓치는 의사결정 정보를 정의하고 반증 가능한 알고리즘 초안을 만든다. NELBO/PPL-bound를 기존 기본 평가로 유지하고 GSM8K를 추가 확인한다.

## Verified evidence
- Context-response mini100: Uniform54, A55, A+C61; AC vs A gained9/lost3, exact McNemar p=.145996. Reused development set.
- Historical cached DLM-Wanda Uniform50=62 on same100/doc/prompt/target hashes and evaluation protocol. Native sparse-prefix Uniform50=54. 두 mask construction은 다르다. End-to-end reference로62를 배제하면 안 되며, allocation objective의 causal comparison은 각 ranking family 안에서 matched baseline을 둬야 한다. 62를 보고 고른 family에서 같은 mini로 재선택하면 개발자료 과적합이다.
- Existing calibration joint distortion:
  - Uniform A=.8240013682, C=.3785679807, AC=1.2025693489.
  - A allocation A=.6652900871, C=.3763384520, AC=1.0416285390.
  - AC allocation A=.6589823092, C=.3669999957, AC=1.0259823048.
  AC allocation은 A와 C 모두 개선하므로61점의 원인을 C 감소만으로 귀속할 수 없다.
- A/AC의 NELBO는 아직 측정하지 않았다.
- C1 support coverage mini: coverage54, pooled50, Uniform54. Pooled대비 작은 NELBO 신호가 있지만 Uniform 우위/새 static-support mechanism은 미확립.
- Previous Global Minimax held-out KL 개선에도 mini24→16; sparse-context one-pass refresh mini19/12 vs Role24. 따라서 일반 fidelity 개선이나 반복 재수집 자체가 해결책이라는 근거는 없다.
- Full A/AC는 사용자가 중단: A107/1319, AC147/1319. Partial generation checkpoint 없음; full accuracy 없음. 재시작하지 않음.

Sources: experiments/dlm_context_response50/{results.json,allocation.json,uniform/joint_distortion.json,A/joint_distortion.json,AC/joint_distortion.json}; research/two_mini50_completion_review_2026-09-17.md; experiments/uniform_wanda_sparsity_sweep/gsm8k_mini/uniform_50.json; experiments/dlm_role_decision_audit/report.md.

## Algebraic findings, not experimental findings
Let e-,e+ be sparse-minus-dense gold-vs-rest logodds errors. With u=(e++e-)/2 and v=(e+-e-)/2:
A=E[u²+v²], C=4E[v²], A+C=E[u²+5v²].
Coefficient1 thus gives differential error five times the common-mode coefficient in these coordinates. It is a fixed design choice, not a derived optimum. This does not make C redundant: it adds cross-state error coupling.

Explicit blind-spot counterexample at masked position i:
dense probabilities (gold=.40, wrong1=.35, wrong2=.25);
sparse probabilities (gold=.40, wrong1=.55, wrong2=.05).
At another masked position j both models have max probability .50.
Gold-vs-rest logodds at i is identical, yet dense commits position j whereas sparse commits wrong1 at position i under confidence top1 decoding. If gold probabilities at every queried position are preserved at both endpoints, A=C=0 is possible despite this action difference.
This proves lack of an implication from A+C=0 to decoder-action equality. It does NOT show that this occurs frequently in our checkpoints or that every such change harms downstream quality. Full-vocabulary KL already detects the probability redistribution, so it is an essential baseline.

## Hypothesis
Some fixed-budget pruning decisions change which position/token is irreversibly committed, and the committed value subsequently changes remaining predictions. The magnitude of a probability error is insufficient to assess the cost of that decision. Weight allocation informed by its downstream cost may outperform endpoint/response matching.

The deployed generate.py with temperature0, gen_length256, steps256 uses one commitment per step, confidence top1 over masked positions; already unmasked positions are copied. No universal claim about all diffusion decoders, and no argument based on simultaneous-token independence error.

This refines the prior 2026-09-17 Decoder-Decision-Aware-Iterative-Allocation sketch. It is not a newly discovered direction or a revival of a disproven rollout experiment.

## Proposed measurable target
Use a state s from a frozen bank containing actual baseline-sparse and dense rollouts on non-GSM calibration documents; generated context and remaining masks are explicit. Mixture weights, prompt lengths, selected steps, and tie-break rules must be fixed before outcomes.

At K=1, action a_M(s)=(selected masked position, argmax token). The sparse candidate uses the existing decoder unchanged.

Fix a reference continuation model B for the whole comparison round. The first conceptual version can use the original dense model D; this makes the evaluator independent of the candidate. Dense repair can hide sparse-future damage, so the shortlist validation below uses candidate self-continuation. Changing B is a separately declared variant.

For a candidate action a:
1. Force the single action into the shared state.
2. Apply h-1 real decoder steps from the same frozen B.
3. At the resulting state z, let R be the fixed original response slots (prompt excluded), C(z) the already committed response slots and U(z)=R minus C(z).
4. Define the terminal readout
   L_B(z;y)=[sum_{i in C(z)} 1[z_i != y_i] + sum_{i in U(z)} (1-p_B(y_i|z))]/|R|.
5. Qhat_B,h(s,a)=L_B(z;y).

This readout equals expected token mismatch if each remaining slot were sampled once from B's marginal at z, retaining already committed tokens. Both summands have the same unit; no arbitrary A/C mixing coefficient. This hypothetical terminal completion is NOT our deployed greedy decoder, NELBO, or GSM8K exact-match. At full completion the second term vanishes and it equals reference token mismatch. Multiple valid text continuations remain a limitation.

The per-state relative cost is Qhat_B,h(s,a_M)-Qhat_B,h(s,a_baseline). Baseline action and reference evaluator are fixed across candidates, so optimizing this difference equals optimizing candidate Qhat. Negative differences are allowed: teacher action is not assumed optimal. Candidate-dependent reference branches would permit gaming and are disallowed.

Identical state+action+reference+schedule implies identical value, allowing exact reuse. Different actions with the same short-horizon consequence are not penalized merely for differing in order. A harmless action beyond h can still be misjudged by the truncated target.

Expected terminal scoring softens the readout conditional on an action but DOES NOT remove the discrete action-selection plateau. If every candidate takes the same action in the state bank, all relative scores are zero. State coverage and informativeness must be measured, not assumed.

## Theoretical motivation and boundary
For a fixed transition system, true full-horizon cost-to-go Q_B, and candidate state occupancy d_M:
J(M)-J(B)=sum_t E_{s~d_M,t}[Q_B,t(s,a_M(s))-V_B,t(s)].
This is the standard performance-difference principle underlying cost-sensitive imitation learning, not our theorem. Finite h, frozen surrogate state bank, and terminal reconstruction readout do not satisfy the prerequisites for a guarantee about full generation or GSM8K.
See Ross & Bagnell, AggreVaTe: https://arxiv.org/html/1406.5979v1

## Allocation algorithm, beyond the old rank mapping
Final output remains one static exact50% mask, with no new decoder or inference-time search.

1. Fix one Wanda ranking family per controlled run and initialize its Uniform50. Historical cached and native sparse-prefix families are separate controlled runs; both remain end-to-end references.
2. Propose a bounded list of EXACT-budget exchange masks: restore q weights in a receiver, remove exactly q in a donor using admissible row-quanta. Verify actual integer counts rather than continuous percentages. Same-shape/type pairs simplify counts; broaden only with correctly matched quanta.
3. Generate proposals from a mix of coarse sensitivity and prespecified structural/random strata. Do not restrict all proposals to the old A+C favorites. No unsupported C1 coverage gate in the first version.
4. Apply each whole exchange mask in the current jointly sparse model and collect its actual actions on the shared states. The receiver/donor act together; do not add independent layer damage estimates and call it joint utility.
5. Evaluate/copy Qhat for unique actions and shortlist low-risk exchanges.
6. For the shortlist, execute each candidate's OWN h-step continuation from the same acceptance states, then apply the SAME frozen terminal readout. This checks effects from changing future candidate decisions, which the shared first-action cache cannot represent.
7. Accept the best directly measured exchange only if an independent calibration acceptance batch improves the registered criterion; retain the baseline if none qualifies. Limited sequential exchanges with fixed compute cap can be considered. Refresh the state bank only after an accepted round under a declared schedule.
8. NELBO/PPL-bound on held-out documents remains an independent required check. A decoder-specialized gain is not a generic likelihood gain; GSM8K labels are never allocation fitness.

No min-cost-flow or matching on independent edge costs is assumed to solve interactions. If a multi-edge bundle is proposed, evaluate the entire bundle before accepting; disjoint matrices can still interact through the network.

## Cost and exact cache semantics
Cache key: original state/token ids, eligible mask positions, transfer quota, action tuple, reference model+mask hash, continuation horizon/schedule/seed/tie-break, terminal readout and gold target identity. Candidate-own continuation is a separate computation and cannot share the reference-action cache.

Let S=states, P=proposed exchange masks, U_s=distinct actions including baseline at state s, Rshort=shortlisted masks.
Approximate post-state-collection model-call cost:
SP + h*sum_s U_s + S*Rshort*(h+1).
The h term counts h-1 reference continuation calls and one terminal readout. Additional model hashes/mask swapping and state-bank generation also cost time. Count forward tokens/FLOPs, not only calls, if lengths differ.

Illustration ONLY, not a frozen design or runtime result: S64,P16,h4,Rshort4, meanU3 gives1024+768+1280=3072 forwards plus state collection. Worst meanU17 gives1024+4352+1280=6656. The existing A+C probes were10240 calls (32blocks*2levels*80pairs*2endpoints), excluding preparation/evaluation. Cache speedup is not established; unique-action counts may eliminate the benefit.
A few hundred proposals or exhaustive224choose2 pairing would be costly. The first diagnostic should establish action-value informativeness before full allocator construction.

## Novelty audit
- A+C context response, decoder-aware iteration, generic search, and cost-to-go reasoning are not newly invented here.
- EvoPress already does budget-constrained discrete compression search using global-model fidelity: https://arxiv.org/html/2410.14649
- OPTD already uses student-visited states and verifies teacher future commitments for few-step transition distillation: https://arxiv.org/html/2608.02942
- Order-Token Search explores position/token trajectory alternatives at inference: https://arxiv.org/html/2601.20339
- AggreVaTe already makes consequences, rather than uniform action imitation error, central: https://arxiv.org/html/1406.5979v1
- Consistent Diffusion Language Models already studies multi-path discrete consistency: https://arxiv.org/abs/2605.00161
- External DLM pruning comparators include Sink-Aware Pruning and Layer Collapse; decoder relevance alone is not a DLM-pruning novelty claim:
  https://arxiv.org/html/2602.17664
  https://arxiv.org/html/2605.06366

Possible contribution, still hypothetical: demonstrate a pruning-specific harmful-commitment failure mode and an efficiently estimated consequence signal that improves static exact-budget masks over full-distribution fidelity at matched search compute. Static weight selection is distinct from changing decoder orders or distilling fewer steps. Cache/greedy search by themselves are engineering components, not sufficient novelty.

## Alternatives considered
- Full-vocabulary endpoint KL with the same exact-budget optimizer: first strong functional control, cheaper and more mature, but generic criterion rather than a compelling novel DLM method.
- A+C with a tuned coefficient, extra progress axes, covariance normalization: simple to implement, but no evidence they repair the action blind spot or give distinct novelty.
- C1 coverage, SAE, clock directions as mandatory combined terms: adds unvalidated assumptions; not part of the first proposal. Existing role distinction is preserved diagnostically, not re-rejected.
- Arbitrary iterative recalibration: previous one-pass refresh failed; no rationale for iteration alone.
- Dynamic per-timestep masks: changes the model/deployment problem and union storage cost; outside the present static-mask task.

## Minimal falsification plan for later, not launched
1. On independently selected calibration documents, find action differences induced by prespecified exact-budget exchanges. Keep correct/incorrect committed tokens, changes in position, mask progression, masked/unmasked context routes distinct in reporting.
2. Show that a value-weighted action signal predicts subsequent candidate-self-continuation damage beyond unweighted disagreement and full-vocabulary KL. Prompt/document-level split; no millions-of-token pseudo-replication.
3. Hold the candidate set/optimizer/state bank/compute fixed while replacing fitness by A, A+C, full-vocabulary KL, action disagreement, and action value. A poor search control must not be mistaken for metric novelty.
4. Independently compare proposal heuristic versus structure/count-matched random proposals if claiming search efficiency.
5. Check both historical62 and native54 ranking families; don't call an improvement over54 a new best if62 is unbeaten.
6. NELBO remains mandatory. Frozen independent downstream and more seeds/tasks/sparsities support stronger claims only after the mechanism screen.
7. Stop or demote the candidate if informative action changes are too rare, short-horizon values fail to predict later damage, same-compute KL performs as well, gains vanish under ranking controls, or likelihood gains do not materialize for the intended claim.

## Decision
Prioritize consequence-valued decoder action under actual exact-budget exchanges as the conceptual research candidate, not as a proven superior method. Keep full-vocabulary KL+same exchange as the simpler engineering competitor. The outcome/readout choice is the largest remaining scientific risk.
No implementation, experiment, new mask, model evaluation, or restarted full run was performed in this turn.

## Related notes
- Research/DLM-Pruning/Hypotheses/2026-09-17-Decoder-Decision-Aware-Iterative-Allocation.md
- Research/DLM-Pruning/Hypotheses/2026-09-16-Context-Response-Allocation-Proposal.md
- Research/DLM-Pruning/Research-State.md


## Broader literature reassessment — 2026-09-18
FAIR-Calib (https://arxiv.org/html/2606.06547) already studies compression-induced irreversible frontier flips, post-commit mismatch and amplification. This does not implement our static pruning exchange/Q readout, but substantially narrows the conceptual novelty. Retain this proposal as an unvalidated, potentially expensive alternative; it is not a mandatory or automatically preferred method. A DLM-specific criterion with an existing allocator remains eligible. See research/dlm_novelty_synthesis_2026-09-18.md for the latest position. No experiments launched.
