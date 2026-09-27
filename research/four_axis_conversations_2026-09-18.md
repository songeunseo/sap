# Axis 1 — 기존 대화에서 이미 나온 함수 보존 후보

Date: 2026-09-18  
Scope: 고정 50% unstructured weight pruning/allocation. 새 실험이나 GPU 작업은 하지 않았다.

## 역사 확인과 설계 제약

Obsidian 상태와 연구 노트를 확인한 뒤 부모가 제공한 실제 대화 추출물도 읽었다. 핵심 thread는 다음 세 개다.

- `01a0a5e9-18dc-7981-a887-2aba4dded13f` — `DLM proxy와 search 방향 결정`
- `01a0a641-ad5a-71c3-ac4a-29cb66d9408a` — `Proxy 검증 실험 설계`
- `01a0a63b-6eb0-7c61-944e-ca3ced8ed9ee` — `role-wanda 결과 표로 정리`

사용자는 첫 thread에서 “Loss를 직접 반영하고 싶지 않음”이라고 명시했고, 그때 제안은 activation geometry와 중간 계산 보존으로 전환됐다. 이후 A+C는 gold-logodds fixed expression으로 별도 승인된 최신 경로이므로 이 과거 선호를 새 hard ban으로 소급하지 않는다. 다만 새 후보의 중심은 label/loss-driven search가 아니라 **DLM에서 실제로 수행되는 함수와 정보 전달의 보존**에 두는 것이 역사와 현재 범위 모두에 맞다.

현재 수치의 경계도 유지한다. Native sparse-prefix mini는 Uniform54/A55/A+C61이고 A+C 대 A는 exact McNemar `p=.145996`; historical Uniform62는 다른 mask/ranking family다. C1 coverage는 54, pooled50, Uniform54였으므로 C1을 새 방법으로 재명명하지 않는다. A/AC full run은 중단됐다.

## 후보 1 — Operation-group preservation allocation

### 이미 제안된 근거

`DLM proxy와 search 방향 결정` thread에서 기존 “Linear 하나의 reconstruction error” 단위를 Q/K, V/output, gated-MLP branch, output-plus-residual 같은 **함께 계산되는 operation group**으로 바꾸자는 제안이 나왔다. 근거는 EXP-006의 receiver interaction이다.

- `block30.ff_out`: reconstruction 약 `0.043`, KL damage 약 `0.009`.
- `block31.ff_out`: reconstruction 약 `0.006`, KL damage 약 `0.023`.
- 같은 timestep의 foreign D31 perturbation은 native damage의 약 `38.3%`만 유지했다.

이는 “error 크기”만으로는 receiver representation과의 결합을 설명하지 못할 수 있다는 관측이다. Q/K 또는 gated branch에 일반화됐다는 증거는 아직 없다. 따라서 이는 새로 발명한 방법이 아니라, 실제 thread에서 이미 제안됐고 아직 검증되지 않은 함수 보존 후보로 보고한다.

### 가설과 objective

한 Linear의 local output을 작게 만드는 것보다 그 Linear들이 함께 만드는 계산을 dense model과 가깝게 보존하는 것이 static allocation에 더 유용할 수 있다. 첫 후보는 한 operation group을 한 번에 다루고, group별 metric을 섞어 임의의 종합 loss를 만들지 않는다.

- Q/K group: `A=softmax(QKᵀ/√d)`를 만들고 `||A_dense−A_sparse||²/||A_dense||²`를 측정한다.
- gated-MLP group: `gate(h)⊙up(h)`의 dense/sparse 차이를 같은 방식으로 측정한다.
- output/residual group: `h + out(h)`의 dense/sparse representation 차이를 측정한다.

이 값들은 label loss나 CE/KL을 allocator fitness로 직접 넣지 않는 중간 함수 보존 신호다. 하나의 group만 먼저 고정하고, D31의 결과를 Q/K·MLP에 자동 전이하지 않는다.

### 예산 선택

Group 내부 weight ranking은 native row-wise Wanda를 고정한다. Q/K, V/output, MLP branch를 각각 독립 Linear로 흩어 배분하지 않고, group별 depth/type rate를 소수의 고정된 parameter로 제한한다. 예를 들어 baseline group rate에서 같은 exact global 50%를 유지하는 bounded `[45%,55%]` group schedule을 사용하고, 실제 row-quanta로 반올림한다. 이 low-dimensional schedule 또는 기존 exact-budget allocator를 backend로 쓸 수 있다.

DSA public-operator/custom `population=8, generations=4, seed=0, lambda=.08`는 필요하면 고정된 allocation backend control로 사용할 수 있다. 원 논문의 DSA PPL search와 local masked-CE/custom controller transfer는 provenance를 분리한다. optimizer나 search 자체를 새 기여로 주장하지 않고, group function metric만 바꾼다.

### 기존 것과의 차이

이는 C1 coverage, role max, timestep weighting의 재포장이 아니다. 평가 단위를 projection에서 **계산 경로**로 바꾸며, 기존 Aggregate가 local output error를 측정한 뒤 최종 sparse model에서만 확인했던 것과도 다르다. EvoPress처럼 전체 후보를 반복 평가하는 일반 search와도 구분한다. 다만 grouped functional preservation 자체는 실제 thread에서 이미 제안됐으므로 새 novelty라고 쓰지 않는다.

### 최소 반증 비교

같은 Wanda mask family, exact budget, calibration compute에서 operation-group metric을 ordinary per-Linear reconstruction, Aggregate, Uniform과 비교한다. Q/K 또는 gate pair를 실제 counterpart와 cardinality-matched random pair로 바꾼 negative control도 둔다. 중간 함수 보존이 dense reference에 더 가깝지 않거나, group metric이 downstream WikiText NELBO/PPL과 GSM8K에서 기존 Aggregate보다 일관되게 나쁘면 이 후보를 내린다. utility prediction은 진단이지 hard gate가 아니다.

### 위험

D31은 한 module의 receiver alignment 결과다. Q/K와 gated MLP에 일반화되지 않을 수 있고, attention matrix 차이가 최종 denoising quality를 보장하지 않는다. Group schedule이 너무 거칠면 유용한 projection별 차이를 지우며, group metric이 loss-free여도 마지막 평가는 여전히 NELBO/PPL과 task로 해야 한다.

## 후보 2 — Matched reveal-response function preservation

### 이미 제안된 근거

같은 `DLM proxy와 search 방향 결정` thread에서 “새 token이 공개됐을 때 다른 masked token의 hidden state가 어떻게 변하는가”를 보존하자는 별도 제안이 있었다. 관측 단위는 token의 상태별 reconstruction이 아니라 **정보 공개 전후의 response vector와 그 response들의 보완성**이다. 이는 Role reconstruction과 다르고, A+C의 gold-logodds 경로와도 다르다.

A+C mini61은 이 방향의 약한 개발 단서일 뿐이다. gold context 추가와 mask count 감소가 함께 일어났고, A+C 대 A도 유의하지 않다. 따라서 A+C를 이 함수 보존 후보의 실증 증거로 소급하지 않는다. A+C는 최신 승인된 separate control로 유지한다.

### 가설과 objective

같은 query와 response span을 유지하면서 context token 하나만 공개한 matched pair `(x−,x+)`를 만든다. Layer/group의 dense response를

`r_D,e = h_D,e(x+) − h_D,e(x−)`

로 두고 sparse candidate의 `r_M,e`와 비교한다. 1차 objective는

`R(M)=mean_e ||r_D,e−r_M,e||²/(||r_D,e||²+ε)`.

여러 공개 사건을 모은 뒤 response direction이 같은 사건만 과보호하지 않도록 covariance/coverage를 **진단**으로 기록한다. 처음부터 diversity penalty나 role max를 임의로 더하지 않는다. 이 방법은 loss를 중요도 계산에 넣지 않고 DLM의 반복적인 정보 공개 기능을 직접 보존하려는 후보다.

### 예산 선택

기존 Wanda 내부 ranking을 유지하고, response metric으로 group/depth schedule 또는 제한된 projection exchange의 marginal을 만든다. Global exact 50%와 `3,489,660,928` pruned weights를 맞추며, donor/receiver 교환은 row-quanta 단위로 수행한다. 모든 후보에서 동일한 existing allocator를 사용한다. full search를 새로 발명하거나, response metric과 DSA fitness를 동시에 섞지 않는다.

### 기존 것과의 차이

현재 A+C는 gold-vs-rest log-odds endpoint와 context response를 함께 쓰지만, gold context reveal이 mask count를 바꾸며 query의 wrong-token redistribution을 놓칠 수 있다. 이 후보는 matched mask count의 hidden response를 직접 보존한다. 그러나 response preservation, activation alignment, token-context pruning 선행이 있으므로 novelty는 “최초 response metric”이 아니라, **이 response signal이 static exact-budget allocation에서 기존 local reconstruction보다 유용한지**로 한정한다.

### 최소 반증 비교

같은 pair와 candidate mask를 사용해 다음만 바꾼다.

- ordinary pooled hidden reconstruction;
- endpoint-only hidden preservation;
- matched response preservation;
- `−/+` pair correspondence를 섞은 shuffled control;
- same-mask-count random pair;
- 최신 승인 경로인 A+C를 별도 비교군.

Shuffled pair와 matched pair가 같거나, response가 dense function을 더 잘 보존해도 held-out NELBO/PPL·GSM8K에서 개선이 없으면 후보를 낮춘다. 이 예측 검사는 우선순위를 정하는 진단이며, 실행 전 hard gate로 만들지 않는다.

### 위험

Hidden response가 유용한 정보 전달이라는 보장이 없고, context pair의 내용 선택이 결과를 쉽게 좌우한다. Response magnitude가 큰 방향을 보존하는 것이 token correctness와 다를 수 있다. A+C와의 mask-count confound를 고쳐도 실제 rollout trajectory를 완전히 재현하지 않는다.

## 제외하거나 낮출 후보

- **C1 cross-state support coverage:** pooled 대비 작은 NELBO 신호는 있으나 Uniform과 동률이다. 이미 제안·실행된 후보를 새 방법으로 재명명하지 않는다.
- **DLM-loss marginal damage:** thread에서 제안됐지만 사용자의 “Loss를 직접 반영하고 싶지 않음”과 충돌한다. 최신 PPL은 최종 평가로 유지하되, 이 proxy를 주 방법으로 밀지 않는다.
- **generic EvoPress/DSA search:** 전체-model fitness search의 강한 비교군이지만 새 DLM 방법 자체가 아니다. DSA는 원 논문 PPL 방법과 local masked-CE transfer를 분리해 기록한다.
- **timestep weighting, role max/minimax, sparse refresh, commitment cost-to-go:** 현재 증거 없이 다시 중심 후보로 올리지 않는다.

## 판단

이번 history에서 가장 실질적인 두 후보는 (1) Q/K·gated-MLP·residual처럼 함께 계산되는 operation group을 보존하는 allocation, (2) matched mask-count context reveal 전후의 hidden response와 response coverage를 보존하는 allocation이다. 둘 다 이미 대화에서 제안된 방향이므로 novelty로 포장하지 않고, “DLM 함수 보존 기준이 실제 exact-budget allocation에 추가 가치를 주는가”를 검증할 후보로만 남긴다.

기존 allocator를 재사용하고 loss를 중요도에 직접 넣지 않는 것이 과거 사용자 선호와 현재 범위에 맞다. A+C는 최신 승인된 gold-logodds control로 보존하되, 과거의 loss-free function-preservation 선호를 무시해 CE/KL search를 새 중심으로 만들지 않는다.

## History pointers

- `research/four_axis_history_additional_2026-09-18.json`: thread `01a0a5e9-18dc-7981-a887-2aba4dded13f`, 특히 grouped operation 보존, reveal-response 보존, “Loss를 직접 반영하고 싶지 않음”, carrier/bulk와 low-dimensional schedule 제안.
- `research/four_axis_history_extract_2026-09-18.json`: thread `01a0a641-ad5a-71c3-ac4a-29cb66d9408a`의 proxy curve/비가산성/상호작용 보정 제안; thread `01a0a63b-6eb0-7c61-944e-ca3ced8ed9ee`의 Aggregate 대 EvoPress 전체-model fitness 구분.
- `Research/DLM-Pruning/Hypotheses/2026-09-16-Context-Response-Allocation-Proposal.md`: A+C 정의와 현재 confound.
- `Research/DLM-Pruning/Hypotheses/2026-09-17-Literature-Inspired-Candidate-Backlog.md`: C1/C3/C4/C6 후보와 이미 확인된 한계.
- `research/two_mini50_completion_review_2026-09-17.md`: Uniform54/A55/A+C61, coverage54/pooled50/Uniform54.
- `research/baseline_reaudit_2026-09-18.md`: DSA layer NELBO2.530142 대 row-wise Uniform2.550805 및 provenance 구분.

