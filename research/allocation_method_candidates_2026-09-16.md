# 문헌 조사에서 도출한 allocation 메서드 후보
Date: 2026-09-16
Status: proposed, untested
사용자 정정: 문헌 요약에서 끝내지 말고 조사 과정에서 메서드를 설계할 것. 아래는 설계안이며 새로운 결과나 실행된 실험이 아니다.

## Objective
고정 Wanda weight 순위와 전체 sparsity 예산을 유지하면서, DLM에서 무엇을 보존할지 정의하는 allocation proxy를 제안한다.

## Evidence
- 두 Uniform50은 정의된 dense 5×mean tail을 모두 보존하지만 NELBO가 다르다. 이는 그 tail 좌표의 생존 여부만으로 차이를 설명할 수 없다는 뜻이다.
- 선택된12교환에서 boundary Wanda-square 선호는 NELBO점추정과6/12일치. 일반화된 예측정확도 측정은 아니지만 단독 energy 설명의 반례가 있다.
- D31 perturbation의 feature direction과 receiver representation interaction은 과거 관측이다. 다른 모듈로의 일반화는 미확인.
- Clean/corrupted aggregate 순위가 비슷하며 mask-profile variation과 signed CE의 조정후 관계는 약하다. 단순 mask/timestep 민감도 proxy를 지지하지 않는다.
- 과거 reveal KL은 all-masked KL보다 mini100 점추정이 높았지만 paired evidence는 불충분. 이를 아래 후보의 성공 근거로 쓰지 않는다.

## Primary hypothesis: 새 문맥에 대한 예측 갱신 보존
LLaDA는 변화하는 visible context에서 같은 masked token을 예측한다. Sparse model이 개별 상태 예측을 비슷하게 유지해도, context가 추가됐을 때 예측을 갱신하는 방향/크기를 잘못 만들 수 있다는 가설.
현재 통계가 이 기능 손상을 직접 측정한 것은 아니다. 현 통계의 불충분성 및 모델의 조건부 예측 구조에서 도출한 미검증 가설이다.

### Measurement
같은 문장의 두 nested mask states x−, x+를 만든다. x+는 x−에서 일부 정답 context token만 추가 공개한다. 평가 query Q는 두 입력 모두에서 masked이며 공개 집합과 겹치지 않는다. Gold context를 쓰므로 ground-truth-conditioned calibration이지 실제 생성 trajectory 평가가 아니다.

모델 M의 query i 정답-vs-rest log odds:
m_M(i,x)=z_M(i,y_i,x)−logsumexp_{v≠y_i} z_M(i,v,x).
Response r_M=m_M(i,x+)−m_M(i,x−).
Dense teacher D, sparse S에 대해:
e±=m_S(i,x±)−m_D(i,x±)
A(S)=E[(e−²+e+²)/2]
C(S)=E[(r_S−r_D)²]=E[(e+−e−)²]
첫 prototype objective J=A+C. 동일 단위의 두 항을 계수1로 고정하는 설계 선택이며 최적 계수의 이론적 유도는 아님. Exact control J=A, 동일 pairs/queries/forward budget.

A는 두 endpoint의 예측 수준, C는 문맥 추가에 따른 응답을 보존한다. C 단독이면 두 상태의 공통 오차가 상쇄되는 blind spot이 있어 A를 유지한다. A+C 역시 실제 NELBO나 downstream 보장을 주지 않는다.
예: e−=e+=1과 e−=−1,e+=1은 둘 다 A=1이지만 C는0과4다. 따라서 pointwise 제곱 오차의 단순 재가중과 구분되는 cross-state covariance 정보를 쓴다.
Dense response를 무조건 크게 만들거나 정답확률 증가를 강제하지 않는다. Dense response는 음수일 수도 있다.

### Explicit allocation rule
1. Uniform50 sparse background와 rowwise Wanda nested rankings 고정. 첫 설계 단위는32 transformer layers; projection 확장은 별도 ablation.
2. 각 unit u의 sparsity를 S−δ/S+δ로 바꿔 paired-state J를 계산. 다른 unit은 baseline 유지. 이는 marginal probe이며 probe 자체는 최종 예산과 다를 수 있음을 명시.
3. c_u=[J(s_u+δ)−J(s_u−δ)]/(2δ N_u). 파라미터당 추가 제거 비용. 부호 보존, 음수 clamp/단조 손상 가정 없음.
4. c_u의 rank r_u를 [0,1]로 놓고 weighted center rbar=sum N_u r_u/sum N_u:
s_u=clip(S−a(r_u−rbar)+b, S−a, S+a).
b는 sum N_u s_u=S sum N_u를 만족하도록 선택. 동일rank ties동일값, 모두동률이면Uniform. 이후 row-quota discrete exact-budget rounding.
5. score가 큰 unit 보호. 뒤 layer 보호를 식에 강제하지 않는다. 초기 proposal δ=.02, 배분 half-width a=.05; 실험 실행 전 exact quota feasibility와 frozen protocol을 기록해야 한다.
6. 최종 joint model에서 J와 NELBO 검증. 단독 probe합이 joint효과를 보장한다고 주장하지 않으며 업데이트 반복을 기본방법으로 추가하지 않는다.

### Relation to reviewed papers
- ALS가 층 사이 representation 관계를 측정한다는 원칙에서 영감: 여기서는 같은 query의 context 변화에 대한 조건부 출력 관계를 측정.
- NeuronAl에서 작은 후보군의 기능적 alignment로 배분한다는 구조를 참고. 입력 activation alignment와 위 log-odds response는 다름.
- OWL류의 bounded allocation을 사용해 proxy 하나의 오류로 과격한 예산 이동을 방지.
- 이전 reveal KL과 다름: 곧 공개될 query를 선택/가중하는 것이 아니라, 계속 masked인 동일 query에서 **다른 context 공개 전후 차이**를 비교. 이전 실패를 새로운 성공 증거로 재해석하지 않음.
- AR에서도 context response 개념은 정의 가능하므로 DLM 독점적 개념이라고 주장하지 않음. arbitrary bidirectional mask context와 shared denoiser에 맞춘 criterion이라는 수준. 우위/새로움 검증 필요.

### Falsification and first comparison
- Primary: A+C가 같은 paired inputs/예산의 A보다 새로운 문서에서 fixed-budget exchange utility 또는 full allocation NELBO를 더 잘 보존하는가.
- 같은 mask-rate strata 내 query/문장 대응을 섞은 paired-error control은 A를 그대로 유지하고 cross-state coupling만 제거할 수 있다. 올바른 pairing의 추가효과 필요. 이 shuffled statistic은 실제 입력쌍이 아닌 진단용 null이다.
- 일반 all-masked KL, 단순depth schedule, Uniform과 비교하여 단순한 full-output probe 또는 depth bias 이상의 추가효과 확인.
- Gold reveal 비현실성, teacher 오답 보존, scalar gold-vs-rest가 오답 후보 사이의 재배치를 놓침, maskrate 변화와 실제 content contribution 혼합은 제한점. 작은 추가 공개 및 rate-stratified null로 일부 분리하되 자동 해결됐다고 주장하지 않는다.
- 기존16article/12선택pairs는 탐색 자료이며 재사용 결과를 독립검증으로 부르지 않는다.
- 반복 seed/AR대조/독립문서/fullgeneration은 후속 주장 수준에 맞춰 필요. 이 턴에서는 실행하지 않았다.

## Secondary candidate: 실제 제거 방향의 output-Fisher cost
고정 nested Wanda 경계의 ΔW를 사용해 final logits perturbation v=J_logits ΔW를 구하고 E[vᵀ(diag(p)−ppᵀ)v]/2를 측정. 실제 방향과 receiver sensitivity를 반영하며 arbitrary Gaussian absorption과 구분된다.
그러나 이는 function-aware second-order pruning의 알려진 형태이며 LLM Surgeon 등에 강한 선행연구가 있다. DLM loss/calibration으로 바꾸는 것만으로 논문 novelty를 주장하지 않는다. 필요하다면 functional baseline으로 사용.
유한한50%변화에서local Taylor의신뢰성미확인, Fisher는 gold CE실제Hessian과일반적으로동일하지않음. Signed CE를clamp해서새cost라고부르지않는다.

## Decision
논문 후보로 우선 검토할 아이디어는 context-response preservation. 구체적 proxy/anchor/배분식/핵심대조를 만들었지만 지지 실험은 아직 없다. Output-Fisher는 근거있는 대조 후보로 분류한다. 단순logvariance/정규화/기존rolemax/genericabsorption 재포장 없음.
새 GPU 작업, 설정 변경, pruning/evaluation 실행 없음.

## Sources
- [LLaDA](https://arxiv.org/html/2502.09992v3)
- [ALS](https://proceedings.neurips.cc/paper_files/paper/2024/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf)
- [NeuronAl](https://arxiv.org/html/2411.07066)
- [LLM Surgeon](https://arxiv.org/html/2312.17244)
- [[Research/DLM-Pruning/Literature/2026-09-16-LLM-Sparsity-Allocation-Methods]]
- [[Experiments/2026-09-16-Distribution-Structure-Hypothesis-Audit]]

