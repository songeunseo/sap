# DLM static allocation 방법론·novelty 재감사

Date: 2026-09-18  
Status: analysis only; no GPU, forward, mask, or Obsidian write  
Scope: OWL, AlphaPruning, DLP, LSA, DSA, ATP, ALS, MRP, EvoPress, BESA, NeuronAl, Lua-LLM 및 현재 DLM pruning 증거

## 결론을 먼저 적으면

AR allocation method보다 항상 좋아야 한다거나, 새로운 circuit을 완전히 규명해야 한다는 기준은 필요하지 않다. 기존 연구도 대체로 `관찰 또는 원리 → 측정량 → 예산 규칙 → 동일 예산 성능/비용 검증`을 기여 단위로 삼았다. 따라서 DLM-specific static allocator는 다음 조건을 충족하면 방법론 기여가 될 수 있다.

1. 현재 Uniform/Wanda 결정이 보지 못하는 DLM 상태 정보를 사전에 측정한다.
2. 그 측정량이 실제 mask allocation으로 변환되는 규칙을 명시한다.
3. 같은 ranking family, sparsity, calibration budget, model revision에서 Uniform과 metric-matched control을 비교한다.
4. 독립 문서의 NELBO/PPL과 downstream을 모두 확인하고, metric의 추가 정보가 allocation utility를 예측하는지 보인다.

반대로 `DLM이라서 다르다`는 설명, 기존 allocator에 새 scalar를 붙인 것, 개발 mini의 점수 하나, 또는 generic search만으로는 약하다. 최적성 증명, 모든 cross-layer interaction의 설명, AR 전체보다 우월하다는 주장은 필수가 아니지만 주장 범위를 좁혀야 한다.

## 문헌에서 실제로 기여로 인정된 단위

아래는 각 논문의 primary source가 실제로 제공하는 측정 대상과 allocation mechanism이다. 논문 간 수치 순위가 아니라 방법론의 차이를 비교한다.

| 방법 | 측정 대상 | allocation/optimization | 현재 프로젝트와의 경계 |
|---|---|---|---|
| [OWL](https://arxiv.org/html/2310.05175v3) | Wanda형 weight×activation score의 layer outlier ratio | outlier가 많은 layer를 보호하는 제한폭 layer sparsity mapping | activation outlier 구조와 allocation의 연결이 기여다. DLM에서 같은 outlier 현상을 관찰하는 것만으로는 부족하다. |
| [AlphaPruning](https://arxiv.org/html/2410.10912v1) | weight ESD의 heavy-tail power-law shape | PL exponent로 block별 sparsity를 정하고 전체 예산 보정 | weight spectral shape→prunability 연결과 여러 모델/고 sparsity 검증이 기여다. 단순 variance/shape metric은 새롭지 않다. |
| [DLP](https://arxiv.org/html/2505.23807v1) | weight와 activation을 결합한 layer reducer, median 기반 redundancy | median으로 layer importance를 만들고 layer sparsity를 동적으로 매핑 | 현재 구현의 DLP reducer와 논문 구현을 구분해야 한다. median/robust reducer 자체는 DLM novelty가 아니다. |
| [LSA](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b805585c7e249c1f65737506d4fe1e4-Paper-Conference.pdf) | 가상 pruning에서의 covariance-aware linear reconstruction error | layer/projection group error를 사용해 finer-grained budget allocation | cross-term을 측정하는 것은 유의미하지만, error를 계산하는 것과 최종 task loss 최적 allocation은 다르다. |
| [ALS](https://proceedings.neurips.cc/paper_files/paper/2024/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf) | intermediate layer 간 normalized activation correlation/relevance | 관계 행렬을 이용한 linear programming allocation | 단일 layer statistic을 inter-layer relation으로 바꾼 것이 기여다. DLM mask state를 붙이는 것만으로 자동 novelty가 되지 않는다. |
| [ATP](https://arxiv.org/html/2502.14770) | layer reconstruction error의 forward propagation 누적 | monotone arithmetic progression과 한 개 common-difference 탐색 | 단순 depth schedule도 propagation 가정과 검증이 있으면 방법이 된다. 현재 실험은 depth schedule의 우위를 지지하지 않았다. |
| [MRP](https://arxiv.org/html/2503.18377v1) | 현재 sparse model의 layer non-outlier ratio | 매 iteration 가장 redundant한 layer를 재측정하여 pruning | dense one-shot 통계를 sparse-state 재측정으로 바꾼 것이 기여다. 반복 자체를 새 방법이라고 할 수 없다. |
| [DSA](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf) | element score를 layer importance로 줄이는 함수와 mapping | preprocess/reduce/transform/postprocess 조합을 evolutionary search | 새 scalar가 아니라 allocation-function discovery와 multi-task 검증이 기여다. generic search는 재사용 가능한 baseline이다. |
| [EvoPress](https://arxiv.org/html/2410.14649) | block별 compression vector와 dense/sparse functional fitness | exact global budget mutation + elitist evolutionary selection; KL/PPL fitness | 전체 모델 후보를 직접 비교한다. 따라서 candidate-specific utility search 자체는 새 novelty가 아니다. |
| [BESA](https://arxiv.org/html/2402.16880) | blockwise reconstruction과 후보 sparsity 조합 | block sparsity coefficients를 학습해 reconstruction을 줄임 | granularity와 allocation parameter 학습이 선행되어 있다. row/block granularity 주장은 BESA와 구분해야 한다. |
| [NeuronAl](https://arxiv.org/html/2411.07066) | dense/sparse projection input activation alignment | block 폭과 row 폭을 alignment로 고르고, Wanda ranking은 유지 | internal activation alignment는 이미 선행된다. DLM-specific state conditioning이 추가 정보여야 한다. |
| [Lua-LLM](https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf) | end-to-end performance loss에 대한 row threshold | soft Top-K와 STE로 layer/intra-layer sparsity를 동시에 학습 | global/row allocation을 task loss로 직접 학습하는 강한 baseline이다. 새 static objective는 Lua-LLM과 같은 budget/compute 축에서 비교해야 한다. |

이 문헌군은 두 계열로 나뉜다. OWL/Alpha/DLP/LSA/ALS/ATP/MRP는 무엇을 중요도·중복성·전파위험으로 볼지 정하고, DSA/EvoPress/BESA/NeuronAl/Lua-LLM은 allocation function, candidate vector, threshold를 직접 탐색하거나 학습한다. 현재 DLM 제안도 이 두 계열 중 어느 기여를 하는지 분명히 해야 한다.

## 현재 결과가 말해 주는 것과 말하지 않는 것

### Context-response A+C

Mini-100은 Uniform 54, A 55, A+C 61이었고 A+C 대 A exact McNemar는 `p=.145996`이다. A+C/ A의 NELBO는 측정되지 않았다. Calibration distortion은 Uniform `(A=.824001, C=.378568)`, A `(A=.665290, C=.376338)`, A+C `(A=.658982, C=.366999)`로 두 항 모두 동시에 바뀌었다. 따라서 downstream 점수를 C 항의 인과효과로 귀속할 수 없다.

같은 100문항과 protocol hash를 가진 historical cached DLM-Wanda50은 62점이었고 native sparse-prefix Uniform은 54점이었다. Mask construction/ranking family가 달라 allocation-only superiority attribution은 금지해야 하지만, end-to-end descriptive comparison에서는 historical 62가 더 높은 점수라는 사실을 보고해야 한다. 따라서 A+C 61을 현재 최고 결과로 부를 수 없다. Full A/AC 실행도 중단되어 1319문항 결과가 없다.

A+C가 수학적으로 놓치는 것은 실제 confidence-based action, wrong-token 재배열, mask-rate와 content의 동시 변화다. 그러나 이 blind spot이 빈번하고 downstream에 해롭다는 것은 아직 관측되지 않았다. 따라서 commitment-value objective는 “필요한 수정”이 아니라 검증할 수 있는 가설이다.

### Role/aggregate reconstruction

65% full GSM8K에서 Uniform 139, Aggregate 248, Role 268이었다. Role 대 Aggregate는 +20, exact McNemar `p=.08498`; Role 대 EIS+type은 유의한 우위가 없었다. Role/Aggregate mask 차이는 224 projection 중 20개, 전체 mask의 0.4328%였다. 이는 reconstruction-informed allocation이 큰 방향성 신호일 수 있다는 근거지만, role 분리 또는 max aggregation이 원인이라는 증거는 아니다.

V2 role exchange에서 role+dense/sparse bundle은 pooled보다 MSE가 4.05% 낮았지만 role만의 기여는 분리되지 않았다. 양 role local reconstruction이 좋아져도 functional KL이 악화된 경우가 40.52%였다. 그러므로 role은 유지할 분석 축이지, 고정 max/mean allocator를 정당화하는 결과가 아니다. 2026-09-14 사용자 범위 결정에 따라 role 전체를 기각하지 않으며, 메인 method에 필수로 넣지도 않는다.

### Shared-state support coverage

Coverage는 mini GSM8K 54, pooled support는 50, Uniform은 54였다. 16개 개발 article NELBO는 coverage가 pooled보다 `-0.001328` 낮았고 bootstrap 95% interval은 `[-0.002221,-0.000367]`, 13/16 article이 개선이었다. 그러나 coverage 대 Uniform interval은 `[-0.003280,+0.001323]`로 0을 포함한다. 이는 C1의 pooled-state 실패를 고치는 탐색 신호이지, Uniform보다 좋은 static-support method의 증명이 아니다.

## 세 가지 현실적 DLM static allocation 방향

아래는 서로 결합하지 않고 각각 하나의 가설로 검증해야 한다.

### 1. Cross-state support coverage allocator — 우선 후보

가설은 평균 activation energy가 아니라, 여러 denoising state에서 필요한 channel support의 합집합이 static weight capacity를 결정한다는 것이다. 현재 C1 결과는 pooled control과의 방향성 차이를 제공하지만, Uniform 대비 우위가 없어 주 근거로 승격할 수 없다.

Projection (u), state (s), input channel (j)에 대해

\[
e_{suj}=\|x_{s,j}\|_2^2\sum_i W_{u,ij}^2.
\]

공통 pooled top-(k) support (P_u(k))와 state별 최적 support (P_{su}(k))를 계산하고

\[
V_u(k)=E_s\left[
\frac{\sum_{j\in P_{su}(k)}e_{suj}-\sum_{j\in P_u(k)}e_{suj}}
{\sum_j e_{suj}+\epsilon}
\right]
\]

를 capacity mismatch로 둔다. (V_u)가 크면 state별 active support를 하나의 static mask로 표현하기 어렵다고 해석해 해당 projection의 sparsity를 낮춘다. 실제 weight mask는 기존 row-wise Wanda ranking으로 고정하여 channel support proxy와 weight pruning을 분리한다.

필수 control은 pooled support 자체, state 내 cardinality-matched random support, clean/corrupted state split, document-held-out support gap이다. 초기 진단으로 (V_u)가 out-of-document exact-budget exchange utility와 어떤 관계가 있는지 확인하고, 최종 후보라면 pooled support와 같은 allocation 폭에서 Uniform 대비 독립 NELBO를 보고한다. utility 예측력이 약하거나 channel support가 unstructured row mask capacity로 이어지지 않으면 이 후보의 우선순위를 낮춘다; 이는 downstream을 시작하지 말라는 hard gate가 아니다.

장점은 rollout이 없어 비용이 낮고 C1의 관측에 직접 연결된다는 점이다. 위험은 activation support와 실제 downstream 기능의 연결이 아직 가설이며, (k), energy threshold, state 수를 바꾸면 metric이 쉽게 변할 수 있다는 점이다.

### 2. Matched-mask paired-response reconstruction — A+C의 실용적 후속 후보

현재 A+C의 가장 큰 문제는 context response를 재는 것 자체보다, gold reveal이 내용과 mask 수를 함께 바꾸고 scalar gold-vs-rest logodds만 사용한다는 점이다. 후속 후보는 같은 query, 같은 masked-token 수, 같은 response span을 유지하고 context 위치만 one-for-one으로 바꾼 paired states를 사용한다.

Projection 또는 block \(u\)의 calibration 입력을 \(X_u^-,X_u^+\)로 두고 dense target을 \(Y_u^\pm=W_uX_u^\pm\)라 하면, 한 번의 static mask를 다음 augmented reconstruction에 적용한다.

\[
\mathcal L_u(M)=
\|Y_u^--(MW_u)X_u^-\|_F^2+
\|Y_u^+-(MW_u)X_u^+\|_F^2+
\|\Delta Y_u- (MW_u)\Delta X_u\|_F^2,
\]

\[
\Delta X_u=X_u^+-X_u^-,\qquad
\Delta Y_u=Y_u^+-Y_u^-.
\]

세 항은 각각 endpoint와 paired response를 나타내지만, arbitrary A/C coefficient를 두지 않는다. 각 항을 dense target energy로 먼저 normalize하거나, 세 target block의 Frobenius energy를 동일하게 맞춘 뒤 coefficient를 1로 고정한다. 실제 allocation은 기존 Wanda row ranking을 고정하고, \(\mathcal L_u\)의 48/52% finite marginal 또는 prespecified exact-budget exchange로 rate를 정한다. Paired state를 섞은 shuffled control은 endpoint 항은 보존하고 response 항의 추가 정보를 제거한다.

이 방향은 현재 A+C를 완전히 버리는 새 closed-loop 알고리즘이 아니라, DLM의 state-to-state relation을 shared static mask에 직접 넣는 작고 검증 가능한 확장이다. 단, relational representation matching, ALS의 inter-layer relation, NeuronAl의 activation alignment, 그리고 diffusion pruning의 sensitivity matching이 선행되어 있으므로 “response를 보존한다”는 문구만으로 novelty를 주장할 수 없다. [2ndMatch](https://arxiv.org/html/2506.05398)는 pruned diffusion model이 dense model의 perturbation sensitivity를 보존하도록 \(J^\top J\)를 맞추므로, 여기서의 discrete context-pair mask response는 continuous Jacobian matching과 무엇이 다른지 명시해야 한다.

필수 control은 ordinary pooled reconstruction, full-vocabulary KL, A-only/A+C, pair-shuffled, same-mask-count random pair다. 독립 문서에서 paired metric이 endpoint/KL보다 exchange utility를 예측하는지 먼저 확인하는 것이 좋지만, 이는 후속 downstream을 막는 hard gate가 아니다. 최종 NELBO/GSM8K 결과와 함께 판단한다. 계산은 response rollout보다 낮고 current A+C calibration을 재사용할 수 있지만, paired input의 선택·energy normalization이 새 hyperparameter가 되지 않도록 사전에 고정해야 한다. Mask-rate curvature는 이 후보의 별도 메인 objective가 아니라, matched-rate pair가 실제로 mask-density만 측정하는지 확인하는 negative control로 둔다.

### 3. Jointly sparse native exchange utility — 고위험·고가치 후보

기존 local reconstruction은 jointly sparse composition을 충분히 반영하지 못한다. D31에서 native feature direction과 receiver-token representation 정렬이 중요했고, dense-background perturbation과 sparse-background bundle의 부호가 달라질 수 있었다. 이는 실제 sparse model에서 직접 exchange를 평가해야 한다는 근거다.

Uniform50을 시작점으로 donor projection (a)에서 (q)개 row-quanta를 복원하고 receiver (b)에서 같은 수를 제거하는 exact-budget exchange (E_{a\to b})를 만든다. 현재 jointly sparse model과 동일한 native masked/unmasked state bank에서

\[
U_{a\to b}=L(S\odot E_{a\to b})-L(S)
\]

를 full-vocabulary KL 또는 NELBO로 직접 측정한다. (U)는 독립 unit cost의 합으로 근사하지 않고, exchange bundle 전체를 평가한다. 후보는 prespecified same-type/cross-depth/count-matched strata에서 제한하고, low-cost structural proposal만 shortlist한 뒤 actual joint forward로 검증한다.

필수 control은 pooled full-vocabulary KL, random exchange, donor/receiver permutation, same-count/same-shape exchange, held-out article NELBO다. 이 방향은 sparse interaction을 실제로 측정한다는 장점이 있지만 EvoPress의 global candidate search, BESA의 learned allocation, Lua-LLM의 end-to-end allocation과 겹친다. 따라서 novelty는 “검색”이 아니라 `DLM masked-state native exchange utility가 기존 full-vocabulary fidelity보다 추가 예측력을 가지는가`에 한정해야 한다.

비용은 높다. 224개 전체 pair search는 부적절하고, 처음에는 후보 수와 state 수를 고정한 exchange diagnostic만 가능하다. utility가 full-vocabulary KL과 구별되지 않으면 방법론 주장의 우선순위를 낮추고, independent NELBO로 이어지는지 확인되지 않은 단계에서는 diagnostic으로만 남긴다.

## Commitment-value 제안에 대한 독립적 재평가

frozen reference continuation으로 irreversible commit의 후속 손상을 평가하는 발상은 DLM과 잘 맞지만, 현재 상태에서 central novelty로 삼기에는 선행과 비용 문제가 크다.

- [FAIR-Calib](https://arxiv.org/html/2606.06547v2)는 DLM의 irreversible write frontier, fragile commit, post-commit amplification을 직접 측정하고 frontier-hit/reliability prior와 teacher-forced hidden-state calibration을 제안했다. static pruning으로 옮기는 것만으로는 개념 novelty가 아니다.
- [OPTD](https://arxiv.org/html/2608.02942)는 student on-policy partial state에서 frozen teacher의 future commitment outcome을 평가한다. candidate state occupancy와 future consequence를 쓰는 구조가 이미 존재한다.
- [Order-Token Search](https://arxiv.org/html/2601.20339)는 generation order와 token value를 함께 탐색하고 likelihood로 partial trajectory를 prune한다. action order/token space를 pruning novelty로 제시할 수 없다.
- Cost-to-go 자체는 [AggreVaTe](https://arxiv.org/abs/1406.5979)의 선행 원리다.

따라서 commitment-value가 기여가 되려면 static weight allocation에서만 발생하는 추가 현상을 보여야 한다. 예를 들면 exact-budget weight exchange가 commit action을 바꾸고, 그 action difference의 frozen suffix value가 full-vocabulary KL보다 independently held-out sparse NELBO를 더 잘 예측한다는 식이다. 이 연결은 아직 가설이다.

현재 target mismatch도 크다. frozen dense continuation은 candidate가 이후 sparse state를 방문하는 분포를 반영하지 않고, 짧은 expected Hamming/suffix loss는 NELBO나 GSM8K correctness가 아니다. candidate-own continuation을 쓰면 실제 오류를 포착하지만 candidate마다 (S\times h) forward가 필요하고 fitness가 candidate-dependent가 된다. 224 projection 교환 후보를 탐색하면 기존 A+C probe보다 비용이 급증한다. historical Uniform62 대 AC61과 중단된 full A/AC 결과를 고려하면, 이 비용을 정당화할 empirical need가 아직 없다.

결정: commitment-value는 main method가 아니라 후속 diagnostic 또는 강한 functional control로 보류한다. 먼저 동일 budget의 full-vocabulary KL exchange와 위 세 static direction 중 하나를 비교해야 한다.

## 검증에 필요한 최소 기준

1. **Ranking family 고정:** historical cached Uniform62와 native sparse-prefix Uniform54를 별도 controlled family로 유지한다. 한 family의 개선을 전체 DLM 최고점으로 부르지 않는다.
2. **Objective control:** Uniform, existing Wanda, full-vocabulary KL, 해당 후보 metric을 같은 exact-budget allocator와 같은 calibration compute로 비교한다. A+C는 개발 비교군이지 성공 증거가 아니다.
3. **State/document split:** metric 계산 문서와 utility/NELBO 평가 문서를 분리한다. 16 article 개발 subset을 독립 test로 부르지 않는다.
4. **Incremental information test:** 후보 metric이 Wanda/pooled reconstruction/full-KL보다 exchange utility를 예측하는지 먼저 보는 것이 좋다. 예측력이 없으면 downstream 우선순위를 낮추되, 최종 판단은 독립 NELBO/GSM8K 결과와 함께 한다.
5. **Claim scope:** LLaDA Base의 고정 confidence decoder와 50% static unstructured allocation에서의 개선만 우선 주장한다. AR 우위, 모든 DLM, universal sparsity law는 별도 근거 없이는 주장하지 않는다.
6. **Role scope:** masked/unmasked는 계속 기록하고 필요한 경우 strata로 분석한다. 역할 분리 전체를 기각하지 않지만, 특정 max/mean aggregation을 필수 원리로 만들지 않는다.

## 최종 권고

현재 근거와 비용을 함께 보면 1번 **cross-state support coverage**와 2번 **matched-mask paired-response reconstruction**을 작고 통제 가능한 후보로 남기는 것이 가장 합리적이다. C1은 pooled control 대비 NELBO 방향성을 보였지만 Uniform 대비 우위가 없으므로 주 결론으로 승격하지 않는다. 2번은 A+C의 명시된 confound를 직접 고친다. 두 후보 모두 독립 split에서 재현되지 않으면 논문 후보의 우선순위를 낮춘다.

현재 A+C의 점수만으로 response 항을 승격하지 않는다. 먼저 pair-shuffled/full-KL controls에서 실제 추가 정보가 있는지 확인한다. 3번 **native exchange utility**는 role/propagation 증거가 독립 split에서 재현될 때만 고려한다. 기존 search/learned-allocation 선행과의 경계가 가장 어렵고 비용도 크므로 첫 method로 시작하지 않는다.

이 순서는 모든 아이디어를 결합하는 kitchen-sink 설계를 피하면서도, DLM-specific static allocation이 실제로 필요한지와 기존 allocator 재사용만으로 충분한지를 동시에 판별한다.

## Primary sources

- OWL: https://arxiv.org/html/2310.05175v3
- AlphaPruning: https://arxiv.org/html/2410.10912v1
- DLP: https://arxiv.org/html/2505.23807v1
- LSA: https://proceedings.iclr.cc/paper_files/paper/2026/file/7b805585c7e249c1f65737506d4fe1e4-Paper-Conference.pdf
- ALS: https://proceedings.neurips.cc/paper_files/paper/2024/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf
- ATP: https://arxiv.org/html/2502.14770
- MRP: https://arxiv.org/html/2503.18377v1
- DSA: https://proceedings.neurips.cc/paper_files/paper/2024/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf
- EvoPress: https://arxiv.org/html/2410.14649
- BESA: https://arxiv.org/html/2402.16880
- NeuronAl: https://arxiv.org/html/2411.07066
- 2ndMatch: https://arxiv.org/html/2506.05398
- Lua-LLM: https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf
- FAIR-Calib: https://arxiv.org/html/2606.06547v2
- OPTD: https://arxiv.org/html/2608.02942
- Order-Token Search: https://arxiv.org/html/2601.20339
- AggreVaTe: https://arxiv.org/abs/1406.5979
