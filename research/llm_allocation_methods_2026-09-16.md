# LLM sparsity target allocation 방법론 조사
작성: 2026-09-16. 문헌 조사이며 신규 실험 실행은 없음.
요청: 기존 allocation 논문이 어떤 관찰을 어떤 측정량·배분 규칙으로 연결했는지 확인.
사전 연구 상태: Obsidian Research/DLM-Pruning/Research-State.md 확인.

## 공통 문제와 읽는 기준
단위 u의 제거 비율 s_u, 파라미터 수 N_u에 대해 전체 예산은 sum(N_u s_u)=S sum(N_u). Transformer layer/block, attention/FFN block, projection, output row는 서로 다른 allocation 단위다. 각 논문의 layer 용어를 그대로 동일시하면 안 된다.
이 표는 논문이 제안한 방법의 비교이며 동일 조건 성능 순위가 아니다. 저자들의 중요도/중복성 해석은 DLM에서 검증된 사실이 아니다.

## 통계·구조 해석에서 배분을 만드는 방법
| 논문 | 출발 가정·관찰 | 실제 측정량과 배분 | 핵심 기여 |
|---|---|---|---|
| [OWL, ICML 2024](https://arxiv.org/html/2310.05175v3) | activation-aware outlier 구조 보존이 중요하고 무제한 global pruning은 불안정 | A=abs(W)·input-column L2 norm; D=비율(A>M·mean A). D가 큰 transformer block은 덜 제거. 목표 S 주변 제한된 폭으로 매핑하고 평균 예산 유지 | outlier 비율의 layer 중요도 해석 + 과격한 비균일 배분 억제 |
| [AlphaPruning, NeurIPS 2024](https://arxiv.org/html/2410.10912v1) | HT-SR 관점에서 spectral heavy tail은 학습 품질과 관련 | WᵀW 고유값 tail의 power-law exponent α 추정. 작은 α/heavier tail을 더 중요한 층으로 취급해 보호; 큰 α는 더 제거. 제한 폭·전체 예산 보정 | weight 크기보다 spectral shape를 학습 상태의 proxy로 해석. Allocation 자체에는 activation 불필요 |
| [DLP, 2025](https://arxiv.org/html/2505.23807v1) | 고정 outlier threshold의 모델 의존성; 중심부가 redundancy를 표현한다는 해석 | 여러 reducer 비교 후 논문은 Wanda-score median 사용. 큰 median→큰 redundancy→더 pruning, importance 역변환·범위 조정 | threshold 의존성 감소와 redundancy reducer의 실증 선택 |
| [LSA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b805585c7e249c1f65737506d4fe1e4-Paper-Conference.pdf) | 개별 weight score만 집계하면 동시 제거의 cross term을 놓침 | H=XᵀX를 사용해 group 단위 greedy 최소 linear reconstruction error E를 계산(50% 가상 제거). **큰 E→더 pruning**. I_l=1−E_l/sum E, 범위 재조정 후 sparsity 역매핑. projection별 파라미터 수 보정도 제안 | covariance를 포함한 redundancy 측정과 finer granularity |
| [ALS, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c573258c38d0a3919d8c1364053c45df-Paper-Conference.pdf) | 다른 층과 겹치는 representation은 중복성이 높다는 해석 | activation 간 normalized feature-inner-product RM 행렬; mutual-information 관점으로 유도. 다른 층과의 RM 합→importance, depth를 반영한 계수 및 크기 제약으로 linear programming. inter/intra-layer allocation | 독립 층 통계에서 층 사이 관계로 확장하고 예산 문제를 LP로 정식화 |
| [ATP, ICML 2025](https://arxiv.org/html/2502.14770) | 앞층 pruning error가 뒤로 전파·누적된다는 분석 | s_i=S−β(L−1)/2+β(i−1). 앞은 덜, 뒤는 더 pruning. 작은 grid에서 β 하나를 PPL로 선택 | 오차 전파 해석으로 고차원 배분을 1차원 schedule family로 제한 |
| [MRP, 2025 preprint](https://arxiv.org/html/2503.18377v1) | pruning metric에 따라 sensitivity가 바뀌며 sparse model의 redundancy 균형이 중요 | 낮은 uniform sparsity에서 시작. 현재 sparse model의 non-outlier ratio 재측정→가장 redundant한 block을 조금 더 제거→반복, 목표 예산에서 종료 | dense 통계의 일회성 배분 대신 pruning 이후 상태를 반복 반영 |

### 해석 시 주의할 점
- OWL의 post-pruning outlier 분석은 pre-pruning 평균을 threshold로 고정하지만 post-pruning score를 평가한다. 우리의 “원래 dense tail 좌표가 모두 살아남는다”는 사실만으로 OWL의 outlier-structure 가설 전체를 반증하지 않는다.
- LSA §4.2는 작은 E에서 상대적 outlier가 생기기 쉽고 큰 E는 더 균일한 error를 나타낸다고 해석한다. **E 최소화 문제를 풀어 통계를 구하는 것과, 그 E를 최종 allocation cost로 최소화하는 것은 다르다.** 저자 해석이며 보편적 implication은 아니다.
- ALS의 실제 RM은 feature correlation 형태다. 정확한 고차원 mutual information을 직접 계산했다고 표현하지 않는다.
- ATP의 등차수열은 논문의 근사적 오차 전파 분석과 경험적 검증으로 정당화한 설계다. 모든 Transformer에서 뒷층이 반드시 덜 중요하다는 정리는 아니다.
- MRP의 non-outlier 통계도 중복성 proxy다. “중복성 균등화”가 실제 task loss의 최적조건이라는 보장은 없다.

## 배분 함수·배분 변수를 직접 탐색하거나 학습하는 방법
| 논문 | 최적화 대상 | 평가 신호·절차 | 기존 통계 방법과의 차이 |
|---|---|---|---|
| [DSA, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf) | elementwise importance→sparsity를 만드는 symbolic operation graph | preprocess/reduce/transform/postprocess 연산 조합을 evolutionary search. 후보로 실제 pruning 후 validation PPL 등 평가 | 특정 log-variance 하나가 논문의 기여가 아니라 allocation 함수 발견 체계가 기여 |
| [EvoPress, ICML 2025](https://arxiv.org/html/2410.14649) | unit별 compression-level vector | 여러 압축 수준을 미리 만들고, 예산을 유지하는 mutation 및 elitist selection. dense/sparse output KL로 후보를 단계별 평가 | local surrogate의 합보다 모델 전체 동작을 직접 평가. pruning·quantization·depth compression 적용 |
| [BESA, 2024](https://arxiv.org/html/2402.16880) | transformer block 내부 projection/row sparsity | Wanda 순위 고정, 후보 sparsity p_d의 조합 α=Σβ_d p_d를 학습. differentiable mask/STE, block reconstruction loss+sparsity penalty, 원래 W는 고정 | 소수 allocation parameter를 학습. block 단위 목표 예산을 둔 순차 처리; row 버전 및 경량 layer 버전 |
| [NeuronAl](https://arxiv.org/html/2411.07066) | block schedule의 폭, 다음으로 row allocation의 폭 | dense/sparse 각 projection의 **입력 activation** 정규화 차이를 alignment loss로 사용. block은 depth에 따른 linear schedule 후보를 만들고 alignment가 가장 좋은 폭 선택; row 단계로 보정 | retraining/gradient 없이 작은 후보군을 평가. 각 block을 자유롭게 독립 최적화하는 방식은 아님 |
| [Lua-LLM, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/b2c39fe6ce838440faf03a0f780e7a63-Paper-Conference.pdf) | 모든 matrix의 row별 pruning threshold | Wanda 순위를 uniformized score로 바꾸고 sigmoid soft top-K로 threshold 미분. 원래 W를 고정하고 end-to-end 성능 loss+sparsity regularizer로 학습 | block local reconstruction을 넘어 inter/intra-layer 예산을 함께 최적화. row 균등 quota도 학습 대상으로 확장 |

## 최근 preprint: 관련 novelty를 판단할 때 구분해 볼 것
- [PALS, 2026-07 preprint](https://arxiv.org/html/2607.07557v1): activation 절댓값 99th percentile을 layer importance로 사용, 표준화 및 제한된 sparsity 조정. 최대값보다 robust한 outlier 척도라는 논리. percentile 변경만으로 독립적인 DLM 기여가 된다고 볼 근거는 없으며, 원문 ablation도 99th percentile의 일관된 최적성을 보이지 않는다.
- [Beyond Layer Importance in Layer-wise Sparsity: An Inter-Layer Perturbation-Absorption Perspective, 2026-06 preprint](https://arxiv.org/html/2606.15161): Gaussian perturbation 주입 후 상대 hidden-state drift R_k=norm(h'_k−h_k)/norm(h_k)를 추적. r_l=E[R_L/R_l]로 증폭/흡수 측정. u_l=clip(log r_l,−log3,+log3), Δ_l=−(u_l−mean u), 기존 OWL/Alpha 배분에 크기가 제한된 Δ 보정을 더하고 예산 복원. 증폭되는 곳을 보호하고 흡수되는 곳으로 pruning 이동. standalone 모든 배분 해법이 아니라 기존 배분의 correction이며 DLM 검증은 아님.

## 방법 설계 관점에서의 해석
여기부터는 문헌들의 공통 구조에 대한 우리의 해석이다.
1. 측정량 제안: OWL/Alpha/DLP/LSA는 “무엇을 중요도 또는 redundancy라고 볼 것인가”를 선택한다. 통계→해석→배분 방향→폭/예산 제약이 한 묶음이다.
2. 관계 제안: ALS는 다른 층과의 representation 중복을, absorption 논문은 뒤층에서 오차가 살아남는 비율을 측정한다. marginal distribution만으로 답하기 어려운 질문을 측정 대상으로 바꾼다.
3. 배분 원칙 제안: ATP는 depth schedule, MRP는 sparse-state redundancy 균등화를 채택한다. 최종 알고리즘은 간단해도 어떤 상태를 만들려는지가 명확하다.
4. 최적화 방식 제안: DSA/EvoPress/BESA/NeuronAl/Lua는 함수, vector, threshold, schedule 폭 등 최적화 대상을 구조화해 비용을 줄인다. 새로운 손수 만든 scalar proxy가 항상 필수는 아니다.

현재 DLM 작업에 대한 결정:
- 기존 “tail은 이미 보호된다 / non-tail 손상이 중요할 수 있다”는 진단만으로 새 allocation 방법이 정해지지 않는다는 사용자 지적을 반영한다. 아직 설계 대상과 배분 원리가 빠져 있다.
- log-variance, norm 정규화, generic error absorption만을 새 DLM novelty로 제시하지 않는다. 해당 통계/개념에는 선행 방법이 있고, 우리 과거 gain/reconstruction 실험도 단순 성공을 뒷받침하지 않는다.
- 문헌에서 배울 것은 식을 복잡하게 만드는 것이 아니라, **무엇을 보존·균등화·최소화하면 좋은 sparse model이 되는가를 정하고 이를 측정과 예산 규칙으로 연결하는 과정**이다.
- 이 조사 자체는 어느 가정이 DLM에서 맞는지 검증하지 않는다. 새 실험이나 후보 method 채택은 수행하지 않았다.

## 우리 baseline과 논문 사이 provenance
- DLP 논문 reducer는 median. 현재 baseline은 기록된 pinned public get_dlp_ratios mean 경로이므로 논문 설명과 baseline 구현을 구분해야 한다.
- DSA의 현재 bounded DLM controller는 공식 전체 search 재현이 아니다.
- EvoPress의 FastOBC weight update가 포함된 결과는 allocation-only 차이로 해석하지 않는다.
- 기존 LSA layer/projection 결과는 mapping 및 폭 설정도 달라 순수 granularity ablation이 아니다.
- 현재 비교 지표는 frozen WikiText validation NELBO/PPL-bound 추정치이며 AR 논문의 PPL 수치와 직접 비교하지 않는다.

## 관련 연구 기록
- [[Research/DLM-Pruning/Research-State]]
- [[Experiments/2026-09-16-Distribution-Structure-Hypothesis-Audit]]
- [[Experiments/2026-09-16-PPL50-Allocation-Mechanism-Review]]
- Local previous analysis: experiments/dlm_distribution_hypotheses/report.md

