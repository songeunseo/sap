# 다음 분석: 역할을 보존한 proxy 단위와 전달 효과 분해

## 고정 전제와 범위

Masked/unmasked 통계를 각각 보존한다. 역할 분리 여부를 다시 kill gate로 삼지 않는다. 이는 연구 설계 선택이며 모든 역할 기반 구현의 우위를 뜻하지 않는다. 기존 Wanda ranking, Role-65 기준 모델과 교환 목록, 데이터와 예산은 유지한다. 기존 결과를 본 뒤 설계한 탐색 분석이며 새 확증으로 부르지 않는다. 현재 문서는 계획이며 새 계산/실험은 실행하지 않았다.

## 코드 감사에서 확인된 점

`dlm_role_exchange_prediction/collect.py:role_feature`에서 역할 r의 통계는

- token: ΔE_r = Σ_r (||W_candidate x − W_dense x||² − ||W_base x − W_dense x||²) / n_r
- energy: ΔR_r = Σ_r (동일 오차 변화) / Σ_r ||W_dense x||²

따라서 ΔR_r = ΔE_r / (Σ_r ||W_dense x||² / n_r). 에너지 통계는 새로운 perturbation 방향 정보가 아니라 역할별 출력 에너지로 나눈 상대 오차다. Sparse-context에서도 기준 Linear 출력은 dense weight에 sparse 입력을 넣은 출력이다. Dense full-model activation과 혼동하지 않는다.

P4는 P3의 raw 통계를 유지한 채 energy 통계를 추가한다. 따라서 이전 P4 vs pooled 개선은 정규화만의 효과나 역할 분리만의 효과로 귀속할 수 없다.

## 질문 1: 절대 오차인가 상대 오차인가?

역할 분리는 모두 유지하고 다음 2×3 설계로 분리한다.

| 입력 context | 역할별 raw만 | 역할별 relative만 | 역할별 raw+relative |
|---|---|---|---|
| Dense | D-raw | D-relative | D-both |
| Dense+Sparse | DS-raw | DS-relative | DS-both |

동일 structure controls와 학습-only 표준화, 기존 mean-loss ridge alpha=1을 사용한다. 단일 pruning 변경으로 학습하되 주 진단 단위는 실제 동일 예산 bundle이다. 사전 지정 비교: D-relative vs D-raw, DS-relative vs DS-raw, DS-both vs DS-raw, DS-both vs D-both. 전체 비교를 함께 공개하고 동시 CI를 보고한다. 기존 final은 탐색용 재사용으로 표시한다.

오차 MSE 외에 rank, 방향 정확도, 무변경/상수 예측 기준, 문서·layer·type·timestep별 일관성을 함께 보고한다. 별도 sparsity allocation이나 GSM8K는 아직 만들지 않는다.

## 질문 2: 정규화가 어떤 순위를 바꾸는가?

원본에 저장된 분모/activation 에너지가 있으면 직접 사용한다. ΔE/ΔR 역산은 0 및 거의 0에서 불안정하므로 주 분석에 쓰지 않는다. 분모가 없으면 이를 명시하고 안전한 추가 수집 계획으로 분리한다.

역할별로 raw/relative 순위 변화, 변화가 큰 projection의 type/layer 분포, 개발 문서별 재현성을 분석한다. 대형 projection이나 큰 출력 스케일이 raw 통계를 지배하는지 확인하되 association을 인과로 해석하지 않는다.

## 질문 3: 단일 변경 예측의 한계인가, 동시 변경 상호작용인가?

저장된 batch1 full-forward 측정으로 실제 bundle ΔKL과 구성 single ΔKL 합을 비교한다. 이때 합은 일반적 상한이나 달성 가능한 모델이 아니라 가산성 진단 기준이다. Single 예측 잔차와 비가산 잔차의 공분산까지 유지한다. 두 RMSE를 독립적인 오차 기여율처럼 해석하지 않는다.

## 해석과 다음 결정

- 상대 오차가 일관되게 낫다면: role별 상대 손상 단위를 채택할 근거로 사용하되 max를 정당화하지 않는다.
- Sparse context가 추가로 유용하면: 기준 sparse 모델에 조건부인 proxy임을 명시하고 필요한 계산 비용을 측정한다.
- 비가산성이 크면: scalar aggregation 변경보다 joint interaction 모델링이 필요하다는 가설로 연결한다. 역할 분리 자체는 유지한다.
- 어떤 예측기가 KL에 유리해도 GSM8K 우위를 뜻하지 않는다. Global minimax의 KL/downstream 불일치를 유지해서 기록한다.

이 분석만으로 max/mean/minimax 중 하나를 선택하지 않는다. 먼저 손상 단위와 예측 한계를 좁힌 뒤, 필요한 목적함수 가정을 명시하여 allocation을 설계한다.
