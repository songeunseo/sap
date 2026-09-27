# Sparse-context 역할별 정보 기여

모든 후보가 dense masked/unmasked 절대·상대 통계를 보존한다.
기존 final 재사용 탐색 분석. 역할 개입의 인과효과가 아니라 예측기 입력 ablation이다.

| 후보 | Bundle MSE | 방향 정확도 |
|---|---:|---:|
| dense_roles | 0.000108190126 | 0.5287 |
| plus_sparse_M | 0.000105872294 | 0.5260 |
| plus_sparse_U | 0.000106541954 | 0.5289 |
| plus_sparse_MU | 0.00010524761 | 0.5266 |

## 사전 지정 4비교

- plus_sparse_M_minus_dense_roles: MSE 감소 2.142%, 동시 조건부 CI [-7.29722090608145e-06, -1.2461966358602113e-07]
- plus_sparse_U_minus_dense_roles: MSE 감소 1.523%, 동시 조건부 CI [-3.942648226901325e-06, -1.0916322397496344e-07]
- plus_sparse_MU_minus_plus_sparse_M: MSE 감소 0.590%, 동시 조건부 CI [-1.663381773526649e-06, 1.5519577125635468e-07]
- plus_sparse_MU_minus_plus_sparse_U: MSE 감소 1.215%, 동시 조건부 CI [-4.4591064263452816e-06, -4.737958188976229e-09]

## 한계

- 기존 final을 반복 사용한 탐색 분석이며 fresh confirmatory evidence가 아니다.
- 신뢰구간은 고정된 OOF 모델에 조건부. 학습 데이터 재표집/재학습 불확실성 미포함.
- 입력 추가는 정보와 ridge의 정규화 효과를 함께 바꾼다. 역할의 인과적 중요도 또는 necessity로 해석하지 않는다.
- 어느 sparse 입력의 추가 이득이 약해도 dense 역할 분리는 유지한다.
- KL 예측 개선은 GSM8K 개선이나 올바른 max/mean/minimax의 증명이 아니다.
