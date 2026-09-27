# Role exchange 수정 재분석

이전에 확인한 데이터를 동일 설정으로 재측정했다. 새로운 독립 확증은 아니다.

| 모델 | 가중 MSE | Spearman | 부호 정확도 |
|---|---:|---:|---:|
| P0_structure | 3.978641e-05 | 0.1862 | 0.5466 |
| P1_pooled_dense | 3.659887e-05 | 0.2207 | 0.5637 |
| P2_role_dense | 3.658116e-05 | 0.2178 | 0.5620 |
| P3_role_dense_sparse | 3.616394e-05 | 0.2170 | 0.5612 |
| P4_role_energy | 3.581437e-05 | 0.2495 | 0.5667 |
| random_101 | 3.649847e-05 | 0.2185 | 0.5625 |
| random_202 | 3.647752e-05 | 0.2185 | 0.5624 |
| random_303 | 3.650854e-05 | 0.2185 | 0.5625 |
| E0_pooled_context | 3.681149e-05 | 0.2052 | 0.5586 |
| E1_role_contrast | 4.297562e-05 | 0.1512 | 0.5463 |
| E2_role_interactions | 5.748475e-05 | 0.0979 | 0.5346 |

## 수정된 측정 경로

{"median_abs_label_change": 0.0018438547849655151, "p95_abs_label_change": 0.007671569287776943, "sign_disagreement_fraction": 0.24561688311688312, "label_spearman": 0.6872432643730613}

## 사전 비교

- H1: inconclusive, MSE 감소 0.05%, CI [-3.9517531613822156e-07, 4.966982270855051e-07]
- H2: inconclusive, MSE 감소 1.14%, CI [-1.6185347297289304e-06, 2.0211685927291936e-07]
- random_control: inconclusive, MSE 감소 -0.24%, CI [-8.050052222881921e-08, 4.6594309135327906e-07]

## 동일 예산 bundle

{
  "token": {
    "unit": "exact-budget bundle/state",
    "both_improve": 3026,
    "kl_worsens": 1226,
    "fraction": 0.4051553205551884,
    "mean_kl_change_when_both_improve": -0.002457969058095739,
    "caution": "sum of local reconstruction changes on fixed A inputs; no universal safety conclusion"
  },
  "energy": {
    "unit": "exact-budget bundle/state",
    "both_improve": 60,
    "kl_worsens": 35,
    "fraction": 0.5833333333333334,
    "mean_kl_change_when_both_improve": -0.00012523395319779713,
    "caution": "sum of local reconstruction changes on fixed A inputs; no universal safety conclusion"
  }
}

유의한 개선을 확인하지 못한 경우 해당 예측기의 근거 부족으로 해석한다. 역할 분리 전체를 기각하지 않는다.
Bootstrap은 고정된 OOF 모델에 조건부이며, 학습 표본 변동 전체를 포함하지 않는다.
