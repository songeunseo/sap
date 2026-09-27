# Role Allocation Trade-off @65%

## 설정
기존 224×6×80 role 통계. 동일 Wanda row-floor 제거 예산. 두 목적은 projection별 normalized reconstruction의 합이다. GPU/downstream 미실행.
정확한 제거 수: 4536008704; global sparsity: 0.6499211238.

## 전체 데이터 결과
|할당|Masked 합|Unmasked 합|기존 Role 대비 M/U 변화|변경 projections|solver gap|
|---|---:|---:|---|---:|---:|
|current_role|11.310842|12.086182|+0.0000% / +0.0000%|0|0|
|masked|11.310842|12.086182|+0.0000% / +0.0000%|0|0.0235|
|unmasked|11.310842|12.086182|+0.0000% / +0.0000%|0|0.0203|
|local_max|11.310842|12.086182|+0.0000% / +0.0000%|0|0.0169|
|minimax|11.310842|12.086182|+0.0000% / +0.0000%|0|0.0185|

두 endpoint 사이 변경 projections: 0; nested mask XOR: 0.000000%.

## Sequence crossfit
각 fold는 7개 sequence로 최적화하고 제외된 1개 sequence의 proxy로 평가한다. 기준 greedy Role도 train에서 다시 만든다.
```json
{
  "masked": {
    "mean_relative_change": [
      0.0,
      0.0
    ],
    "descriptive_bootstrap_95_ci": [
      [
        0.0,
        0.0
      ],
      [
        0.0,
        0.0
      ]
    ],
    "both_improve_sequences": 0,
    "role_improve_sequences": [
      0,
      0
    ]
  },
  "unmasked": {
    "mean_relative_change": [
      0.0,
      0.0
    ],
    "descriptive_bootstrap_95_ci": [
      [
        0.0,
        0.0
      ],
      [
        0.0,
        0.0
      ]
    ],
    "both_improve_sequences": 0,
    "role_improve_sequences": [
      0,
      0
    ]
  },
  "minimax": {
    "mean_relative_change": [
      0.0,
      0.0
    ],
    "descriptive_bootstrap_95_ci": [
      [
        0.0,
        0.0
      ],
      [
        0.0,
        0.0
      ]
    ],
    "both_improve_sequences": 0,
    "role_improve_sequences": [
      0,
      0
    ]
  }
}
```

## 해석 한계 및 결정
- epsilon-constraint 11개 지점은 전체 discrete Pareto frontier가 아니다. Scalar endpoint는 ties 때문에 유일한 allocation 또는 strong Pareto point라 단정하지 않는다.
- 솔버 gap/status를 확인하고 최적성 인증과 feasible 개선을 구분한다.
- Crossfit bootstrap은 8개 fold 기술적 구간이다. Training overlap과 frozen 전체 calibration Wanda ranking 때문에 독립적인 새 데이터 검증은 아니다.
- 두 proxy의 동시 개선은 실제 KL/GSM8K 개선을 의미하지 않는다. Projection 간 가산성은 검증되지 않았다.
- 방법 선택 또는 downstream 실행 없이 allocation geometry 근거로 사용한다.
