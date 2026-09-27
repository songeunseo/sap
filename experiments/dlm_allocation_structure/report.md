# DLM Allocation 구조 분해 — CPU 진단

## 질문

Projection sparsity allocation의 유효한 설명 단위가 depth, type, depth+type, 개별 projection 중 무엇이며,
DLM corruption timestep이 reconstruction으로 설명되지 않는 구조적 취약성을 추가하는가?

## 고정 데이터와 범위

기존 LLaDA-8B-Base 224×80×6 single-projection Standard-Wanda curve만 재분석했다.
새 mask/model/GPU/downstream 평가는 만들지 않았다. 상태는 random-corruption DLM calibration states이며 실제 reverse trajectory가 아니다.

## Raw marginal costs

평균 curve 기준 음의 projection-increment는 11개,
하나 이상 음의 increment를 가진 projection은 10/224개다.
원시 분포와 type/layer-quartile breakdown은 `raw_marginal_summary.json`에 있다. 값을 보정하거나 envelope로 바꾸지 않았다.

## 정적 구조의 sequence 교차 검증

| 설명 수준 | 평균 validation additive KL damage ↓ |
|---|---:|
| Global | 0.391376 |
| Depth | 0.301096 |
| Type | 0.371811 |
| Depth+Type | 0.289171 |
| 개별 projection | 0.269333 |

개별 projection 설명이 Depth+Type보다 양방향 fold 모두 낮은 damage를 냈는가: **True**.
Sequence별 projection−Depth+Type damage 차이: mean -0.0198382, 95% CI [-0.0230253, -0.016966].
각 fold의 exact-budget allocation, level counts, reference mask XOR은 `static_structure.json`에 있다.

## Timestep 구조와 reconstruction 중복

Primary 변수는 marginal KL / (0.05×projection parameter 수)다. Train-timestep RMS로 정규화한 projection+timestep model에
depth×timestep/type×timestep 상호작용을 추가했을 때의
validation sequence MSE 차이(structured − static)다. 음수면 상태별 구조가 추가 설명력을 갖는다.

- Raw marginal cost: mean 0.000652867, 95% CI [-0.00220034, 0.00377565], 양방향 개선=False.
- Reconstruction 보정 residual: mean 0.0010269, 95% CI [-0.00186494, 0.00430091], 양방향 개선=False.
- Bootstrap 단위는 8개 held-out sequence이며 projection/state 수를 독립 표본으로 세지 않았다.

## 판정

사전 기준에 따른 **DLM 상태 조건부 구조: 미지지**.
이는 후속 진단 우선순위를 정하는 결과이며 pruning 방법 또는 downstream 성능의 성공 판정이 아니다.
기존 full GSM8K에서는 oracle Capacity 250/1319, Reconstruction 255/1319, oracle-derived EIS+type 263/1319로
세 비균일 allocation의 차이가 유의하지 않았다. 따라서 이 additive 구조 순위를 jointly sparse downstream 순위로 해석하지 않는다.
AR control이 없으므로 DLM 고유성은 주장하지 않는다. 기존 Probe16 FAIL 및 oracle 결과도 재분류하지 않는다.

## 다음 단계 제한

공식 EIS/OWL, actual trajectory, matched AR/DLM control 또는 full sparse downstream은 이 결과를 검토한 뒤 별도 사전 계획으로 진행한다.
