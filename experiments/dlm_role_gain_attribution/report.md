# 기존 Role-Wanda는 왜 잘됐는가? — 저장 결과 원인 감사

## Hypothesis / Setup

Role 분리는 유지한다. 추가 성능 이득의 원인은 미확정이다. 기존 Aggregate/Role 65%와 동일 1319개 예측, 80-state role 통계만 재사용했다. 새 생성·GPU·mask 변경 없음.
source/receipt/model config/manifest/문항·prompt·target·protocol identity, strict EM 및 mini100의 full prefix 완전 일치를 검증했다. Mask payload는 다시 읽지 않았으며 XOR는 기존 nested Wanda 검증을 전제한다.

## Result: 정확도 및 출력 상태

| 범위 | Aggregate | Role | 차이 | paired McNemar p |
|---|---:|---:|---:|---:|
| full | 248/1319 | 268/1319 | +20 | 0.084981 |
| mini_first100 | 19/100 | 24/100 | +5 | 0.226562 |
| remaining1219_exploratory_not_new_holdout | 229/1219 | 244/1219 | +15 | 0.183657 |

나머지1219도 이미 관측된 결과다. mini와 full을 독립 재현 두 번으로 세지 않는다. 모든 추가 분해는 탐색적이다.
Full paired accuracy difference 95% bootstrap CI: [-0.001516300227445034, 0.03184230477634572]. 고정 mask/생성 run에서 문항만 재표집한 CI이며 calibration/seed 변동은 포함하지 않는다.

행=Aggregate, 열=Role. strict_invalid는 저장된 extractor가 `[invalid]`를 반환한 경우다. 미완성·반복·잘못된 추론으로 자동 해석하지 않는다.

| Aggregate → Role | correct | valid_wrong | strict_invalid |
|---|---:|---:|---:|
| correct | 197 | 38 | 13 |
| valid_wrong | 54 | 542 | 74 |
| strict_invalid | 17 | 85 | 299 |

정답 순증의 산술 분해: strict-invalid↔correct +4, valid-wrong↔correct +16. 이 전이 분해는 인과 mediation 분석이 아니다.

## Result: 정확한 예산 이동

양 모델 pruned=4,536,008,704/6,979,321,856 (64.99211238%). 변경 20/224개. 추가 pruning과 복원 각각 15,101,952 weights. Nested-mask XOR 30,203,904.

| Projection | Aggregate % | Role % | Δ pruned | Aggregate 지점에서 normalized error가 큰 role |
|---|---:|---:|---:|---|
| block_00.q_proj | 75 | 70 | -839,680 | unmasked |
| block_00.k_proj | 75 | 70 | -839,680 | unmasked |
| block_02.attn_out | 70 | 65 | -839,680 | masked |
| block_03.attn_out | 70 | 65 | -839,680 | masked |
| block_04.ff_out | 70 | 75 | +2,519,040 | unmasked |
| block_06.ff_proj | 65 | 70 | +2,519,040 | unmasked |
| block_08.up_proj | 55 | 60 | +2,519,040 | unmasked |
| block_10.up_proj | 55 | 60 | +2,519,040 | unmasked |
| block_11.q_proj | 70 | 65 | -839,680 | unmasked |
| block_15.ff_out | 70 | 65 | -2,514,944 | masked |
| block_16.attn_out | 50 | 55 | +835,584 | masked |
| block_17.attn_out | 50 | 55 | +835,584 | masked |
| block_17.q_proj | 60 | 55 | -839,680 | unmasked |
| block_23.ff_out | 65 | 60 | -2,519,040 | masked |
| block_24.attn_out | 50 | 55 | +835,584 | unmasked |
| block_26.attn_out | 55 | 50 | -835,584 | unmasked |
| block_26.ff_out | 70 | 65 | -2,514,944 | masked |
| block_26.k_proj | 70 | 65 | -839,680 | unmasked |
| block_29.k_proj | 65 | 60 | -839,680 | unmasked |
| block_31.up_proj | 70 | 75 | +2,519,040 | unmasked |

타입별 양수는 추가 pruning, 음수는 보호다. 이는 전체 Role-vs-Uniform 구조가 아니라 작은 Role-vs-Aggregate 차이다.

- attn_out: 6개 변경, Δpruned=-8,192
- ff_out: 4개 변경, Δpruned=-5,029,888
- ff_proj: 1개 변경, Δpruned=+2,519,040
- k_proj: 3개 변경, Δpruned=-2,519,040
- q_proj: 3개 변경, Δpruned=-2,519,040
- up_proj: 3개 변경, Δpruned=+7,557,120

## Interpretation: 아직 귀속할 수 없는 것

- 20개 allocation 변경이 동시에 적용된 두 모델의 출력만으로 각 projection 효과를 식별할 수 없다. 문항이1319개여도 intervention vector는 두 개뿐이다. 같은 allocation을 문항마다 복제해 회귀해도 모듈별 인과 효과는 식별되지 않는다.
- normalized error가 큰 role은 실제 기능적으로 더 중요한 role과 같지 않다. proxy 변화는 allocation 결정의 설명이지 GSM8K 개선의 원인 증명이 아니다.
- 기존 causal decomposition은 28개 single projection의 DLM KL, decision audit는 Role/Exact-Max/Minimax 교환의 KL을 측정했다. 여기 필요한 Aggregate→Role의 GSM8K 개입 효과를 직접 측정하지 않았다.
- parsing 전이가 정답 차이를 산술적으로 설명해도, 출력 형식만 바뀌었는지 reasoning·completion이 바뀌었는지는 별도 검증이 필요하다. 기존 strict 점수는 변경하지 않는다.

## Next Experiment — 설계만, 미실행

문항 정답을 보지 않고 count/canonical order로 변경집합을 6개의 exact-budget bundle로 분해했다. 이것은 유일한 원인 분해나 새 allocation 후보가 아니라 counterfactual 측정 단위다.

- B0: block_04.ff_out, block_23.ff_out; Δbudget=0
- B1: block_16.attn_out, block_26.attn_out; Δbudget=0
- B2: block_00.q_proj, block_00.k_proj, block_02.attn_out, block_06.ff_proj; Δbudget=0
- B3: block_03.attn_out, block_08.up_proj, block_11.q_proj, block_17.q_proj; Δbudget=0
- B4: block_10.up_proj, block_15.ff_out, block_17.attn_out, block_26.k_proj; Δbudget=0
- B5: block_24.attn_out, block_26.ff_out, block_29.k_proj, block_31.up_proj; Δbudget=0

후속 승인 시 각 bundle b에 대해 Aggregate+b(충분성)와 Role−b(필요성)를 둘 다 측정한다. 둘 다 원래 exact global budget을 유지하며 local Wanda masks는 기존 두 모델에서만 가져온다. 동일 문항/seed/5-shot/256-step/strict EM 유지. 기존 mini100 전체를 먼저 사용하고 role-only 정답들만 선별하지 않는다. 결과는 원인 진단이지 후보 selection이 아니다.
두 방향 효과가 다르면 context dependence/interaction의 증거다. full 효과를 주장하려면 전체1319에서도 별도 측정이 필요하다. bundle 내 개별 projection이나 role 경로의 원인으로 더 쪼개려면 후속 role-specific patching이 추가로 필요하며, 이전 KL proxy 최적화를 그대로 반복하지 않는다.

## Decision

역할 분리는 유지. 이번에는 기존 성공 결과의 설명 범위만 좁혔다. 새 pruning 방법, GPU 실행, full 평가를 시작하지 않았다.
