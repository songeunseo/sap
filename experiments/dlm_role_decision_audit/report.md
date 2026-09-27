# Role Allocation Decision Audit

## 결과

이 문서는 downstream 정답률을 사용하지 않고, frozen bundle 교환의 held-out DLM 인과 효과를 측정한다.

### Full allocation held-out DLM

| 방법 | Mean KL | Median KL | Top-1 |
|---|---:|---:|---:|
| A | 0.660051 | 0.442351 | 0.6156 |
| B | 0.666506 | 0.435958 | 0.6065 |
| C | 0.645010 | 0.419768 | 0.6142 |

Paired comparison:

- B − A mean KL: **+0.006455** (sequence-cluster bootstrap 95% CI `[+0.000980, +0.011335]`), 14/40 states improved, exact 8-sequence sign-flip p=0.0703.
- C − A mean KL: **−0.015040** (95% CI `[−0.025965, −0.003932]`), 26/40 states improved, 6/8 sequence means and 3/5 timestep means improved, exact sign-flip p=0.0469.
- 위 두 full-allocation 비교까지 하나의 family로 보정하면 C의 Holm p는 0.09375이다. 따라서 8개 sequence만으로 family-wise 유의성을 확정했다고 표현하지 않는다.

### Frozen bundle 교환

| 배경 | Bundle | ΔKL | 95% CI | Holm p | interaction |
|---|---|---:|---:|---:|---:|
| dense | B1_proxy_better_kl_worse | 0.00055881 | [0.000138105, 0.0010242] | 0.3281 | 0.000197784 |
| dense | B2_proxy_better_kl_better | -0.000404014 | [-0.00115835, 0.000489436] | 1 | 9.27935e-06 |
| dense | B3_kl_neutral | 2.71469e-05 | [-0.000407227, 0.0004] | 1 | 0.000251083 |
| dense | C1_proxy_better_kl_worse | 0.00129416 | [0.000699999, 0.00182584] | 0.1406 | 0.000208419 |
| dense | C2_proxy_better_kl_better | -0.0229146 | [-0.0250642, -0.0207249] | 0.09375 | -0.000287055 |
| dense | C3_kl_neutral | 0.000832153 | [0.000478096, 0.00113586] | 0.1406 | 7.84207e-05 |
| role_sparse | B1_proxy_better_kl_worse | 0.00350659 | [0.000326513, 0.00718646] | 0.5859 | -0.000235121 |
| role_sparse | B2_proxy_better_kl_better | 0.00596291 | [0.00286321, 0.00929294] | 0.09375 | -0.0001708 |
| role_sparse | B3_kl_neutral | 0.000135451 | [-0.00389014, 0.00401702] | 1 | -0.00168159 |
| role_sparse | C1_proxy_better_kl_worse | 0.00555545 | [0.00295866, 0.00807802] | 0.1406 | -0.000920341 |
| role_sparse | C2_proxy_better_kl_better | -0.027425 | [-0.0342076, -0.0209837] | 0.09375 | 0.000599423 |
| role_sparse | C3_kl_neutral | 0.00116803 | [-0.0010677, 0.00366575] | 1 | -0.00064946 |

## 해석 원칙

- ΔKL<0이면 A에서 해당 B/C bundle로 바꾼 것이 개선이다.
- dense와 Role-sparse 배경 차이는 composition/context dependence를 뜻한다.
- interaction이 0에서 벗어나면 single-projection damage의 단순 합으로 bundle 효과를 설명할 수 없다.
- 이 감사는 allocation을 새로 고르지 않으며 GSM8K를 사용하지 않는다.

## 관측된 메커니즘

1. **Exact local-Max(B)는 Role greedy(A)를 개선하지 않았다.** Max 목적을 더 정확히 최적화했지만 held-out DLM mean KL과 top-1 모두 악화됐다. 따라서 `max` 자체를 이론적으로 우월한 aggregation으로 정당화할 증거는 없다.
2. **Global Minimax(C)의 mini 실패는 DLM fidelity 실패가 아니다.** C는 held-out DLM KL을 A보다 0.01504 낮췄지만, 이미 동결된 GSM8K mini 결과는 A 24/100 대 C 16/100이었다. 이는 reconstruction/DLM-fidelity 목적과 downstream mathematical capability 사이의 objective mismatch를 직접 드러낸다.
3. **Dense-background 판단은 jointly sparse 효과를 완전히 예측하지 못한다.** B2는 dense 배경에서 ΔKL −0.000404였지만 Role-sparse 배경에서 +0.005963으로 방향이 뒤집혔다. C2는 양쪽에서 크게 개선됐으며 full C 개선의 중요한 구성요소였다.
4. **Masked/unmasked causal route는 실제로 다른 효과를 보인다.** 예를 들어 sparse C2의 masked-only ΔKL은 −0.030641, unmasked-only는 +0.003986이었다. 반대로 sparse B1/C1의 악화는 주로 unmasked-only route에서 나타났다. 이는 role 분리의 분석적 가치를 지지하지만 특정 aggregation rule을 자동으로 정당화하지는 않는다.
5. **Bundle 비가산성은 존재하지만 모든 실패의 단일 원인은 아니다.** 일부 bundle interaction은 작았고, B3 sparse처럼 큰 경우도 있었다(−0.001682). 더 강한 현상은 dense→sparse 배경 변화에 따른 effect-size/방향 변화였다.

## Decision

- `max(E_M,E_U)`를 더 정교하게 푸는 branch는 중단한다.
- masked/unmasked 분리는 유지하되, reconstruction scalar의 고정 aggregation으로 곧바로 allocation을 정하는 방식은 아직 메인 방법으로 채택하지 않는다.
- 다음 method-development에서는 **jointly sparse background에서 측정 가능한 저비용 projection exchange utility**와 **task-relevant capability를 보존하는 proxy**를 구분해 설계해야 한다.
- C의 DLM KL 개선을 근거로 GSM8K 결과를 사후적으로 무효화하지 않는다. 두 결과의 mismatch 자체가 다음 분석 대상이다.
