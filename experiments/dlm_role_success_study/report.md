# 기존 Role 성공의 문항별 재분석

## Objective / Hypothesis

Role이 새로 맞힌 문항의 회복/소실과 공통정답 손상을 분리한다. 특정bundle이 성공원인이라는 가정 없이 joint outcome과 설계 식별성을 분석한다.

## Setup

기존14모델×동일100문항. 새GPU/생성/할당변경없음. 개별총점은 이미 관측한 상태의 탐색분석이며 endpoint로선택한집단은 설명용이다. Source/receipt/identity/strictEM 검사 및입력불변확인.

## Result: 실제로 어떤 문제를 잃었나

기준집단: {'role_only': 8, 'aggregate_only': 3, 'both_correct': 16, 'both_wrong': 73}

아래 두 lost 열은 Role에서해당묶음을되돌렸을때 소실된정답수다. 총점순변화와달리 다른문항에서얻은정답은아직차감하지않은gross값이다.

| Bundle | add로 Role-only8 중 회복 | revert로 Role-only8 중 소실 | revert로 공통정답16 중 소실 | revert로 기존 Role오답 중 회복 | Role−revert 순효과 |
|---|---:|---:|---:|---:|---:|
| B0 | 2 | 6 | 1 | 3 | +4 |
| B1 | 2 | 3 | 4 | 2 | +5 |
| B2 | 4 | 4 | 0 | 2 | +2 |
| B3 | 1 | 4 | 0 | 4 | +0 |
| B4 | 1 | 5 | 1 | 2 | +4 |
| B5 | 1 | 2 | 2 | 4 | +0 |

14개설정모두정답 10, 모두오답 61, 정오가바뀐문항 29.
Role-only8중단일add어느것에서도회복안된문항 3; 하나이상revert에서소실된문항 7.
공통정답중hybrid에서하나라도틀린문항 6; 공통오답중hybrid에서하나라도맞힌문항 12.

| Role-only 문항ID | 6개add 중 정답수 | 6개revert 중 정답수 |
|---|---:|---:|
| 16 | 0 | 5 |
| 19 | 0 | 0 |
| 48 | 2 | 2 |
| 55 | 3 | 3 |
| 58 | 1 | 2 |
| 71 | 4 | 6 |
| 76 | 0 | 1 |
| 96 | 1 | 5 |

## Result: 전체변경 비가산성

100문항의14개결과를함께bootstrap했다. CI는개별/탐색용,model/calibration/seed불확실성은포함하지않는다.

- endpoint_role_minus_aggregate: +5.000pp, CI [-1.0, 12.0]
- endpoint_gain_minus_sum_add_benefits: +3.000pp, CI [-19.0, 26.0]
- sum_reverse_benefits_minus_endpoint_gain: +10.000pp, CI [-12.0, 33.0]
- mean_background_benefit_difference: +2.167pp, CI [-4.5, 8.667]

accuracy는이산결과라additive한연속latent score를threshold해도비가산성이생길수있다. 비가산성은직접적인neural-path상호작용입증이아니다. 여러bundle에서같은문항을잃으면효과합에는중복계산되므로합을전체gain의고유기여분해로쓰지않는다.

## Result: 현재설계에서 식별할 수 있는 것

{'observed_coalitions': 14, 'possible_coalitions': 64, 'additive_columns': 7, 'additive_rank': 7, 'pairwise_columns': 22, 'pairwise_rank': 13, 'pairwise_nullity': 9, 'interpretation': 'Rank deficiency prevents unique pairwise attribution even with infinitely many questions at these same design rows.'}

같은14설정의문항만늘리면평균효과CI는줄일수있지만rankdeficiency는해결되지않는다. 특정pair상호작용/Shapley기여는추가가정이나새coalition측정없이식별불가.

## Result: 검출력 설계민감도

가정한참효과/discordance에대한별도simulation이다. 실제효과추정이나실패원인판정이아니다. Bonferroni수준은보수적인12비교기준이지실제Holm검출력이아니다.

| n | 가정discordance | 가정참차이 pp | 단일검정 검출률 | 보수적12비교 검출률 |
|---|---:|---:|---:|---:|
| 100 | 10% | 1 | 2.8% | 0.2% |
| 100 | 10% | 2 | 4.9% | 0.4% |
| 100 | 10% | 3 | 8.8% | 1.1% |
| 100 | 10% | 5 | 24.2% | 4.6% |
| 100 | 20% | 1 | 3.3% | 0.2% |
| 100 | 20% | 2 | 4.6% | 0.4% |
| 100 | 20% | 3 | 6.4% | 0.6% |
| 100 | 20% | 5 | 14.3% | 2.1% |
| 1319 | 10% | 1 | 18.2% | 3.6% |
| 1319 | 10% | 2 | 60.3% | 25.9% |
| 1319 | 10% | 3 | 92.6% | 70.2% |
| 1319 | 10% | 5 | 100.0% | 99.8% |
| 1319 | 20% | 1 | 11.7% | 1.7% |
| 1319 | 20% | 2 | 34.9% | 9.7% |
| 1319 | 20% | 3 | 66.2% | 31.1% |
| 1319 | 20% | 5 | 98.1% | 87.8% |

## Interpretation / Decision

이번분석은기존Role성공의증거를꾸미거나새할당을고르는작업이아니다. Role-only문항과공통문항손상을구분하여다음원인검증대상을정확히한다. 역할분리유지. neural기전은아직미확정.

## Next Experiment

단순히같은교환을더많이생성하는것과실제role경로를검증하는것을구분한다. 후자는동일task-state에서masked/unmasked projection perturbation을분리하고정답readout변화를측정해야한다. 현재생성trace/공통입력state가없으므로이보고서에서는수행하지않았다. 기준모델두개와sham/Both재현부터시작하고,semantic-role vs cardinality-matched randomcontrol 및연속taskreadout를포함하는좁은설계가필요하다. 기존WT2 KL negative결과를새로발견한것처럼반복하지않는다.
