# 문헌에서 다시 설계한 DLM pruning 후보 — 2026-09-22

Status: hypothesis / design review. No implementation, GPU forward, pruning mask, or evaluation was run in this review.

## Decision

우선순위를 “A를 악화시키지 않는 제약 추가”에서 **어떤 출력 정보를 보존할 것인가**로 옮긴다. 먼저 기존 gold-reveal bank에서 target confidence와 대안 토큰 분포를 분리해 보존하고, 그 대안 분포가 문맥 공개에 따라 바뀌는 반응을 추가로 보존하는 후보를 검토한다. 생성·오류 문맥은 별도 후속 축으로 남긴다.

이는 현재 A+C보다 성능이 좋다고 관측된 방법이 아니다. DKD, 관계 증류, 문맥 민감도 분석에서 설계 근거를 가져온 **검증 전 조합**이다. 새로운 solver나 “DKD+관계” 조합 자체를 novelty로 주장하지 않는다.

## Existing evidence

- 현재 동일 native 50% mini 비교는 Uniform 54, A-only 55, A+C 61이다. A+C 대 A의 rescue/regress는 9/3이며, mini 하나에서 우월성이 확정되지는 않는다.
- 별도 cached Uniform 62는 ranking/mask 생성 경로가 달라 이 표에 직접 혼합하지 않는다.
- A+C full evaluation은 중단되었고 독립 NELBO 결과는 아직 없다.
- 현재 A와 C는 모두 gold-versus-rest log-odds 하나를 읽는다. C는 같은 masked query에서 gold reveal 전후의 scalar 변화가 dense와 얼마나 다른지 측정한다.
- 현재 natural pair는 문맥 내용과 mask 수가 함께 변한다. 이를 순수한 의미 정보 효과라고 부르지 않는다.
- 과거 vector-logit response, trajectory bank, exact-budget swaps는 이미 논의되었다. 이번 검토에서 새 발견처럼 재소개하지 않는다.

## Paper-to-design ledger

| Primary source | 확인한 결과 / 적용 범위 | 우리 설계로 가져오는 내용 |
|---|---|---|
| [DKD, CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html), §3 Eq5–7 | image classification/distillation에서 target-vs-rest와 conditional non-target distribution을 분해한다. 일반 KL은 후자에 teacher non-target mass를 곱한다. | scalar A+C가 버리는 대안 간 정보를 명시한다. DKD의 이미지 결과가 frozen DLM pruning의 성능 증거는 아니다. |
| [RKD, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html), §3.2 | sample 간 distance/angle 관계를 증류한다. 개별 출력값이 중요한 문제에는 관계 항만으로 부족하다. | 문맥 전후 관계를 보되 endpoint anchor도 유지한다. magnitude를 없애는 angle/correlation만 복사하지 않는다. |
| [CoRe, 2026](https://arxiv.org/html/2602.04096), §3–5 | LLaDA의 inference-time remasking에서 visible context를 가린 뒤 token support가 얼마나 무너지는지가 단일 상태 confidence와 다른 수정 신호를 준다. | 문맥 반응은 DLM에서 기능적 의미를 가질 수 있다. 이 논문은 decoding 변경이며 static weight allocation의 직접 증거는 아니다. |
| [Shin et al., EMNLP2024](https://aclanthology.org/2024.emnlp-main.68/), §3 Tables1–2 | AR LLM reconstruction에서 calibration 오차 감소와 held-out quality가 엇갈릴 수 있다. surviving weight를 갱신하는 설정이다. | calibration-A 하한은 성능 보장이 아니다. 목적함수 개선과 downstream 개선을 분리한다. |
| [Corrective DLM, 2026 revision](https://arxiv.org/html/2512.15596), §2–4 | 기존 masked-only 학습은 visible-but-wrong token 판별에 한계가 있고, 제안한 corrective training은 학습 목적을 바꾼다. | generated/corrupted bank를 먼저 주력화하지 않는다. dense 반응을 복사한다고 없는 오류 수정 능력이 생기지는 않는다. |
| [LiPRA, publisher abstract/introduction](https://www.sciencedirect.com/science/article/abs/pii/S092523122602151X) | 전체 budget을 유지한 rate perturbation, global sensitivity와 KKT 기반 allocation을 설명한다. 전문 미확보. | budget exchange/global sensitivity는 가까운 기존 방법이다. 수식·벤치마크를 독립 검증했다고 주장하지 않는다. |
| [DRDKD, publisher abstract/introduction](https://www.sciencedirect.com/science/article/pii/S0925231226012476) | target/non-target 분리와 sample 관계 보존을 이미 결합한다. 비전 증류 설정이며 전문 전체는 미확보. | “decoupling + relation”은 신규성 근거가 아니다. 같은 masked query의 조건부 변화와 static weight budget이라는 구체적 문제로 좁혀야 한다. |

LiPRA publisher page에는 미래 issue date(2026-11-28)가 표시되지만 이 검토 시점에 abstract/introduction은 조회되었다. 정확한 online-first 날짜나 full-text 접근을 추정하지 않는다. 연도 이상의 출판 시점 주장은 보류한다.

## 왜 scalar가 부족할 수 있는가

구성한 반례이며 실측 빈도가 아니다. 세 후보의 첫 좌표가 reference gold일 때:

- dense before: (.4, .5, .1)
- dense after:  (.4, .1, .5)
- sparse before/after: (.4, .5, .1)

gold-versus-rest는 모두 .4/.6이므로 기존 scalar A=C=0이다. 그러나 dense의 최고 후보는 바뀌고 sparse는 반응하지 않는다. **일반 full-distribution KL도 이 반례를 잡는다.** 따라서 이 예는 scalar blind spot만 증명하며 response 항의 필요성까지 증명하지 않는다.

## One concrete provisional objective

기존 bank의 동일 query를 사용한다. D는 dense, M은 sparse mask 모델, e=0/1은 공개 전/후, y는 두 endpoint에서 동일한 실제 corpus reference token이다.

각 endpoint의 확률 p에서:

- b_X^e = (p_X^e(y), 1-p_X^e(y))
- q_X^e(v) = p_X^e(v)/(1-p_X^e(y)), v != y

b는 reference confidence, q는 reference 이외 후보 사이의 조건부 확률이다. q는 gold logit을 제외한 logits의 softmax로 안정적으로 계산한다. 데이터에 없는 teacher 생성 token을 gold라고 부르지 않는다.

표준 KL의 정확한 분해는:

KL(p_D || p_M) = KL(b_D || b_M) + (1-p_D(y)) KL(q_D || q_M).

현재 squared log-odds A는 binary KL과 동일한 metric이 아니다. 따라서 아래 후보는 C만 바꾸는 통제가 아니라 readout/metric을 바꾸는 비교다.

**Endpoint:**

A_dec(M) = E_(pair,query) [ (1/2) sum_(e=0,1) { KL(b_D^e || b_M^e) + beta KL(q_D^e || q_M^e) } ].

**Context response:**

C_alt(M) = E_(pair,query) || (q_M^1-q_M^0) - (q_D^1-q_D^0) ||_2^2.

**Allocation target:**

min_M A_dec(M) + lambda C_alt(M), subject to an exact global surviving-weight count, frozen surviving weights, and the same within-row Wanda support family.

이 식은 우리 가설적 적용이다. DKD가 C_alt를 제안하거나 CoRe가 이 weight objective를 검증한 것은 아니다. pair 순서를 동시에 뒤집으면 squared loss는 같다. 따라서 “directional”은 forward/backward 비대칭이 아니라 각 vocabulary 좌표의 signed response를 비교한다는 의미다.

C에는 normalized probabilities를 사용해 모든 극저확률 logit에 균일한 큰 오차를 부과하지 않는다. 그 대가로 saturation과 작은 확률의 중요한 변화에 둔감할 수 있다. beta가 큰 상태에서는 confident teacher의 희박한 대안에도 과도한 budget을 줄 수 있다. 최적 metric이라고 주장하지 않는다.

beta, lambda, temperature는 아직 실험 설정으로 확정하지 않았다. 첫 비교에서는 beta=1, temperature=1을 명시적 기본안으로 둘 수 있지만 lambda의 단위/정규화는 시작 전 calibration-only 규칙으로 고정해야 한다. mini-100 성적을 반복 조회하며 고르지 않는다. 계수를 확정하지 않은 이 문서는 실행 승인이거나 실행된 setup이 아니다.

## Endpoint와 response의 역할 및 반례

A_dec는 양 endpoint의 전체 분포를 규정한다. 완벽한 endpoint matching이면 C_alt도 0이다. 그러므로 C_alt는 endpoint에 없는 관측을 마법처럼 추가하지 않는다. 한정된 sparsity에서 어떤 오차를 더 허용할지 바꾸는 regularizer다.

e_i=q_M^i-q_D^i이면 C_alt=E||e_1-e_0||²이다. 이는 문맥에 따라 바뀌는 오류를 더 크게 벌하지만 일정한 공통 오류에는 둔감하다. 그 공통 오류는 endpoint 항이 벌해야 한다.

따라서 서브에이전트의 원안 A_binary + centered non-target response만 채택하지 않는다. non-target 상대 logit의 동일한 편향이 양 endpoint에 남아도 response는 0일 수 있고, gold logit을 조절하면 binary A까지 0으로 유지할 수 있기 때문이다. non-target endpoint anchor가 필요하다.

## Why this is a DLM research hypothesis

가설은 “기존 allocation은 모두 uniform보다 나쁘다”가 아니다:

> 동일한 sparsity와 유사한 endpoint fidelity에서도, 부분 공개된 문맥이 바뀔 때 후보 분포를 갱신하는 반응의 손실이 DLM 생성 손상의 차이를 설명하며, 이 반응을 보존하도록 budget을 배분하면 endpoint matching만 했을 때보다 품질이 좋아질 수 있다.

CoRe는 문맥 반응의 decoding relevance를, DKD는 누락된 분포 성분을 알려준다. 그러나 AR에서도 조건부 반응은 정의할 수 있다. DLM 고유성/더 큰 민감도는 별도 실험 없이는 주장하지 않는다. 작은 endpoint loss가 downstream quality를 보장한다는 주장도 하지 않는다.

## Common allocator, not new machinery

1. 공통 bank, initial mask, pruning domain, within-row ranks, exact budget을 고정한다.
2. 현재 sparse model에서 budget을 유지한 소량의 제거/복원 후보를 만든다.
3. complete candidate mask의 실제 목적함수를 같은 상태들에서 측정한다.
4. 측정된 개선을 수용하고, 같은 계산 예산 또는 사전 정의한 개선 정지 조건에서 종료한다.

기존 finite-probe allocator와도 먼저 비교할 수 있다. 중요한 것은 객관함수 사이에서 backend/지원 mask 집합을 바꾸지 않는 것이다. solver 개선과 objective 개선을 동시에 자기 방법의 성과로 합산하지 않는다.

## Minimal discriminating comparison

기존 scalar A+C를 reference로 두고, 같은 bank/backend에서:

1. Full-vocabulary endpoint KL.
2. Decoupled endpoint A_dec.
3. A_dec + C_alt.

Uniform과 A-only는 기존 실험 anchor로 함께 보고, 필요하면 동일 backend에서 재생성한다. DSA의 DLM-loss search는 관련 강한 baseline이므로 최종 연구 비교에 포함한다.

- 1→2 개선: 주로 target/non-target weighting에서 온 증거.
- 2→3 개선: paired response regularization의 추가 가치에 관한 증거.
- 3이 scalar A+C만 이김: response novelty를 아직 입증하지 못함.
- 모든 objective가 비슷하게 개선: 공통 solver/calibration의 기여일 수 있음.

하나의 mini-100에 반복 적합하지 않는다. calibration과 분리된 task accuracy를 사전 지정한 primary quality로 보고, NELBO/full-KL/response error를 complementary diagnostics로 보고한다. NELBO 향상을 task 향상의 필요조건으로 요구하지 않는다. 가능하면 비슷한 held-out endpoint KL을 가진 여러 mask 사이에서 response error와 실제 실패의 관계를 확인한다.

## Deferred branch: generated-context interventions

DLM agent는 생성 문맥에서 controlled visible-token intervention에 대한 반응을 보자는 후보를 제안했다. 이는 실제 decoding state와의 정합성이 장점이나, 기존 Reveal-KL의 dense trajectory calibration과 겹치며 bank/readout 변화가 동시에 들어간다.

후속 단계에서 탐구할 수 있지만, 오류임이 알려지지 않은 alternative substitution을 correction이라고 부르지 않는다. 같은 mask 수의 pair도 위치·증거 내용이 함께 바뀌므로 순수 semantic control은 아니다. matched-count에서 gain이 사라져도 “원인은 전부 clock이었다”라고 결론내릴 수 없다.

## Novelty assessment

- DKD를 DLM pruning에 적용했다: 단독으로는 얇은 적응.
- 복잡한 constrained solver를 붙였다: 기존 sensitivity/global allocation과 겹침.
- DLM pruning이 confidence/endpoint fidelity와 conditional response를 다르게 손상시킴을 관찰하고, 그 차이를 겨냥한 budget allocation이 강한 통제 대비 개선: 더 설득력 있는 기여 후보.
- 생성 문맥까지 포함한 framework: 각 구성요소가 독립적으로 필요하다는 근거가 생길 때 확장한다.

현 단계는 논문감이 입증된 방법이 아니라, 무엇을 관찰하면 method를 정당화할 수 있는지가 이전보다 구체적인 연구 가설이다.

## Reports

- literature_objectives_2026-09-22.md: distillation agent's review, including later critique.
- literature_dlm_2026-09-22.md: DLM agent's alternative and limitations.
- literature_allocation_2026-09-22.md: parent allocation/pruning review.
- ac_framework_upgrade_2026-09-22.md: earlier framework, now lower priority on a hard A floor.

