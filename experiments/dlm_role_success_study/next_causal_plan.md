# 다음 원인 검증의 좁은 설계 — 아직 실행하지 않음

## 질문

Role이 만든 task-output 차이가 실제 masked/unmasked 경로의 차별적 보존에서
오는가? 또는 작은 여러 weight 변경이 정답 선택/decoding의 임계값을 넘긴
효과로도 설명되는가?

이 두 설명은 배타적이지 않다. 분석이 역할 분리를 전제로 삼는 것과,
관측된 개선 원인이 역할 분리임을 입증하는 것은 구분한다.

## 기존 실험과 다른 점

- 기존28projection causal decomposition은 WT2 state에서 role별 손상이
  최종 masked-token KL로 어떻게 전달되는지를 봤다. 같은 주장을 반복하지 않는다.
- 이번6bundle개입은 GSM8K EM 변화만 측정했다. neural 경로와 연속적 readout은
  저장하지 않았다. 문항을1319개로 늘려도 새 role-path 측정이 생기지는 않는다.
- 새 진단은 **GSM8K의 동일 입력 state**에서 경로를 분리하고,
  연속적인 정답 관련 출력과 원래 generation 결과를 별도로 비교해야 한다.

## 준비 단계: 공통 입력과 기준 재현

1. Aggregate/Role의 frozen masks, model revision, mini100와 decoding을 그대로 유지한다.
2. 동일100문항 전체를 사용한다. Role-only8개나 ID19만 골라 효과를 일반화하지 않는다.
3. 실제 각 모델 trajectory의 고정 step에서 입력 token IDs/role masks를 저장한다.
   각 원본 generation의 strict 출력 재현 여부를 먼저 확인한다. 불일치시 중단한다.
4. 두 trajectory를 각각 공통-input panel로 사용한다. 두 모델을 서로 다른 입력에
   평가한 차이를 순수 parameter effect로 해석하지 않는다.

## 필수 구현 검증

- 20변경projection만을 대상으로 source A/R의 정확한 BF16 Linear 출력을 계산한다.
- 같은 batch1 입력 경로에서 sham 및 Both가 원본 A/R를 정확히 재현해야 한다.
  차이벡터를 더하는 방식으로 BF16 반올림 경로를 바꾸지 않는다.
- masked 위치에만 A→R 출력 교체, unmasked 위치에만 교체, Both를 비교한다.
- cardinality-matched random partition을 반드시 함께 비교한다. 단순 역할별
  효과 차이는 random control보다 큰 semantic-role 차이를 자동 의미하지 않는다.
- 이 role별 출력 교체는 static equal-budget sparse model이 아닌 **진단용 개입**이다.
  이를 새 방법 성능이나 동일예산 baseline으로 내세우지 않는다.

## Readout을 먼저 고정해야 하는 이유

실제 trajectory의 생성 slot은 gold chain의 token 위치와 자동 정렬되지 않는다.
따라서 현재 slot에 gold token을 임의로 대응시켜 NLL을 계산하면 안 된다.

선택지는 다음 두 개이며 혼합해서 성능을 주장하지 않는다.

- 공통-input 출력/commit decision의 변화를 측정하고 마지막 EM은 기존 결과와
  연관성만 분석한다. 이것만으로 정답 원인이라고 단정하지 않는다.
- 별도의 명시적 teacher-forced answer panel을 만들어 올바른 답과 정해진
  대조답의 sequence score를 측정한다. gold-context 분포라는 한계를 기록하고,
  이를 trajectory capability 또는 downstream 성능의 대체 기준으로 쓰지 않는다.

정렬 가능한 task readout을 구현·검증하기 전에는 대규모 수집하지 않는다.
이 계획은 준비 방향이며 완성된 predictor/할당 점수나 새 실험 실행 기록이 아니다.

## 중단 조건 / 비목표

- sham/Both 재현 실패: 측정 경로를 수정하기 전 실행 중단.
- role 효과가 random partition 대비 구분되지 않음: 해당 진단에서 미확정으로 보고.
- 연속 readout과 EM이 불일치: 그 불일치를 기록하고 readout을 사후 교체하지 않음.
- B0/B1 등의 수동보호, 최고 hybrid 선택, 전체64coalition sweep, 새로운max/mean
  튜닝, downstream labels로 allocation 학습은 수행하지 않는다.
