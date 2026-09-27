# 사례 점검 — EM 정답과 reasoning 보존은 같지 않다

## 선정 기준과 한계

`question_outcomes.json`에서 Role-only 8문항 중 어떤 단일 add hybrid도
맞히지 못한 문항은 ID16,19,76이다. 이 기준으로 세 사례의 이미 저장된
출력을 읽었다. 결과를 보고 선택한 설명 사례이며 독립 검증 집합이 아니다.
별도 모델 채점, 새 생성, strict evaluator 변경은 하지 않았다.

## ID16 — 목표량의 선택

- 정답230, Aggregate310, Role230.
- 여섯 add hybrid는 모두460을 출력했다.
- revert B0…B4는230, revert B5는460이었다.
- Aggregate는 두 기차의 첫날 총거리160에 둘째날 한 기차의150을 더했다.
  Role은 한 기차의 두 날 거리80+150=230을 출력했다.

관측된 차이는 어떤 수를 합치는지/최종 요구량이 무엇인지와 관련된다.
이 텍스트 차이를 특정 projection의 semantic function으로 귀속할 수는 없다.

## ID19 — 모든 되돌림에서 EM이 사라지는 사례

- 정답6, Aggregate3, Role6.
- 여섯 add와 여섯 revert 모두 strict EM 오답이다.
- Aggregate는 남은 거리6을 구했지만 `6/4=3`이라는 잘못된 등식을 출력했다.
- Role은 답6을 출력했지만, 남은 시간1시간을 기준으로 필요한 속도를
  도출하는 gold reasoning 전체를 명시적으로 재현한 것은 아니다.

현재 14개 설정 중 Role만 정답이라는 관측이다. 각 bundle이 고유한 신경회로를
담당한다거나 비선형 neural interaction을 증명한 것은 아니다.
연속적인 점수가 여러 작은 변경으로 임계값을 넘는 모델도 이 패턴을 만들 수 있다.

## ID76 — 정답이어도 중간 계산이 틀릴 수 있다

- 정답5, Aggregate400, Role5.
- gold는 나머지185일에2컵씩: 총550컵/110=5포대.
- Role 출력은 나머지를180일로 잡아 총540컵을 만든 뒤 `540/110=5`라고 적었다.
- 즉 strict EM은 맞지만 기록된 중간 계산은 gold reasoning과 다르며
  표시된 등식도 정확한 산술 등식이 아니다.
- 여섯 add는 모두 틀렸고, revert 중에는 B5만 정답5를 유지했다.

따라서 Role-only gain을 전부 '올바른 reasoning 복원'이라고 해석하면 안 된다.
한 사례를 근거로 Role의 전체 개선이 우연이라고 주장하는 것도 불가능하다.

## 방법론에 주는 제한

총 EM gain, 특정 gain 문항의 보존, 출력에 적힌 올바른 reasoning 보존,
masked/unmasked 기능 경로 보존은 서로 다른 주장이다.
이 중 앞의 두 가지는 저장된 counterfactual 출력으로 조사할 수 있지만,
마지막 기능 경로 주장은 공통 입력에서의 별도 causal intervention이 필요하다.
