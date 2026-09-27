# Cross-chain Full GSM8K 구현 감사 — 2026-09-27

## 판정

수식·배분·현재 생성 결과를 잘못 계산했다는 증거는 발견하지 않았다. 다만 운영 명령, 중복 검출, 설정 기록과 비용/재현 보고에 수정할 부분이 있다. 이전 감사 PASS는 실행 가능한지에 대한 판정이었으며, 계획의 모든 운영·보고 요구를 완전히 충족했다는 뜻으로 확대하면 안 된다.

현재 실행의 과학적 설정은 유지하고 있다. 이 감사는 CPU만 사용했고, 실행 중인 코드·설정·감사 영수증·마스크·답변은 수정하지 않았다. 감사 스크립트와 증거는 별도 research 디렉터리에 저장했다.

## 확인된 문제

| 중요도 | 항목 | 관측 또는 재현 | 영향과 조치 |
|---|---|---|---|
| 높음: 운영 | 실행 중 `validate`를 재호출하면 감사가 무효화됨 | `run.py:49`의 checks()가 time을 포함한 cpu_validation.json을 덮어쓰고, `run.py:43`은 기존 파일 SHA를 요구한다. | 현재 worker는 계속될 수 있으나 다음 worker 또는 최종 report에서 실패한다. 이번 감사는 launch_gates()만 읽기 전용 호출했다. 검증 명령을 읽기 전용 검사와 사전 receipt 생성으로 분리해야 한다. |
| 중간: 무결성 | 동일 ID의 다른 파일명 중복이 묵인됨 | shared evaluation.py:98은 int(filename) dictionary로 변환한다. 실제 checkpoint를 임시 폴더의 0000.json/0.json 두 파일로 복사했을 때 예외 없이 한 행을 반환했다. | 실제 snapshot에는 중복·비표준 파일명이 없었다. 로딩 전에 정규 파일명과 ID 유일성을 검사해야 한다. |
| 중간: 비용 | 최초 실패 시도의 비용이 최종 합계에서 빠짐 | run.py:514는 현재 output/costs만 합산한다. 별도 보존한 첫 시도에 dense 128 forwards가 존재한다. | 완료 후 비용 보고에 현재 run과 보존된 실패 시도를 함께 집계해야 한다. 감사 evidence.json에는 둘을 구분해 집계했다. |
| 중간: 비용 | SIGTERM 종료 시 비용 기록 누락 가능 | worker에는 SIGTERM을 Python 예외로 바꾸는 handler가 없다. finally에서만 costs를 쓰므로 기본 SIGTERM 종료 시 실행이 보장되지 않는다. 최종 report는 attempts와 costs의 차이를 검사하지 않는다. | 문항 checkpoint로 재개는 가능하나 중단 문항의 실제 forward 수는 복구 못 할 수 있다. 주기적 비용 기록, 종료 handler, 누락 attempt를 표시하는 비용 보고가 필요하다. 현재 누락 목록은 실행 중인 네 worker에 해당한다. |
| 중간: 기록 | config.plan에 과거 mini 계획이 남음 | prepare.py:173에서 옛 config를 복사하고 plan을 교체하지 않는다. full_gsm8k=false, max_new_generations=800, max_unique_questions=200, 과거 후보/선별 규칙이 남았다. | 실제 코드·requests는 새 Full1319×4이고 해당 과거 선별 규칙을 실행하지 않는다. 유효 설정을 effective_plan.json에 별도로 정리했다. 동결 config 원본은 보존했다. |
| 중간: 산출물 | 이전 200문항의 답변 재현 비교와 오류 유형 보고가 없음 | report()는 200개 점수를 출력하지만 과거 generated_text/정오표를 비교하지 않는다. strict-invalid/valid-wrong 등의 오류 분류도 제공하지 않는다. | 원본 답변에서 CPU 후처리로 보완 가능하다. 현재 확보된 A/Multi 첫100은 이번 감사에서 과거 출력과 문자열까지 모두 일치했다. 나머지 과거100은 생성 후 비교해야 한다. |

## 독립 검증

- 공개된 수식을 별도 루프로 구현해 Uniform β와 32×2 probe의 네 marginal cost를 다시 계산했다. 최대 차이 5.13e-23이며, 네 후보의 최종 rank/DP row count가 모두 일치했다.
- 네 후보의 224개씩, 총 896개 실제 packed mask 파일을 읽어 각 row의 제거 비트 수와 mask hash를 확인했다. 후보별 제거 수는 모두 3,489,660,928이다.
- A/Multi의 저장 sparse-model SHA와 mask identity가 원본과 일치했다. 현재 실행의 GPU worker도 실제 모델 SHA를 검사한 뒤 생성했다. 감사 자체에서는 GPU 모델을 새로 로드하지 않았다.
- 1,319개 prompt 해시와 tokenizer 입력 해시를 CPU에서 모두 재계산했다.
- 확인 시점의 checkpoint 518개(A131, Multi130, Cross129, CrossMatched128)를 공식 strict-match로 재채점했다. 원본 요청, 모델 fingerprint, row hash, ID/파일명/shard가 모두 일치했다.
- A/Multi의 이전 개발100문항 각각에 대해 generated_text, extracted_answer, correct가 모두 일치했다. 이는 현재 모델·prompt·생성 경로가 원본을 재현한다는 직접 관측이다.
- teacher와 네 후보의 초기화 완료 영수증 5개, 12개 후보-bank readout의 형태와 metric 계산을 확인했다.
- exact McNemar p값을 직접 이항계수 합과 비교했고 Holm 세 비교 예제를 별도 확인했다. 1,119문항의 세 주 비교와 seed20260927의 paired bootstrap 구조는 계획과 같다.

## 주의해서 읽어야 하는 진단 정의

현재 response_sign_flip_rate는 기존 코드의 같은 chain 전체 28개 단계쌍에서 |dense response|>1e-6인 위치의 부호 반전을 계산한다. C가 사용하는 12개 multiscale edge 또는 cross-chain 부호 오류와 같은 정의가 아니다. 계획에서 sign metric의 edge 집합을 별도로 고정하지 않았으므로 수식 오류로 판정하지 않았지만, 결과 표에는 이 정의를 명시해야 한다.

## 실행 판단과 남은 작업

이번 점검에서 현재 답변 생성이나 allocation을 다시 돌려야 할 오류는 발견하지 않았다. 이미 수정한 teacher→initializer scheduler 전환 이후 첫 shard 네 개가 완료되고 다음 shard들이 시작된 것도 확인했다. 현재 실행은 유지한다.

실행 중인 코드나 `*.py/*.sh` 목록을 바꾸면 동결 hash gate가 깨지므로, 위 결함을 수정한 새 버전은 별도 revision으로 검증해야 한다. 완료 후 CPU 후처리에서는 중복 파일명 엄격 검사, 이전200 답변 재현 비교, 오류 유형, 보존된 첫 시도까지 포함한 비용 집계를 반드시 추가한다. 현 상태를 "계획의 모든 항목 구현 완료"로 보고하지 않는다.

## 산출물

- audit.py: 원본을 읽는 CPU 감사 코드
- evidence.json: 실제 검증 결과와 입력 코드/설정 해시
- effective_plan.json: 구 mini metadata와 구분한 실제 Full 실험 설정

## 독립 감사자 교차 확인

기존 crosschain_audit 감사자는 과거 개발100 및 별도100의 총200 요청에서 doc/prompt/target/token hash와 reference_answer가 모두 같고, 주 비교1119와 겹치지 않음을 확인했다. 운영·비용·metadata 문제에 동의했으며 추가적인 중대 수식·배분·주 통계 오류는 발견하지 않았다. 이전 답변 문자열 비교와 오류 유형은 주 통계 오류가 아닌 최종 산출물 보완 사항으로 구분한다.
