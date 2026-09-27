# Proposal

## Why

기존 scalar A+C는 동일 native pipeline의 GSM8K mini100에서 Uniform 54, A-only 55에 비해 61을 기록했지만, 어떤 설계 확장이 유효한지는 미확정이다. 중단된 multiscale 실험과 미구현 후보를 동일 50% 예산의 개발용 병렬 screen으로 비교하여, 추가적인 추측보다 재현 가능한 품질·비용 근거로 다음 후보를 선택한다.

이 후보들은 서로 다른 질문에서 도출되었다. Multi는 biased-product Fourier/noise 분석을 문맥 공개 간격에 연결하고, Square는 두 공개 집합의 mixed finite difference를 네 상태로 관측한다. Centered-logit Vector는 scalar가 놓치는 대안 토큰 간 상대 출력을 보존하며, Exchange는 목적함수를 실제 hard mask의 배분에 반영하는 별도 backend다. 출처, 직접 유도, 미검증 pruning 가설, 계산 예산을 위한 설정을 `design.md` §0에서 구분한다. 기존 방법들이 모두 Uniform보다 나쁘다는 전제나 새로운 optimizer 발명은 요구하지 않는다.

## What Changes

- 새 `experiments/dlm_ac_screen50/` 실행 계층에서 10개 평가 arm을 관리한다: 기존 multiscale의 A/Short/Path/All/Multi, Square-A/Square-AC, Vector-A/Vector-AC, Exchange-AC.
- Multiscale의 검증된 64개 probe와 5개 allocation, Short/Multi의 부분 문항 체크포인트를 기존 코드·설정 변경 없이 재사용한다. 기존 Uniform/A/AC mini100은 identity가 맞는 reference로 재사용한다.
- 네 상태 reveal bank 및 scalar A/네 연결 C, 기존 80-pair bank의 centered-logit vector A/C, 기존 AC mask에서 시작하는 제한된 exact-budget 교환을 각각 독립적으로 구현한다. 후보 간 교차 조합은 하지 않는다.
- 이론의 적용 조건과 구현 불변조건을 명세화한다: 고정 query와 product coupling, 정확한 edge/평균 정의, Square의 네 모드 가중치, Vector의 공통 logit 이동 불변성, 실제 정수 weight 예산 및 실측 exchange acceptance. 기존 Multi의 bank·수치·소스는 재해석하거나 재생성해 바꾸지 않는다.
- 명시적으로 지정한 GPU별 작업 큐, tmux 지속성, 원자적 체크포인트, identity 검증, 안전한 중단·재개, 사용자용 상태/ETA 명령을 제공한다.
- 기존 문서에서 불명확했던 진단 규칙을 확정한다: Square는 별도 seed, Vector는 8개 clean span에서 10개 mask 상태를 생성하고, Exchange는 같은 새 진단 입력에서 anchor/final을 비교한다. 새 family의 Uniform calibration 및 완성된 mask의 실측 손실과 그 비용도 명시한다. 이는 기존 calibration bank·mini100 설정의 변경이 아니다.
- 문항 정오답·paired 비교·계산량을 함께 보고한다. 미완료 결과를 100문항 결과처럼 표시하지 않으며 calibration 최적화에 GSM8K 정오답을 사용하지 않는다.
- 준비·검증·dry-run은 CPU 전용이다. 이 변경의 제안 단계에서는 구현이나 GPU 실행을 하지 않는다.

## Capabilities

### New Capabilities

- `ac-screen-protocol`: 고정 실험 행렬, bank/readout 정의, 같은 조건의 대조, exact sparsity, 측정 및 채점 규약.
- `ac-screen-execution`: 기존 실행 호환성, GPU 작업 스케줄링, artifact identity, 중단·재개, 진행 상황과 연구 기록.
- `ac-screen-reporting`: 완결성·출처·paired 성능 및 계산 비용을 구분한 mini100 개발 보고서.

### Modified Capabilities

없음. 현재 OpenSpec main spec 목록은 비어 있다. 기존 실험 코드와 동결 산출물의 의미는 변경하지 않는다.

## Impact

- 재사용 대상: `dlm_multiscale_ac50/{core,artifacts,evaluation,prepare,run,gpu,analysis}.py`, `dlm_context_response50/{core,run}.py`, `dlm_owl65.core.exact_row_counts`, 기존 native Wanda ranking 및 mask manifests.
- 기존 실행의 source-hash 검증 때문에 위 모듈은 원칙적으로 읽기 전용 dependency로 유지하고, 새 코드·설정·출력은 별도 디렉터리에 둔다.
- 기존 PyTorch/NumPy/SciPy, native LLaDA evaluation harness, tmux, 로컬 모델·데이터 cache를 사용한다. 새 학습 모델이나 외부 서비스 의존성은 추가하지 않는다.
- 실제 실행 시 Obsidian에 실험 시작/결과를 기록하고 쓰기 결과를 확인한다. planning만으로 running 상태를 만들지 않는다.
- 범위 밖: full GSM8K, 자동 confirmation, sparsity/seed/계수 sweep, weight finetuning, soft-mask/미분 가능한 배분, DKD-style 확률 readout, 후보들의 전수 조합. 이들은 별도 후속이며 이번 비교의 숨은 fallback으로 넣지 않는다.

DKD는 centered-logit의 다른 이름이나 기각된 아이디어가 아니다. 9월 18일 centered-logit 제안과 9월 22일 DKD 제안은 서로 다른 출력 geometry를 갖는 유효한 경쟁 후보다. 이번 수정은 기존 10-arm screen을 유지하므로 DKD를 새 실행 arm으로 추가하지 않으며, §0.6에 별도 정의와 필요한 대조군을 보존한다. 이는 범위 선택이지 centered-logit의 이론적 우월성이나 사용자의 세부 설정 확정으로 표현하지 않는다.

## Evidence and interpretation

기존 기록에서 공통 56문항의 AC/Short/Multi는 35/33/31이다. 이는 multiscale의 이득을 입증하지 않으며, 전체 100문항과 같은 bank의 대조군 평가가 남아 있다. 구현 완료의 acceptance는 성능 개선이 아니라 명세 준수·검증·재개 가능한 실행 준비이며, 실제 screen 성공은 모든 요청 arm의 유효한 결과 또는 명확한 실패 기록으로 정의한다.

기록된 점수는 과거 receipt를 재검증한 경우에만 재사용한다. 이번 문서 수정은 신규 성능 측정이 아니다. Holm 보정도 이미 반복 사용한 mini100을 독립 confirmation으로 바꾸지 않는다. 실제 실행 범위는 후속 사용자 요청에 따른다. 구현과 실행을 함께 요청하면 준비 검증 뒤 같은 요청 안에서 실행할 수 있으며, 이 문서는 추가적인 이중 승인 단계를 요구하지 않는다.
