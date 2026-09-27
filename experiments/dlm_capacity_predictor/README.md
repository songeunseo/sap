# DLM Capacity Predictor

두 경로: (1) 16-state 단일 sparsity probe로 local reconstruction curve 보정,
(2) dense DLM-state 입력 통계로 oracle-supervised 보정 계수 예측.
두 경로 모두 기존 Standard Wanda 후보 마스크와 정확한 Uniform-65 parameter budget을 사용한다.

## 실행

저장소 루트에서 실행한다. 동일 이름 세션이 이미 있으면 중복 실행하지 않는다.

```bash
tmux new-session -d -s dlm_capacity_predictor 'bash experiments/dlm_capacity_predictor/run_tmux.sh'
tmux attach -t dlm_capacity_predictor
```

기본 실행은 기존 `projection_capacity_followup65_downstream` 종료, full 결과 audit 성공,
GPU 0의 다른 compute process 부재를 확인한 뒤 아래 세 단계만 진행한다.

1. 기존 네 방법 결과 audit.
2. Frozen 80 states dense forward 통계 수집 (gradient/pruning 없음).
3. 원시 통계·중복성 진단, 고정 교차 검증, 경로별 최대 1개 후보 동결.

상태와 로그:

```bash
cat experiments/dlm_capacity_predictor/status.json
tail -f experiments/dlm_capacity_predictor/logs/pipeline.log
tail -f experiments/dlm_capacity_predictor/logs/dense_collection.log
```

작업은 기존 GSM8K를 중단하거나 GPU 1을 사용하지 않는다. 코드/config가 대기열 생성 이후
변경되면 실행을 중단한다. 실패 원인은 `status.json`과 각 단계 로그에 기록된다.

## 조건부 후속 평가

기본 실행에 포함되지 않는다. 동일 후보를 유지한 채 아래 옵션으로 이어갈 수 있다.

```bash
bash experiments/dlm_capacity_predictor/run_tmux.sh --through heldout
bash experiments/dlm_capacity_predictor/run_tmux.sh --through downstream
```

위 명령도 비싼 작업이므로 별도의 tmux 세션 안에서 실행한다. Held-out은 새 disjoint 40 states를
검증하며, downstream은 사전 등록 DLM gate를 통과한 후보만 평가한다.
Mini는 후보 선택에 사용하지 않는다. 공식 EIS/OWL/LSA 및 추가 모델 비교는 별도 후속 과제이다.

## 연구 해석

- `probe_diagnostics.json`은 이미 저장한 intervention 결과를 재사용한 진단이다.
  신규 probe GPU 실행시간을 측정한 것이 아니다.
- `dense_diagnostics.json`은 회귀 전에 생성된다. 후보 통계의 유용성은 사전 가정하지 않는다.
- `predictor_validation.json`은 고정 Wanda 후보군 조건에서 검증한다. Wanda ranking 자체는
  역사적 80 states로 고정되어 있다.
- Dense predictor는 oracle-supervised 학습이다. Dense deployment 통계와 offline oracle 비용을 구분한다.
- Additive single-projection damage, full-model KL, GSM8K를 서로 대체해 해석하지 않는다.
- EIS+type은 oracle-derived descriptive control이지 공식 독립 EIS baseline이 아니다.
