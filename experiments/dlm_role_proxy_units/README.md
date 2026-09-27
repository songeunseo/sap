# 역할 보존 proxy 단위 분석

기존 V2 측정 재사용, CPU-only 탐색 분석. 모든6개 후보에서 masked/unmasked 분리 유지.
실험 전제·4개 주 비교는 `../dlm_role_exchange_prediction_v2/next_analysis_plan.md` 참조.

## 실행 / 확인

```bash
tmux new-session -d -s role_proxy_units 'bash experiments/dlm_role_proxy_units/run.sh > experiments/dlm_role_proxy_units/logs/analyze.log 2>&1'
cat experiments/dlm_role_proxy_units/progress.json
tail -n 10 experiments/dlm_role_proxy_units/logs/analyze.log
```

실행 전 logs/ 디렉터리를 만들고 중복 tmux/프로세스가 없는지 확인한다.
이미 완료된 results.json이 있으면 재실행을 거부한다. 실행 중 예외 발생 시 로그를 확인한다.

## 산출물

- config.json: 후보, 비교, 입력/코드 hash 동결.
- results.json: 완료 표시, 모든 metrics/strata/fit/CI/잔차분해.
- report.md: 한국어 요약 및 해석 한계.
- rank_audit.json: 역할/context별 순위 변화, 개발 문서 LOO 재현성, 전체 projection/layer/type 분포.
- predictions.npz, prediction_keys.json: 재현 가능한 단일 및 bundle 예측과 행 식별자.
- progress.json, logs/analyze.log: 단계별 진행 상태.

V2와 같은 P2/P3/P4에 해당하는 후보는 기존 MSE를 재현해야 분석이 진행된다.
분모가 원본 role_feature에 저장되지 않았으므로 출력 스케일의 직접 원인 분석은 보류한다.
KL 예측력은 GSM8K 우위와 다르다. 새 allocation이나 downstream 평가는 실행하지 않는다.
