# Sparse-context 역할별 입력 기여

모든4개 후보가 dense masked/unmasked 절대·상대 오차를 보존한다.
추가 sparse 통계만 none/M/U/MU로 구분한다. 방법의 역할 분리 여부를 다시 검정하지 않는다.
기존 V2 final 반복 사용 탐색 분석이며 새 GPU forward, allocation, GSM8K 실행 없음.

## 실행과 확인

전용 tmux에서 `bash experiments/dlm_role_sparse_attribution/run.sh` 실행.
중복 프로세스가 없는지 확인하고 로그는 `logs/analyze.log`에 기록한다.

```bash
cat experiments/dlm_role_sparse_attribution/progress.json
tail -n 10 experiments/dlm_role_sparse_attribution/logs/analyze.log
```

완료: `results.json`, `report.md`, `predictions.npz`.
행 순서는 config가 hash로 고정한 이전 `dlm_role_proxy_units/prediction_keys.json`과 같다.
기준/full의 기존 single 및 bundle 예측값 재현, 실행 전후 입력 hash 검증을 통과해야 완료한다.
완료된 결과를 덮어쓰는 재실행은 거부한다.

주 비교4개: M-base/U-base/MU-M/MU-U. 동일예산 bundle MSE에 대해4비교 동시 bootstrap.
이는 fitted OOF 모델에 조건부이며 역할의 인과적 중요도나 통계적 necessity 검정이 아니다.
변수가 추가될 때 ridge 정규화 효과가 변할 수 있으며, 이 한계를 보고서에 명시한다.
