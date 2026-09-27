# 기존 Role 개선의 exact-budget bundle 진단

6개 고정 bundle × 2개 jointly-sparse 배경 = 12개 hybrid.
각 hybrid는 기존 동일 mini100을 평가하며 새 후보 선택에 사용하지 않는다.
Aggregate19/Role24 기준 예측은 해시·문항·protocol 검증 후 재사용한다.

## 진행 확인

```bash
python3 /home/tmluser1/sap/experiments/dlm_role_bundle_mini100/status.py
watch -n 10 python3 /home/tmluser1/sap/experiments/dlm_role_bundle_mini100/status.py
tmux ls
```

GPU0: `role_bundle_gpu0` — add_b0…add_b5 순차 실행.
GPU1: `role_bundle_gpu1` — revert_b0…revert_b5 순차 실행.
현재 모델의 생성 진행률과 GPU별 전체 대기열 ETA를 표시한다.
준비 단계는 명시적으로 표시하고 generation tqdm 로그만 0…100 진행률로 센다.
모든 모델 완료 시 `results.json`과 `report.md`가 자동 생성된다.

## 부호

- add_b: `Aggregate + b` − `Aggregate`
- revert_b: `Role` − `Role − b`

둘 다 양수이면 해당 배경에서 Role 쪽 bundle이 도움이 됐다는 뜻이다.
배경 차이는 `revert benefit − add benefit`으로 계산한다.
12개 paired McNemar는 Holm 보정하며 bootstrap CI는 개별 CI이다.
배경 interaction CI 6개는 보정 없는 탐색 결과임을 표시한다.

## 출처 / 실행 안전성

`config.json`은 두 원본 allocation, 6bundle, 기존 evaluator, 코드와 protocol을 고정한다.
모든 pruning mask는 원본 파일 해시·payload 해시·shape·count 검증 후 적용한다.
평가 전후 sparse model hash를 비교한다. 정확 pruned-count는 모두4,536,008,704이다.
각 hybrid는 별도 프로세스에서 dense를 새로 로드하므로 복원 누락/이전 mask 누적이 없다.
완료 예측은 receipt를 검증한 경우만 재사용한다. 실패 시 해당 GPU 대기열을 중지한다.
기존 evaluator의 특성상 생성 중 문항별 결과는 저장하지 않으며 100문항 완료 후 기록한다.
중단된 미완료 hybrid 재실행은 100문항 처음부터이고 원본 예측을 덮어쓰지 않는다.

작업 완료는 원인 진단의 완료가 아니라 이12개 conditional counterfactual 측정의 완료이다.
full1319 인과 귀속 또는 개별 projection/semantic-role 기전으로 일반화하지 않는다.
