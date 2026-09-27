# 실행 기록 — 최종 상태 (2026-09-10)

승인 계획의 두 경로 검증은 종료했다. Runner 완료: 13:20:29 KST, `through=heldout`.

| 단계 | 최종 상태 |
|---|---|
| 기존 네 방법 KL/mini/full GSM8K 및 예산·protocol audit | 완료 |
| 16-state probe 교차 검증 및 후보 동결 | 완료 |
| Dense 통계 수집·원시 진단·8-fold 검증 | 완료, 통과 feature 없음 → 경로 중단 |
| 새 disjoint 40 states 및 same-path sham 검증 | 완료 |
| Probe16 full-model held-out gate | FAIL |
| 통과 후보의 GSM8K mini/full | 통과 후보 없음 → 실행하지 않음 |
| 공식 EIS/OWL/LSA, 추가 sparsity/selector/model | 조건부 후속 연구, 이번 실행 범위 아님 |

최종 수치는 `heldout_dlm_results.json`, 해석은 `report.md` 상단을 기준으로 한다.
실제 Probe16/local curve deployment 비용은 미측정이다. 실패 후보를 위해 추가 GPU 측정을 실행하지 않았다.
`status.json`의 일반적인 `next_step` 문구는 추가 승인 실험이 남았다는 뜻이 아니다.

---

# 과거 중간 실행 기록 (아래 대기/미완료 표기는 최종 상태가 아님)

## 완료

- 기존 네 방법 결과 audit 및 strict EM 재계산. EIS+type full은 아직 미완료.
- 고정 16-state subset의 저장 curve CPU 진단; 두 sequence fold 모두 additive damage 개선.
- 정확한 row-floor 예산으로 probe 후보 동결. 신규 full-model 개선을 의미하지 않음.
- Dense 통계 수집 및 layer×sequence 교차 검증 구현. 누수 방지 테스트 추가.

## 구현 검증 완료

- Collector/평가 runner의 provenance·resume·실행 순서 검사 보강.
- 관련 신규/기존 회귀 테스트 총 67개 통과; Python compilation, shell syntax 검사 통과.
- 224×80×6 source curve 정합성 재검증.
- 기존 GPU 0 GSM8K 정상 종료 후 실행되는 진단 전용 tmux 대기열 구성.

## 대기/미측정

- 실제 실행 상태는 `status.json`, 단계별 상세 로그는 `logs/`를 기준으로 한다.
- EIS+type full 결과와 새 dense 통계 수집은 아직 대기 중.
- 새 후보의 full-model KL/GSM8K 및 실제 probe deployment 비용은 미측정.

## 실행 경계

기본 대기열은 audit → dense 통계 수집 → 원시 진단/교차 검증/후보 동결까지 실행한다.
이번 대기열에서는 새 full GSM8K를 자동 시작하지 않는다. `--through heldout/downstream`은
후속 검증을 위한 별도 실행 옵션이다. 기존 실험과 GPU 1 작업은 변경하지 않는다.

## 해석 제한

Local reconstruction KL 열세가 downstream 열세는 아니다. 현재 full correct는
Uniform 139, Capacity 250, Reconstruction 255/1319이다. EIS+type 결과까지 audit 후 비교한다.
Wanda ranking은 역사적 80 calibration states로 고정되어 있으므로 calibration 내 교차 검증은
고정 후보 마스크 조건의 predictor 검증이다. 완전히 독립된 downstream 검증으로 부르지 않는다.
적은-probe GPU wall time과 local curve deployment 비용은 아직 측정하지 않았다.
