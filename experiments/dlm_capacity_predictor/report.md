# DLM Capacity Predictor — 최종 결과 (2026-09-10)

**승인 계획의 두 경로 검증 완료. Dense 통계 경로 중단, Probe16 held-out gate FAIL.**

## 최종 setup 및 결과

Historical Standard Wanda의 local ranking과 224 projection, 50/55/60/65/70/75% grid를 유지했다.
모든 비교 모델의 제거 수는 4,536,008,704 / 6,979,321,856 = 64.99211238%로 같다.
명목 65%와의 차이는 row-wise 정수 절삭 때문이다.
새 8 spans(sequence 24–31) × t={0.1,0.3,0.5,0.7,0.9}=40 states에서 평가했다.
기존 calibration/held-out을 포함한 네 split의 token-span 비중첩 검증 완료.
Same-path dense sham max absolute difference=0.0.

Mean KL은 state별 masked-token mean KL의 평균, median은 pooled masked-token median이다.

| Method | Mean KL ↓ | Median KL ↓ | Top-1 agreement ↑ |
|---|---:|---:|---:|
| Uniform | 0.604508 | 0.426077 | 65.04% |
| EIS+type | 0.571788 | 0.346235 | 65.85% |
| Probe16 | 0.595493 | 0.385738 | 64.29% |

### Probe16 − Uniform

- Mean difference −0.009014、relative reduction 1.49%; median state difference −0.036716.
- Paired state bootstrap 95% CI **[−0.039765, +0.027175]** (20,000 resamples, seed 0).
- Sequence-cluster bootstrap 95% CI [−0.054768, +0.046587].
- 개선/악화 states 28/12, 개선 sequence means 5/8, timestep means 3/5.
- CI 전체가 0 미만이라는 사전등록 조건 미충족 → **FAIL**.

### Probe16 − EIS+type (descriptive)

- Mean difference +0.023705; state bootstrap 95% CI [+0.002371, +0.045508].
- Sequence-cluster bootstrap 95% CI [−0.002379, +0.050523].
- 개선 states 14/40, sequence means 2/8, timestep means 1/5.

### Dense 통계 경로

8개 fold 전부에서 reconstruction-only보다 개선해야 한다는 기준을 적용했다.
variation_ratio 5/8, feature_use_variability 6/8, log_pr_ratio 3/8 개선으로 모두 탈락했다.
선택 feature 없음. 기준을 변경하지 않고 경로를 중단했다.

## 종료 결정과 해석 제한

Probe16의 GSM8K mini/full은 gate 실패로 실행하지 않는다. 공식 EIS/OWL/LSA 및 추가
sparsity/selector/model 검증은 이번 후보의 자동 후속 실행이 아니다.

이번 저비용 근사 방식은 명확한 full-model 이득을 입증하지 못했다. 기존 oracle allocation의
성공을 뒤집는 결과는 아니다. Probe 수 부족, curve-shape 오차, projection 상호작용 중
실패 원인을 이 결과만으로 확정할 수 없다. KL은 downstream 성능의 대체 지표가 아니다.
EIS+type은 oracle에서 유도한 descriptive control이며 공식 독립 저비용 EIS가 아니다.

실제 Probe16 및 local 6점 reconstruction curve 수집 비용은 미측정이다.
저장 oracle 기록을 재사용한 진단 시간을 deployment speedup으로 주장하지 않는다.
사전 교차 검증은 기존 80-state Wanda mask family를 고정한 조건이며 selector까지 독립 검증한 것은 아니다.

원본: `heldout_dlm_results.json` (state/sequence/timestep 상세 및 gate),
`predictor_validation.json`, `state_verification.json`, `existing_results_audit.json`.
고정 계획·설정·최종 결과 JSON은 변경하지 않았다.

---

# 과거 중간 연구 보고서 (아래의 평가 대기 문구는 최종 상태가 아님)

## 현재 범위

새 weight score가 아니라 Standard Wanda의 projection별 sparsity allocation을 검증한다.
기존 6점 curve는 학습/진단 자료이며 새 방법의 무료 deployment 자료로 간주하지 않는다.

## 16-state probe 진단

교차 검증 판정: 후보 동결.

| Construction sequences | Reconstruction (40 states) | Probe (8 states) |
|---|---:|---:|
| [0, 1, 2, 3] | 0.527045 | 0.294106 |
| [4, 5, 6, 7] | 0.522125 | 0.297102 |

표의 수치는 unseen sequence에서 평가한 single-projection additive damage이다.
Full sparse model KL이나 GSM8K 성능이 아니며, 좋은 결과를 보장하지 않는다.

## Dense 통계 경로

선택 feature: None

## 비용과 남은 검증

16-state probe 실제 실행시간, local reconstruction curve 수집시간은 아직 미측정이다.
기존 oracle lookup으로 측정한 CPU 분석시간을 deployment speedup으로 해석하지 않는다.
후보 동결 후 새 disjoint states에서 full-model gate, 통과 후보의 downstream 검증이 필요하다.
EIS+type은 oracle에서 유도한 기술적 대조군이며 공식 EIS baseline이 아니다.
