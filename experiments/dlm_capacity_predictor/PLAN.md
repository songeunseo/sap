# DLM capacity predictor: 승인된 두 경로 실행 계획

## 고정 조건

LLaDA-8B-Base revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, 기존 224 projection 순서,
Standard Wanda 후보 마스크, 50/55/60/65/70/75% grid를 유지한다.
실제 row-floor Uniform-65 예산은 4,536,008,704 / 6,979,321,856 weights이다.
기존 실험은 수정하지 않는다. GPU 0의 기존 GSM8K 종료·성공 확인 후 새 GPU 작업을 시작한다.

## 실행 순서

1. 기존 네 방법의 KL/mini/full GSM8K 및 동일 예산·예제·protocol audit.
2. 기존 224×80×6 곡선으로 16-state probe 진단. 각 sequence의 t=.15,.85만 anchor로 사용.
   Reconstruction-only, anchor×공통 shape, anchor/local×local shape, measured curves 비교.
   Sequence 0–3/4–7 양방향 교차 검증; common shape는 construction states/modules만 사용.
3. Dense forward에서 sequence/token 위치를 맞춰 timestep 입력 통계 수집.
   상태 변화 비율, feature second-moment 사용 변화, 평균/변화 성분의 64차원 sketch covariance PR.
   원시 분포, type별/층·type 보정 상관, activation energy/outlier와 중복, split 안정성을 먼저 저장.
4. 통계 진단 후 feature별 univariate 표준화 선형 회귀로 log(D65/E65) 예측.
   4/4 sequence 양방향 및 layer 4-fold를 교차하여 unseen layer/state additive damage 검증.
   학습 fold에서만 표준화/회귀. 각 fold의 test projection에서 정확한 Uniform-65 예산 적용.
   모든 fold에서 reconstruction-only를 이긴 후보 중 평균 validation damage가 가장 낮은 1개만 선택.
   실패하면 dense-statistics 경로 중단. Oracle-supervised 학습이며 oracle 비용은 별도이다.
5. Probe 경로는 양방향 sequence split에서 reconstruction-only를 이길 때만 1개 후보 freeze.
6. 후보/계수/마스크 hash freeze 후 기존 세 split과 disjoint한 새 8 spans×5 t=40 states.
   Same-path dense sham, paired KL CI, sequence/timestep majority gate. EIS+type 비교는 보고만 한다.
7. Gate 통과 후보만 frozen mini/full GSM8K. Mini는 sanity check이며 후보 선택에 사용하지 않는다.
   공식 EIS/OWL/LSA, 추가 sparsity/SparseGPT/Dream 검증은 결과 이후의 조건부 후속 단계이다.

## 사전 명세 보완 (결과를 보기 전에 고정)

- Feature-use variability: 각 t의 feature second moment를 feature 합으로 정규화한 뒤
  평균 분포와의 squared L2 deviation의 평균 / 평균 분포의 squared L2 norm.
- 세 번째 단일 predictor feature: log(PR_state-varying / PR_state-mean).
  두 PR도 각각 원시 통계로 저장한다. 퇴화한 covariance는 결측으로 처리하고 후보를 부적격 처리한다.
- Sketch는 input dimension별 seed=20260910의 고정 Gaussian projection, dimension=64.
  PR은 sketch 공간의 근사값이며 원래 차원의 정확한 effective rank라고 부르지 않는다.
- 추가 threshold나 수작업 layer 보호 없음. Decision effect는 allocation 변화와 실제 후보 mask XOR로 측정.
- 과거 E(r)는 진단 재사용이다. 이 lookup의 실행시간을 새 방법의 deployment local-curve 수집 비용으로
  주장하지 않는다. 실제 probe/dense collection 비용과 offline oracle training 비용을 구분한다.
- 최초 실행 단위는 audit + 두 경로 진단 + 후보 freeze이다. 새로운 관측 없이 방법 성공을 주장하지 않는다.

## 산출물

config.json, existing_results_audit.json, probe_diagnostics.json, dense_statistics.pt,
dense_diagnostics.json, predictor_validation.json, candidates.json, allocations/, logs/, report.md.
