# Multiscale A+C: GPU 확보 후 실행할 소규모 검증 계획

작성: 2026-09-22. 상태: **planned — 실행하지 않음**.

이 문서는 실험 계획이다. 새 모델 실행, GPU 예약, 자동 실행 등록은 하지 않았다. 설정 파일은 `ac_multiscale_experiment_plan_2026-09-22.json`이며, 새 bank/evaluator를 지원하는 runner 구현은 아직 필요하다.

## Objective

**같은 calibration 입력과 pruning backend에서, 여러 크기의 문맥 변화에 대한 반응을 함께 보존하는 것이 단순한 대조군보다 좋은 allocation을 만드는가?**

이를 세 가지로 나누어 확인한다.

1. 기존 짧은 변화만 보존하면 놓치는 중간·긴 변화의 오차가 실제로 있는가?
2. Multi로 배분한 최종 sparse 모델이 다른 문장의 반응도 더 잘 보존하는가?
3. 그 변화가 GSM8K 소규모 생성 성능 개선으로 이어지는가?

Full GSM8K와 full PPL/NELBO는 이번 계획에서 제외한다. 생성 길이와 denoising step은 기존 조건을 유지하고 **평가 문항 수를 제한**한다.

## Prior evidence

검증된 동일 native pipeline의 GSM8K development mini-100:

| 기존 방법 | 정답 수 |
|---|---:|
| Uniform50 | 54/100 |
| A-only | 55/100 |
| 기존 A+C | 61/100 |

출처: `experiments/dlm_context_response50/results.json` 및 각 방법의 `results.json`. 기존 A+C 대 A-only는 9문항 개선/3문항 악화이며, 정확 McNemar p=0.145996이다. 유망한 development 결과이고 확립된 우월성은 아니다.

이 100문항은 여러 차례 사용되었다. 다른 mask/pipeline의 Uniform 62/100은 이 비교의 Uniform 기준으로 섞지 않는다. 중단된 full A/AC는 완료 결과로 사용하지 않는다.

## Hypothesis

- **H1 — 실제 문제 존재:** Short의 중간·긴 구간 response error가 남아 있고, 이 오차가 짧은 구간 error와 완전히 같은 순위를 주지는 않는다.
- **H2 — allocation 기여:** Multi가 동일 비용·backend에서 Short/Path/All보다 나은 최종 mask를 선택한다.
- **H3 — task 연결:** response 개선이 mini-100 및 별도 확인 100문항의 정답률 개선과 함께 나타난다.

아직 이 가설들의 실제 모델 결과는 없다. 이론식 검산 결과와 구분한다. 특히 H1만 지지되어도 H2/H3가 자동으로 성립하지 않는다.

## Planned Setup

### 고정할 모델·pruning·평가 조건

- 모델: GSAI-ML/LLaDA-8B-Base, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, bfloat16.
- Global sparsity: 정확히 50%, 6,979,321,856개 prunable weight 중 3,489,660,928개 제거.
- 배분 단위: 32개 block, 각 block의 7개 projection에 공통 비율. 224개 projection의 독립 allocation으로 바꾸지 않는다.
- 기존 Uniform50 sparse-prefix Wanda activation/ranking을 고정하고 살아남는 weight 값도 유지한다. 기존 cache는 hash 및 224개 baseline mask 재현 검증 후 재사용한다.
- Probe: Uniform50 배경에서 block 하나만 48%/52%로 바꾸는 64개 후보.
- 기존 signed marginal cost, 45–55% rank mapping, exact row-count rounding을 그대로 사용한다.
- Readout: gold-vs-rest log odds, `e = f_sparse - f_dense`. FP32 readout, FP64 집계.
- GSM8K: 기존 5-shot, generation length 256, block length 256, denoising steps 256, temperature 0, native batch 1, strict-match EM.
- Seed: evaluation random=0, numpy/torch/fewshot=1234. 기존 평가 protocol hash는 JSON에 기록했다.

목적함수와 solver를 동시에 바꾸지 않는다. 기존 solver는 목표 loss를 정확히 최적화하는 해법이 아니며, 48/52 probe가 최종 45–55 allocation의 상호작용을 보장하지 않는 한계가 남는다.

### 새 bank: 기존 A+C에서 달라지는 설정과 이유

기존 bank는 8개 clean span × 10개 mask 상태 × before/after = 160개 state forward였다. 이번에는 **같은 clean span에서 여러 크기의 변화 구간을 구성**하기 위해 다음 bank로 바꾼다. 새 후보끼리는 아래 bank를 모두 공유한다.

| 항목 | 고정 계획 |
|---|---|
| Allocation용 clean span | 기존 WikiText train sequence index 0–7, 길이 256 |
| 별도 response 진단 span | 기존 heldout manifest의 index 8–15, 길이 256 |
| 각 span의 chain | 독립 mask coupling 2개 |
| 각 chain의 phase | 8개 |
| query Q | span마다 사전에 균등 무작위로 정한 8개 위치; 모든 phase/두 chain에서 masked 유지 |
| context visibility p | 0.05, 0.108789, 0.220647, 0.396365, 0.603635, 0.779353, 0.891211, 0.95 |
| grid | logit(p) 균등 |
| chain 생성 | Q 외 위치마다 독립 U를 한 번 뽑고, U≤p일 때 gold token 공개 |
| bank seed | allocation=20260922, response 진단=20260923 |
| state 수 | 각 split 8×2×8=128개 |
| 집계 | query→chain→span 순서로 동일 가중 평균 |

Q 선택과 U 추출의 RNG stream을 분리하고, split/span/chain별 seed를 manifest에 기록한다. 최소 한 token 공개를 강제하거나 변화 없는 pair를 버리지 않는다. `p`는 Q를 제외한 context의 공개 확률이며, 전체 입력의 실제 mask 비율을 별도로 기록한다.

기존 heldout span은 allocation span과 token 구간이 겹치지 않는다는 audit이 있다. **문서 단위 독립성이나 프로젝트에서 처음 쓰는 데이터라는 뜻은 아니다.** 이번 allocation에는 사용하지 않는 response 진단 split으로 취급한다.

새 bank와 기존 bank가 다르므로, 기존 A+C와 Multi만 비교해서 gain을 multiscale 때문이라고 결론 내리지 않는다. 아래 A/Short/Path/All 대조가 필요한 이유다.

### 같은 bank에서 만들 5개 후보

`A = mean_phase(e²)`이고, 각 C는 지정한 edge에서 `(e_j-e_i)²`의 평균이다.

| 이름 | 목적함수 | 확인할 설명 |
|---|---|---|
| A | A | 새 bank 자체 또는 endpoint 보존만으로 충분한가? |
| Short | A+C1 | 짧은 변화 보존만으로 충분한가? |
| Path | A+C_path | 모든 인접 phase를 연결하는 단순한 방식이면 충분한가? |
| All | A+C_all | 모든 phase 쌍, 즉 phase 간 error variance만 줄이면 충분한가? |
| Multi | A+(C1+C2+C4)/3 | 여러 구간 크기를 균형 있게 반영하는 것이 추가 이득을 주는가? |

- C1: (0,1), (2,3), (4,5), (6,7).
- C2: (0,2), (1,3), (4,6), (5,7).
- C4: (0,4), (1,5), (2,6), (3,7).
- C_path: 인접한 7개 edge 전체.
- C_all: 28개 phase pair 전체. K=8에서 `C_all = (16/7) Var_phase(e)`와 일치해야 한다.

총 C 계수는 모두 1이다. inverse-gap normalization, λ 탐색, 추가 loss는 넣지 않는다. Path의 endpoint/internal node 가중치는 서로 다르므로 모든 대조가 node별로 완전히 동일한 가중치를 가진다고 표현하지 않는다.

## Execution stages

### 0. GPU 사용 전 입력·구현 준비

1. 새 node bank와 5개 objective adapter를 구현한다. 기존 `run.py`는 before/after pair 및 A/AC mode를 가정하므로 그대로 실행할 수 없다.
2. 실제 Q/U/node별 token IDs와 source hash를 freeze한다. Monotone reveal, Q 불변, pair edge 수·normalization을 CPU에서 확인한다.
3. 아래 GSM8K development/confirmation의 원래 document ID, question hash, few-shot prompt hash를 freeze한다.
4. 선택된 document만 생성하는 evaluator와 문항별 checkpoint/resume을 준비한다. 기존 helper의 `limit=100`만 바꾸면 별도 subset 평가가 되지 않는다.
5. Cache receipt 및 source revision을 확인하고, 실험 출력을 기존 결과와 다른 경로에 저장한다.

Confirmation IDs는 결과를 읽지 않고 미리 선택해 JSON에 기록했다. 범위 200–1318에서 `SHA256(multiscale-ac-confirm:20260922:<doc_id>)` 순으로 100개를 선택하고, 원래 ID 순으로 평가한다. Development는 기존 ID 0–99다.

원래 dataset ordering과 question identity를 다시 검증한다. Subset 선택 때문에 few-shot RNG 소비 순서가 바뀌지 않도록 원래 순서의 prompt를 CPU에서 구성·freeze한 뒤 선택한다. 선택하지 않은 문항을 모델에 생성시키지 않는다.

### 1. GPU smoke check → 공통 probe 수집

- 최대 32회의 짧은 추가 state forward로 dense 재현성, baseline mask 적용/복원, finite readout을 확인한다.
- Dense의 두 bank readout을 cache한다.
- Allocation bank 128개 state에서 64개 block probe를 한 번씩 평가한다.
- Scalar 출력은 block별로 즉시 저장한다. 같은 출력에서 A, C1/C2/C4, Path, All을 CPU로 계산한다.
- 각 objective의 signed cost 및 allocation을 계산하고 physical mask를 만든다.
- 모든 최종 mask의 정확한 제거 수, surviving weight 불변, source hash를 확인한다.

**5개 objective마다 64-probe 탐색을 반복하지 않는다.** 출력 cache와 backend를 공유하므로 probe forward는 총 8,192회다.

최종 mask가 동일한 후보는 hash로 묶어 downstream을 한 번만 실행한다. Multi가 Short와 같은 mask라면 이번 backend에서는 Multi의 task 이득을 검증할 대상 자체가 없으므로 중복 생성 평가를 중단한다.

### 2. 완성된 mask의 response 진단

Uniform, 기존 A+C, 새 후보 5개를 allocation/진단 두 bank에서 측정한다. 최종 jointly sparse 모델을 실제로 평가하며 block probe의 loss 합으로 대체하지 않는다.

저장할 항목:

- A, C1, C2, C4, Path, All 및 phase별 error.
- Multi 대 Short/Path/All의 중간·긴 구간 error 개선과 endpoint A의 변화.
- Dense response와 sparse response의 부호 불일치. Dense response가 0에 가까운 pair의 비율을 함께 표시하고 과도한 해석을 피한다.
- 고정 Q에서의 gold-token CE. 이것은 query CE이며 NELBO/PPL로 부르지 않는다.
- 32개 layer의 cost, sparsity, 후보 간 mask 차이.
- Span 단위 CPU resampling에 따른 cost 순위/배분 안정성. Edge 28개를 독립 표본 28개로 세지 않는다. 8개 span 기반 안정성 추정은 거친 진단이다.

이 단계에서 수치 오류·mask 동일성·복원 실패는 잡지만, **proxy가 좋아진 후보만 GSM8K에 보내는 필터로 사용하지 않는다.** 정상적이고 서로 다른 5개 후보는 다음 단계에서 직접 비교한다.

### 3. Development GSM8K mini-100

새 A/Short/Path/All/Multi 각각 같은 100문항을 평가한다. 최대 **5×100=500회 답변 생성**이다.

기존 Uniform54/A55/AC61은 model/mask/prediction/protocol/document identity 검증 후 cached reference로 사용한다. Cache 재사용 검증이 실패하면 숫자를 섞지 않고 원인을 해결한다. 추가 baseline 재실행이 필요하면 별도 비용으로 기록한다.

주 비교는 Multi 대 Short다. Path와 All은 “multiscale 설계가 필요하다”는 설명에 대한 강한 대조군이다. A는 새 bank 효과를 확인한다. 기존 A+C는 실제 방법론 업그레이드 여부의 기준이다.

총 정답 수 외에 각 문항의 개선/악화, paired difference와 exact McNemar를 기록한다. 여러 비교를 우월성 검정으로 해석한다면 사전에 정의한 네 새 대조군 비교에 Holm 보정을 적용한다. 반복 사용한 development set이므로 작은 p값만으로 최종 일반화 주장을 하지 않는다.

중간 20/50문항에서 승자를 고르거나 중단하지 않고, 정상 실행되는 후보는 같은 100문항을 완료한다.

### 4. 유망한 경우만 별도 100문항 확인

추천 진입 조건: **Multi가 새 단순 대조군 모두보다 높고, 기존 A+C보다 낮지 않을 때.** 1–2문항 차이는 약한 신호로 표시하고, 3문항 이상 차이는 후속 검증의 우선순위를 높이는 실용적 기준으로만 쓴다. 유의성·성공을 보장하는 문턱이 아니다.

별도 100문항에서는 다음 세 모델만 평가한다.

1. Multi.
2. Development 정답 수가 가장 높은 새 단순 대조군. 동률 순서: Short → Path → All → A.
3. 기존 A+C.

최대 **3×100=300회 답변 생성**이며 동일 mask가 있으면 합친다. 확인 데이터로 λ, p-grid, seed, mask를 다시 선택하지 않는다. 주 비교는 Multi 대 선택된 가장 강한 단순 대조군, 보조 비교는 Multi 대 기존 A+C다. 두 비교에서 추론적 주장을 할 때 Holm2를 적용한다.

이 subset은 **현재 방법 선택에서 제외한 확인용 데이터**다. 다른 과거 프로젝트 실험에서 사용되었을 가능성이 있어 “처음 보는 독립 test set”으로 부르지 않는다. Development와 합친 200문항 점수를 단일 사전 고정 test처럼 보고하지 않는다.

이 단계 후 종료한다. Full GSM8K, 다른 sparsity, 다중 seed, λ sweep은 자동으로 이어서 실행하지 않는다.

## Budget and scheduling

| 작업 | 최대 기본 비용 |
|---|---:|
| Block probes | 64×128 = 8,192 state forwards |
| Dense, Uniform의 두 bank | 2×256 = 512 state forwards |
| 최종 새 후보 5개의 두 bank | 5×256 = 1,280 state forwards |
| 기존 A+C의 두 bank | 256 state forwards |
| 위 수집·진단 합계 | 10,240 state forwards + smoke 최대 32 |
| Development GSM8K | 500 answers = 128,000 generation forwards |
| 조건부 confirmation GSM8K | 300 answers = 76,800 generation forwards |
| GSM8K 전체 상한 | 800 answers; 서로 다른 문제는 200개 |

Calibration state forward는 길이 256이다. GSM8K generation forward는 prompt와 response를 포함하므로 횟수만으로 두 작업의 시간을 동일하게 환산하지 않는다. Mask hash 검증·구축·model loading·I/O 비용도 위 forward 수에 포함되지 않는다.

기존 A/AC/Uniform mini100의 순수 평가 시간은 각각 약 2,064초, 즉 **34.4분/방법**이었다. 비슷한 장비·부하라면 참고 가능한 generation-only 규모는 다음과 같다.

- 새 후보 5개: 약 2.9 GPU-hours.
- 조건부 확인 3개까지: 약 4.6 GPU-hours.
- 실제 총시간은 collection/pruning/loading을 더해야 하며, 새 GPU의 첫 측정에서 ETA를 갱신한다.

GPU 1개면 순차 실행한다. GPU 2개가 확보되면 probe block을 0–15/16–31로 나누어 각 GPU가 독립 dense 모델을 보유하고 동일 Uniform 배경에서 평가한다. 파일은 block별로 분리하고 coordinator가 receipt 확인 후 합친다. Dense cache는 한 GPU에서 생성·검증한 뒤 공유한다. Probe 중 공용 모델/weight 상태를 동시에 수정하지 않는다.

최종 mini 평가는 후보 단위로 나누어 실행한다. 비교 쌍의 우선순위는 Multi/Short → Path/All → A다. GPU slot은 실행 시 실제로 확보된 장치로 지정한다.

## Actual Setup

아직 없음. 모델을 로드하거나 GPU 실험을 시작하지 않았다.

## Progress / Notes

- [x] 최신 이론 설계와 실제 기존 runner/config/result 확인.
- [x] 5개 objective, 50% backend, bank 크기, 평가 예산 결정.
- [x] 기존 source hash, phase grid, confirmation original IDs를 JSON에 기록.
- [ ] Node-bank 및 subset-evaluation adapter 구현.
- [ ] 실제 입력/prompt manifest freeze와 CPU 검증.
- [ ] Cache/mask 재현 검증 및 GPU smoke check.
- [ ] 실험 시작 시 Obsidian Experiments 노트 작성, status=running 및 시작 시각 기록.
- [ ] tmux/프로세스 중복 확인 후 전용 tmux session에서 실행.

진행 표시에는 stage, method/block, completed/total, elapsed, ETA, GPU, log path를 포함한다. 새 runner의 status 명령과 tmux 이름은 구현 시 확정하고 실제 동작 확인 후 제공한다. 현재 존재하지 않는 실행 명령을 ready-to-run이라고 안내하지 않는다.

## Results

새 실험 결과 없음.

## Interpretation and Decision

| 관측할 패턴 | 해석 및 결정 |
|---|---|
| Multi와 Short의 physical mask 동일 | 이번 allocation backend에서 차이 소실. 중복 평가 중단; 목적함수 전체를 기각한 것은 아님 |
| A가 Multi와 같거나 더 좋음 | 새 bank/endpoint 보존만으로 충분할 수 있음; multiscale 효과 주장 보류 |
| Path 또는 All이 Multi와 같거나 더 좋음 | 더 단순한 설계를 우선. Multi 고유의 필요성이 지지되지 않음 |
| Calibration response만 개선 | Calibration 적합 또는 probe 근사 문제 가능. Heldout/joint/task를 분리 확인 |
| Heldout response 개선, GSM 악화/무개선 | 현재 scalar/gold-reveal proxy와 생성 성능 연결이 약함. 더 많은 full 평가로 해결하려 하지 않음 |
| 새 대조군보다 좋지만 기존 A+C보다 나쁨 | 제한된 bank 내 이점은 가능하나 실용적 업그레이드 아님 |
| Development 개선, confirmation에서 재현 안 됨 | 결론 보류. mini/seed 선택 효과 가능; 즉석 튜닝 후 같은 확인 set을 다시 독립 검증으로 취급하지 않음 |
| Confirmation에서도 강한 대조군과 기존 A+C보다 개선 | 후속 연구를 진행할 근거. 보편적 우월성·새 수학 정리·DLM 전용성의 증명은 아님 |

성공 여부를 “61 또는 62점을 넘겼다” 하나로 판단하지 않는다. 새 bank 효과, C 추가 효과, scale 구성 효과, 실제 기존 방법 대비 효과를 각각 분리한다.

## Next Experiment

이번 단계 이후 자동 실행할 실험은 없다. 성공 시 다음 후보는 calibration seed 재현성 또는 generated-context 진단이다. 실패 시 위 표에서 확인된 병목에 맞춰 설계를 바꾸고 새 계획으로 기록한다.

## Related Notes / Obsidian sync

- 이론: `research/ac_multiscale_monotone_design_2026-09-22.md`.
- 기존 mini 결과: `research/two_mini50_completion_review_2026-09-17.md`.
- 기존 구현: `experiments/dlm_context_response50/core.py`, `run.py`, `config.json`.
- Allocation/진단 clean span 및 cache source의 절대 경로와 SHA256은 동명의 계획 JSON 참조.
- 직전 이론 노트는 Obsidian `Research/DLM-Pruning/Hypotheses/2026-09-22-Multiscale-AC-Monotone-Reveal-Design.md`에 저장·조회 성공한 상태다.
- 이번 계획은 **로컬 저장, Obsidian 반영 대기**다. 현재 catalog에 `mcp__obsidian__get_sync_status`, `mcp__obsidian__read_note`가 없다. get_sync_status 호출의 원문 오류: `TypeError: tools.mcp__obsidian__get_sync_status is not a function`. read_note는 callable 부재로 실행하지 못했다. Catalog를 다시 확인했으나 호출 가능한 함수가 없어 실제 MCP 재시도는 불가능했다. 이는 서버 연결 단절을 관측한 것이 아니다.
- Tool이 다시 노출되면 연결 프로토콜을 실행하고, 이 계획을 status=planned인 실험 노트로 동기화한 뒤 조회 검증한다. 실제 launch 시에만 running으로 전환한다.
