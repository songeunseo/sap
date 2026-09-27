# 다음 실험 제안: pruning 손실은 병렬 생성에서 더 커지는가?

Date: 2026-09-27  
Status: proposed; no new model forwards, masks, or experiment launched.  
Current priority: 다음 실험 선택. 집필 작업은 보류한다.

## 결정

**기존 dense 모델과 A의 exact50% mask를 고정하고, 256/64/32 denoising steps에서 pruning으로 인한 정확도 손실을 비교한다.** 생성 길이와 block length는 모두 256으로 유지한다. 이는 한 번에 확정하는 토큰 수 1/4/8을 비교하는 여섯 조건의 개발 실험이다.

이 실험은 새로운 allocation 방법 자체가 아니다. 다음 allocation이 해결할 손상을 현재 모델에서 확인하기 위한 첫 비교다. 첫 단계에서 early-layer 보호, timestep 가중치, 새로운 C 계수를 동시에 도입하지 않는다. 새 방법의 개발에 언제나 이런 진단이 선행해야 한다는 일반 규칙도 아니다.

## 현재 증거

- 완료된 cross-chain 비교의 primary1119 정답 수: A 646, Multi 636, Cross 637, CrossMatched 634. Multi−A는 −0.894%p, gain/loss 64/74, exact paired p=.4437, 사전 지정 세 비교의 Holm p=1. 추가 C의 이득과 natural pairing의 우위는 확인되지 않았다. 동등성이나 모든 conditional-response 방법의 실패를 입증한 결과는 아니다.
- fresh diagnostic의 natural C는 A .868318, Multi .867225로 매우 가깝고, Cross .835103 및 CrossMatched .832330이 더 낮다. scalar C를 줄이는 것과 정확도 사이의 인과관계는 이 결과에서 확립되지 않았다.
- 현재 `generate.py`는 length=steps=block_length=256, temperature=0에서 매 step 한 토큰을 확정하고 이미 보이는 토큰은 복사한다. 따라서 여러 토큰을 동시에 확정하는 데서 오는 오류를 현재 결과의 원인으로 소급해 설명할 수 없다.
- 기존 role/timestep 연구도 재활용해서 새 발견으로 부르지 않는다. `experiments/dlm_dual_role_allocation/decision.json`은 NOT SUPPORTED이며, role/aggregate marginal reconstruction의 Spearman은 .99098이다. 이는 해당 구현/진단의 결과이며 모든 role 방법의 부정은 아니다.

결과 출처: [verified closeout](/home/tmluser1/sap/writing/dlm_pruning/closeout.json), [frozen run config](/home/tmluser1/sap/experiments/dlm_crosschain_control50/output/config.json), [decoder](/home/tmluser1/sap/generate.py).

## 외부 논문에서 가져올 수 있는 것

| 논문 | 확인한 근거와 적용 범위 |
|---|---|
| [Confidence Shortcut](https://arxiv.org/html/2605.29123) | 약 0.4M–21M의 task-specific 모델에서 높은 confidence와 논리적 선행조건의 해결 순서가 어긋난다. LLaDA-8B의 parameter importance나 safe-to-prune 신호를 직접 검증한 논문은 아니다. |
| [ADAS, v4](https://arxiv.org/html/2606.10829v4) | LLaDA-8B-Base의 GSM8K Top-k 결과에서 k=1은 69.98→69.98, k=4는 53.90→50.11, k=8은 22.67→42.46이다(Table 19). 이 논문의 prompt 등은 우리 설정과 다르다. 논문 점수를 우리 dense baseline으로 대체하지 않는다. 의존성을 고려한 decoding 개선은 조건에 따라 달라지며, static pruning의 개선 증거는 아니다. |
| [Parallelism and Generation Order](https://aclanthology.org/2026.findings-acl.357/) | 8개 모델·58개 benchmark에서 parallelism과 생성 순서의 제한을 분석한다. 같은 NFE라도 task/model별 차이가 있으므로 우리 설정에서 직접 비교할 근거가 된다. 이 논문과 Induction은 9월 18/22일 검토에 이미 포함됐다. |
| [Mechanism Shift, v5](https://arxiv.org/html/2601.14758v5) | 두 7B ARM–MDM 모델군의 네 통제 과제에서 task별 회로 재구성이 다르다. global 과제의 early-layer 결과를 LLaDA-8B 전체의 고정 보호 규칙으로 옮길 수 없다. |
| [Induction in Both Directions, v2](https://arxiv.org/html/2607.15893v2) | 작은 attention-only 모델의 양방향 induction과 mask-rate 개입 증거다. mask fraction을 통제·기록할 이유는 되지만 timestep-weighted allocation의 성능을 보장하지 않는다. |
| [Time Is a Feature](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76931eaba1fcb55b70cde7d0de0161ef-Abstract-Conference.html) | 중간에 완성한 후보 답의 변동과 temporal voting/RFT를 연구한다. 아직 masked인 위치의 임시 예측 변화와, 이미 확정한 토큰을 실제로 덮어쓰는 사건을 구분해야 한다. 우리의 decoder는 visible token을 복사한다. |

원문 범위 검토는 같은 디렉터리의 `dependency-review.md`, `mechanism-review.md`에 별도 보관한다. 두 검토자의 실험 제안과 여기의 최종 우선순위가 다르면 이 문서가 현재 제안이다.

## Hypothesis

**H1:** 동일한 static mask라도 한 번에 더 많은 토큰을 확정하면 dense 대비 정확도 손실이 커진다.

**H2, 보조 가설:** 일부 손실은 새 문맥을 받은 뒤 예측을 갱신하는 반응의 손상과 관련된다. H1만으로 H2가 증명되지는 않는다. confidence calibration, 일반적인 모델 품질 저하, 서로 다른 생성 궤적도 H1을 설명할 수 있다.

**후속 방법 가설:** 실제 사용되는 문맥에 대한 반응을 보존하는 allocation이 동일 비용의 full-distribution fidelity allocation보다 유리할 수 있다. 아직 검증되지 않았고 이번 비교에서 새 mask를 만들지는 않는다.

## Planned Setup

| 모델/고정 mask | 256 steps: 1 token/step | 64 steps: 4 tokens/step | 32 steps: 8 tokens/step |
|---|---|---|---|
| Dense LLaDA-8B-Base | 평가 | 평가 | 평가 |
| A exact50% | 기존 결과 재사용 가능 | 평가 | 평가 |

- Revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, BF16, 같은 tokenizer와 모델 파일.
- A mask identity `fd22875b43d65e22ba2c0091b0f9b552e5f73d524cdae388f73030075c0c1d10`, sparse model SHA256 `7d5dc817890f1079a6a8f2970088c7bee80afa60c11ebea30cb435d14d40922c`.
- A는 현재 endpoint-only 방법을 대표하는 고정 비교 대상이다. 이 결과를 모든 50% mask의 성질로 일반화하지 않는다.
- 3,489,660,928 / 6,979,321,856 prunable weights 제거. surviving weights, Wanda ranking, layer allocation 모두 그대로 둔다.
- 5-shot prompt, generation length=256, block length=256, temperature=0, cfg=0, confidence Top-k, strict-match grading을 유지한다. 바뀌는 decoding 변수는 steps 하나다. seed도 기존 설정을 유지한다.
- 첫 비교는 현재 config의 `crosschain_control.exposed_ids` 200개와 저장된 prompt를 그대로 사용한다. 이미 노출된 개발 자료임을 표·출력에 표시한다. 과거 primary1119도 이제 결과가 관측됐으므로 새 confirmation set이라고 부르지 않는다.
- 후속 방법 선택에는 독립 calibration 문서를 사용한다. 최종 검증 집합은 학습·calibration·few-shot·개발 노출 이력과 중복을 확인한 뒤 별도 고정한다. 현재 제안은 독립 최종 평가를 완료한 것으로 간주하지 않는다.
- NFE가 바뀌면 새 protocol hash와 별도 출력 디렉터리를 만든다. 기존 실험 config, 결과, audit receipt는 수정하지 않는다.
- 기존 256-step 결과는 모델/mask, doc/prompt/target/tokenizer, decoding 및 grading identity가 모두 같은 경우에만 재사용한다. dense 결과의 존재나 일치 여부는 아직 확인하지 않았다.

## Results and Interpretation: 측정할 것

### 1. 핵심: paired accuracy interaction

`gap(T) = accuracy(Dense,T) - accuracy(A,T)`로 놓는다.

주 비교는 `I32 = gap(32) - gap(256)`이다. 64-step은 중간 정도의 병렬 생성에서 방향이 유지되는지를 보여주는 보조 비교다.

- 여섯 조건의 절대 정확도, 각 T에서의 paired gain/loss, I32를 함께 보고한다.
- 같은 문제의 네 correctness 값을 묶어 I32의 95% paired bootstrap 구간을 계산한다. 문제를 resampling unit으로 삼고 토큰이나 step을 독립 표본으로 세지 않는다. bootstrap seed=20260927, draws=10000을 제안한다.
- 개발 자료에서 나온 구간은 탐색적 불확실성 요약이다. 200문항은 작은 효과를 명확히 구분하지 못할 수 있다. 유의하지 않다는 이유만으로 동등하다고 판정하지 않는다.
- dense와 sparse가 모두 32-step에서 바닥 성능이면 gap은 오히려 줄어들 수 있다. 절대 정확도 및 64-step을 함께 읽고, 사후에 가장 좋아 보이는 NFE를 primary로 바꾸지 않는다.
- NFE와 실제 latency/forward tokens를 따로 기록한다. unstructured 50% mask 자체의 GPU speedup은 이 비교로 보장되지 않는다.

### 2. 작은 보조 측정: 동일 문맥에서 토큰 하나를 공개했을 때의 반응

H1 결과를 곧바로 dependency의 인과 효과라고 부르지 않기 위한 보조 자료다. 여섯 조건 평가를 이 진단의 성공에 종속시키지 않는다.

1. Dense의 32-step 궤적에서 step 0/8/16/24 직전 상태를 저장한다. sparse 모델의 자체 궤적과 비교할 때 생기는 입력 차이를 없애기 위해 D/A에 똑같은 상태를 넣는다.
2. Dense confidence Top-8을 고정한다. 그중 첫 위치 i의 dense argmax 토큰 하나만 counterfactual copy에서 공개한다. 나머지 일곱 위치 j는 계속 masked로 둔다. 순위·token 선택에 sparse 결과나 정답을 사용하지 않는다.
3. D/A의 공개 전후 full-vocabulary distribution을 측정한다. gold-logodds 하나로 압축하지 않는다. 모든 후보에 동일한 공개 토큰과 위치를 적용한다.
4. 문맥 반응 크기 `||p_D(after)-p_D(before)||_1`, 반응 오차 `||(p_A(after)-p_A(before))-(p_D(after)-p_D(before))||_1`, 전후 각각의 `KL(p_D||p_A)`, argmax 변화/불일치를 함께 기록한다. probability readout은 FP32로 계산한다.
5. 문제별 평균을 먼저 구하고 mask fraction과 공개 전 dense confidence별로 결과를 표시한다. 작은 반응도 수치 잡음 없이 보고하고, 결과를 본 뒤 어려운 토큰만 골라 강조하지 않는다.

이 측정은 functional response fidelity다. 자유형 reasoning의 중간 token이 정답인지, 반응 오차가 최종 오답을 발생시켰는지는 별도로 입증해야 한다. 전후 mask 수 변화는 두 모델에 동일하지만 semantic reveal과 progress 효과가 완전히 분리되는 것은 아니다. 단순 response error 감소는 endpoint fidelity의 개선만으로도 나타날 수 있다.

## 비용 범위

- 캐시를 사용하지 않을 때 여섯 조건의 generation: `200 × 2 × (256+64+32) = 140,800` forward calls.
- 기존 A/256의 200개 결과를 전부 검증 후 재사용하면 새 generation은 89,600 calls. 일치하는 Dense/256 캐시가 있으면 추가로 줄일 수 있으나 아직 계산에 반영하지 않는다.
- 보조 진단 800개 상태 × D/A × 전후 두 상태 = 최대 3,200 calls. Dense before logits를 저장·재사용하면 줄어든다. full-vocabulary tensor를 계속 보관할 필요 없이 집계값과 재현 가능한 상태를 저장한다.
- 위 수는 batch size 1의 model-call 계산이며 측정된 runtime이 아니다. 로딩, mask 재구성·검증, I/O, 계측 검증 비용은 별도로 기록한다. GPU 처리량을 확인하지 않고 소요 시간을 단정하지 않는다.

## Decision: 결과에 따라 다음에 할 일

1. **낮은 NFE에서 pruning 손실이 뚜렷하게 커짐:** low-NFE에 맞춘 calibration/allocation을 실험할 근거가 생긴다. dependency 반응도 손상되는지 보조 측정을 읽되, 정확도 interaction만으로 메커니즘을 확정하지 않는다.
2. **Dense와 A가 비슷하게 나빠짐:** 이 비교에서 pruning 특유의 병렬 생성 취약성은 확인되지 않았다. 일반 decoder의 한계와 구분하고, 이 가설의 우선순위를 낮춘다. 넓은 구간이면 미결이다.
3. **반응 오차가 커도 최종 정확도와 맞지 않음:** response 크기 자체를 importance라고 부르지 않는다. 이런 지표를 줄이는 새 C 항만으로 다음 full run을 정당화하지 않는다.
4. **후속 allocation을 시험할 경우:** 실제 상태와 dense가 고른 reveal을 사용하되, full-distribution endpoint KL과 response를 넣은 목적을 동일 ranking, exact budget, 후보 수/탐색 비용으로 비교한다. 이때 mask/calibration/readout/optimizer를 한꺼번에 바꿔 원인을 잃지 않는다. 독립 NELBO와 downstream 성능을 측정한다.
5. **ADAS는 그다음 decoder 대조군:** 먼저 기본 여섯 조건을 측정한다. 후속으로 32-step Dense/A 양쪽에 같은 ADAS를 적용할 수 있다. 양쪽이 비슷하게 개선되면 decoder의 이득이며 pruning-specific allocation의 근거는 아니다. 논문과 달리 우리 설정에서 효과가 없을 가능성도 남긴다.

## 실행 상태와 역할

- Master: 현재 결과·decoder를 대조하고 이 제안을 선택했다.
- Experiment agent: 원본 experiment.md 및 프로젝트 역할 지침에 따라 Confidence Shortcut/ADAS/Parallelism의 증거와 비교 설계를 검토했다.
- Writing agent: 원본 writing.md 및 프로젝트 역할 지침을 주장·원문 검증에 적용해 나머지 세 논문의 전이 범위를 검토했다. 현재 역할은 집필 진행이 아니다.
- Actual Setup / Results: 새 실행 없음. 이 문서는 실행 계약이나 성공 결과가 아닌 제안이다. 실행을 시작할 때 별도 실험 노트, 중복 프로세스 확인, tmux 및 checkpoint 절차를 적용한다.

## 고정 출처

- Source config SHA256: `59912694489062d4d7c08dece821c36ad292f95a4b42a7603851b7d14347d06b`.
- Source requests SHA256: `3ed3e16ecd93ab9081c40a9bd523f09e6bffb8b6da7f78803b7a729609d4f5a8`.
- A mask manifest SHA256: `5afb660b8a005bfb7f9ec3f630de54fe13e0d56dee8b7da66c683b857903b7bc`.
- 역할과 source review는 `research/next_experiment_2026-09-27/`에 보관한다. 이전 `writing/dlm_pruning/paper-v1.md` 작업의 완료를 의미하지 않는다.
