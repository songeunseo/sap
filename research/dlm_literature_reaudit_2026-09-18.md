# DLM pruning 문헌 재감사 및 기존 실험 재해석

- 조사일: 2026-09-18
- 목적: 정적 post-training 50% weight allocation/pruning 문제에서 A+C의 위치와 실제 선행 겹침을 다시 판정
- 범위: DLM을 대상으로 한 weight pruning/quantization/compression, 그리고 같은 DLM 고유 신호를 이용하는 inference/cache/decoder 방법
- 제약: 새 실험/GPU 실행 없음. 아래의 문헌 사실과 프로젝트 관측 결과를 분리한다.

## 판정 요약

1. **DLM 전용 알고리즘 자체는 충분히 novelty가 될 수 있다.** 모든 DLM 방법이 AR 비교를 해야만 하는 것은 아니다. Sink-Aware처럼 DLM의 attention sink 통계를 pruning 입력에 넣거나, Quant-dLLM처럼 timestep/mask-aware calibration과 blockwise mixed precision을 결합하면 알고리즘적 기여가 가능하다. 다만 “DLM에 Wanda를 적용했다” 또는 “반복 denoising이 있으므로 누적 오차를 보자”만으로는 부족하다.
2. **A+C의 현재 증거만으로는 새 DLM pruning algorithm이라고 주장할 수 없다.** A+C는 static pruning mask를 만들기 위한 `gold-logodds endpoint error + paired response error` 후보 proxy다. 실제 rollout KL도 아니고, decoder commit harm도 아니다. 특히 FAIR-Calib가 이미 DLM의 irreversible commit frontier, downstream amplification, teacher probing, off-policy calibration을 직접 다룬다. 따라서 A+C에 commitment-value framing을 덧씌우면 novelty 주장이 약해지지만, 기존 allocator와 결합한 새 DLM signal이 실험적으로 예측력과 이득을 보이면 알고리즘 기여가 될 가능성까지 배제할 수는 없다.
3. **직접적인 50% unstructured weight allocation 경쟁자 수는 cache/attention 논문보다 훨씬 적다.** [Sparse-dLLM](https://arxiv.org/html/2508.02558), [SparseD](https://arxiv.org/html/2509.24014), [dLLM-Cache](https://arxiv.org/html/2506.06295), [Focus-dLLM](https://aclanthology.org/2026.acl-long.556/), [DARE](https://arxiv.org/html/2605.08134), [Window-Diffusion](https://arxiv.org/html/2601.20332) 등은 token/attention/KV/activation compute를 줄이는 방법이다. 이들은 DLM-specific temporal signal의 선행으로는 중요하지만, static weight mask의 직접 비교군으로 부르면 안 된다.
4. **프로젝트 내부 결과도 Uniform이 보편적 최강 baseline이라는 전제를 지지하지 않는다.** 현재 기록상 PPL50에서 DSA가 Uniform보다 낮은 NELBO를 보였고, GSM8K historical 결과에서도 구조적 allocation이 Uniform을 앞섰다. 따라서 “AR-derived 방법은 모두 Uniform보다 나쁘다” 또는 “DLM에는 Uniform만 강하다”를 주장하면 안 된다.
5. 가장 방어 가능한 후속 포지션은 “DLM-specific static allocation” 자체를 넓게 주장하는 것이 아니라, **기존 signal과 구별되는 scalar/allocator가 exact budget에서 reproducible improvement를 내는지**를 검증하는 것이다. A+C는 현재 그 기준을 충족했다고 볼 수 없다.

## 먼저 고정할 문제 정의

프로젝트의 A+C probe는 shared masked query에서 gold context reveal 전후의 gold conditional log-odds endpoint를 비교한다. `e_-`, `e_+`를 각각 endpoint log-odds error라고 두면 기록된 정의는

\[
A=E[(e_-^2+e_+^2)/2],\qquad C=E[(e_+-e_-)^2],\qquad J=A+C.
\]

오차를 평균 성분과 변화 성분으로 `e_-=u-v`, `e_+=u+v`로 쓰면 `A=E[u^2+v^2]`, `C=4E[v^2]`, 따라서 `A+C=E[u^2+5v^2]`이다. 이것은 **gold-vs-rest log-odds의 endpoint와 paired response를 합친 정적 probe**이지, sparse model의 multi-step rollout KL이나 NELBO가 아니다. x_-→x_+가 gold 내용과 mask count를 함께 바꾸는 설정이면 순수한 context-utilization intervention이라고도 단정할 수 없다.

또한 gold 확률이 고정되어도 wrong-token mass의 배치가 바뀌면 선택 action이 뒤집힐 수 있다. 예를 들어 gold 확률이 .4로 고정된 채 wrong distribution이 (.35,.25)에서 (.55,.05)로 바뀌면 gold-vs-rest log-odds는 같아서 A와 C가 0일 수 있지만, 다른 position과의 relative confidence ranking은 바뀐다. 이 반례는 A+C의 정보 손실을 보이는 수학적 반례이지, 실제 빈도나 downstream harm의 측정은 아니다.

## 가장 가까운 12개 문헌

아래에서 **직접**은 static weight/quantization budget을 바꾸는 경우, **인접**은 DLM 고유의 temporal/commit 신호를 이용하지만 weight mask가 아닌 경우다. 각 행은 원문에서 확인한 알고리즘과 A+C와의 충돌을 함께 적는다.

| 문헌 | 구분 및 실제 알고리즘 | A+C와 겹치는 지점 | 남는 차이/공백 |
|---|---|---|---|
| [Sink-Aware Pruning for Diffusion Language Models](https://arxiv.org/html/2602.17664) | **직접**. noised calibration timestep의 attention sink soft score를 평균해 `omega_j=1-bar(phi)(j)`로 activation row를 재가중하고, 그 activation을 Wanda/SparseGPT pruning에 넣는다. temporal centroid variance는 동기/진단이고, pruning criterion 자체는 평균 soft sink reweighting이다. | DLM-specific calibration signal을 기존 weight pruning score에 주입한다는 점. | A+C의 gold-logodds endpoint나 paired reveal을 쓰지 않으며, unit별 commit value/Q도 만들지 않는다. 50% LLaDA/Dream 등에서 baseline 대비 성능을 비교하는 실제 direct competitor다. |
| [Layer Collapse in Diffusion Language Models](https://arxiv.org/html/2605.06366) | **직접에 가까운 allocation baseline**. LLaDA-8B의 early-layer collapse/super-outlier를 분석하고 동일 평균 sparsity에서 early-is-sparser(EIS)와 deeper-is-sparser(DIS)를 비교한다. 50%에서 LLaDA EIS가 reverse strategy보다 +8.4%라는 보고가 있다. | DLM training dynamics가 layerwise compression allocation을 바꾼다는 주장. | 새 per-weight objective라기보다 구조 분석과 allocation schedule이다. “DLM에서는 early layer를 더 prune한다”는 일반 novelty는 이미 충돌한다. 프로젝트의 EIS+type은 이 논문과 구별되는지 확인해야 한다. |
| [Quant-dLLM](https://arxiv.org/html/2510.03274) | **직접**. timestep/mask-aware Masked Calibration Simulation(MCS), multi-binary Data-aware Any-order Quantizer(DAQ), strict 2-bit average budget의 Adaptive Blockwise Mixed Precision(ABMP)을 결합한다. block importance에 따라 1/2/3-bit order를 배정한다. | DLM의 mask/timestep activation distribution을 static weight compression calibration과 allocation에 직접 연결한다. “native DLM calibration이 필요하다”는 넓은 주장과 충돌한다. | 주 목적은 2-bit weight-only quantization이고, A+C의 unstructured 50% binary pruning과는 compression operator가 다르다. 그래도 DLM-specific calibration+budget allocation novelty의 기준을 낮춘다. |
| [FAIR-Calib](https://arxiv.org/html/2606.06547) | **직접에 가까운 PTQ**. teacher probing으로 frontier-hit와 masked-stage reliability를 합친 static position prior를 만들고, off-policy fully-observed teacher-forcing에서 weighted hidden-state MSE로 W4A4 PTQ를 보정한다. commit flip과 post-commit mismatch를 측정한다. | “반복 refinement→irreversible commit→오차 증폭”이라는 DLM-specific mechanism을 직접 다룬다. frontier/commit value와 downstream amplification을 명시하고, rollout 없이 teacher-forcing surrogate까지 제공한다. | quantization calibration이며, 50% static unstructured weight mask와 unit-level budget allocator는 아니다. 그래도 commitment-aware novelty의 가장 강한 collision이다. A+C가 이와 다른 점은 gold-logodds endpoint일 뿐이며, 그것만으로 downstream harm을 보장하지 못한다. |
| [Sparse-dLLM](https://arxiv.org/html/2508.02558) | **인접**. dynamic cache eviction과 delayed bidirectional sparse caching으로 prefix/suffix를 cache하고, block transition에서 갱신한다. | Denoising step 사이의 persistent token importance와 dynamic pruning이라는 DLM-specific intuition. | attention/KV cache inference method; weight pruning이 아니며 static 50% mask와 직접 비교할 수 없다. |
| [dLLM-Cache](https://arxiv.org/html/2506.06295) | **인접**. static prompt는 긴 간격으로 cache하고 response는 짧은 간격으로 refresh하며, adjacent-step Value cosine similarity(V-verify)로 partial update token을 고른다. | temporal feature stability를 pruning/reuse decision에 쓴다. | cache/update scheduling이며 weight allocation이 아니다. A+C를 temporal consistency의 최초 metric이라고 부를 수 없다. |
| [SparseD](https://arxiv.org/html/2509.24014) | **인접**. head-specific blockwise attention pattern을 한 번 precompute하고 이후 denoising step에 재사용한다. early steps는 full attention으로 두어 quality loss를 막는다. | DLM-specific head/time structure와 early-step sensitivity를 algorithm에 반영한다. | attention sparsity/reuse이고 weight mask가 아니다. 특히 “DLM temporal sparsity structure를 처음 활용” 같은 포괄적 표현은 피해야 한다. |
| [Focus-dLLM](https://aclanthology.org/2026.acl-long.556/) | **인접**. past confidence로 아직 unmasked인 region을 예측하고 sink-aware attention sparsification과 cross-layer sink reuse를 한다. | confidence, unmasked region, sink를 이용해 DLM의 미래 상태를 추정한다. | long-context attention compute pruning이며 static parameter pruning과 무관하다. |
| [DARE](https://arxiv.org/html/2605.08134) | **인접**. successive-step query drift를 token/layer temporal redundancy proxy로 써서 DARE-KV와 DARE-O activation reuse budget을 layerwise 배분한다. | temporal drift→layerwise resource allocation이라는 패턴. | KV/output reuse이고 weight budget이 아니다. A+C가 temporal response-aware allocation의 최초라는 주장을 약화한다. |
| [Efficient Token Pruning for LLaDA-V](https://arxiv.org/html/2601.20168) | **인접/멀티모달**. LLaDA-V에서 cross-modal aggregation이 middle-to-late layer에서 일어난다는 분석을 바탕으로, 첫 denoising step의 지정 layer 이후 visual token을 영구 제거한다. | DLM의 spatial×temporal execution structure를 pruning 위치에 연결한다. | visual token/FLOPs pruning이며 base text model weight allocation이 아니다. |
| [Order-Token Search](https://arxiv.org/html/2601.20339) | **인접**. generation order와 token value를 joint beam search하고, newly revealed block의 DLM likelihood estimator로 partial trajectory를 prune한다. | action likelihood/trajectory-level selection을 정식 objective로 쓴다. | decoder order search이고 static weight pruning이 아니다. “order-aware objective가 새롭다”는 주장과 충돌한다. |
| [OPTD: On-Policy Transition Distillation](https://arxiv.org/html/2608.02942) | **인접/transition distillation**. student on-policy masked states에서 teacher의 future candidates를 비교해 outcome-preserving longest prefix를 찾고, set-bottleneck certainty와 teacher KL로 few-step distillation을 한다. | on-policy state, trajectory preservation, teacher KL을 사용한다. | student training/step reduction이며 post-training weight allocation이 아니다. |

### 추가 충돌 확인

- [TA-GRPO-d](https://aclanthology.org/2026.acl-long.1723/)는 trajectory-level AUC z-score와 token unmasking-time weight를 RL reward에 넣어 confidence-gated denoising을 학습한다. trajectory+unmasking time 자체는 새롭지 않다.
- [Scheduling Thoughts](https://arxiv.org/html/2606.23567)는 reveal decision별 reward와 Self-Aware Scheduling을 학습하고, sequential decoding mismatch KL bound를 제시한다. “DLM은 반복하므로 pathwise KL을 보자”만으로는 novelty가 아니다.
- [Consistent Diffusion Language Models](https://arxiv.org/html/2605.00161)는 multi-path discrete consistency와 posterior bridge로 path-independent denoiser를 학습한다. 단순 path/order consistency 역시 선행과 충돌한다.
- [COPSD](https://aclanthology.org/2026.findings-acl.1344/)는 later decoding step의 unnormalized target을 early prediction에 self-distill하여 early token prediction을 보정한다. future context를 이용한 early calibration은 이미 있다.
- [DARE](https://arxiv.org/html/2605.08134)와 [SparseD](https://arxiv.org/html/2509.24014)는 각각 temporal activation reuse와 head-specific attention reuse를 통해 DLM 고유 temporal structure를 algorithm에 넣는다. 둘 다 weight pruning은 아니지만 DLM-specificity 주장에 대한 선행이다.
- `dPrune`라는 이름으로 검색된 결과 중 확인 가능한 것은 DLM weight pruning 논문이 아니라 unrelated data-pruning package였다. 검증되지 않은 dPrune을 competitor로 인용하지 않는다.
- OpenReview identifier `HD7tuVakmR`는 현재 primary page에서 제목을 검증하지 못했다. ICLR 2026 목록에는 DLM 관련 `FlashDLM`, `ES-dLLM`, `SparseD` 등이 보였지만 identifier와 제목을 일치시키지 못했으므로 이 메모의 주장에 사용하지 않는다.

## 일반 compression 선행이 주는 기준

[EvoPress](https://arxiv.org/html/2410.14649)는 layer/block별 여러 compression level을 준비하고, global compression budget 아래에서 base-model KL fitness를 사용해 evolutionary joint search를 한다. 따라서 “여러 layer에 독립 점수를 매겨 Uniform 대신 budget을 배분한다” 자체는 LLM compression에서 이미 established pattern이다. 프로젝트의 DSA도 PPL50에서 Uniform보다 좋은 NELBO를 기록했으므로, static joint allocation을 실제로 잘하는 것이 더 중요하지 novelty의 자동 증거는 아니다.

또한 [LSA 공식 ICLR 2026 원문](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b805585c7e249c1f65737506d4fe1e4-Paper-Conference.pdf)처럼 per-layer linear reconstruction-error criterion으로 sparsity allocation을 정하는 일반 pruning 선행이 있다. 이는 최종 end-to-end error를 직접 최적화한다는 뜻으로 읽으면 안 된다. DLM이므로 layer score가 달라질 수 있다는 관측은 가능하지만, 기존 layer/block allocation과 구별되는 DLM-native signal과 exact-budget benefit을 보여야 한다.

## 프로젝트의 기존 결과를 다시 읽은 판정

아래는 문헌 fact가 아니라 저장소/Obsidian에 기록된 **프로젝트 관측 결과**다.

- frozen validation PPL50 기록: DSA NELBO `2.530142`, Uniform `2.550805`, OWL `2.569351`, LSA layer `2.576876`, LSA projection `2.579124`, Alpha `2.670538`, DLP `2.698444`. DSA–Uniform 차이는 `-0.0206634`, 기록된 MCSE는 `0.00010649`이나 이는 MC-draw uncertainty만 반영하며 최종 독립 검증이 아니다.
- layer-global Uniform `2.663777`가 projection-rowwise Uniform `2.550805`보다 나빴다. 따라서 “Uniform”은 granularity에 따라 다른 baseline이고, allocation unit을 명시하지 않으면 비교가 불가능하다.
- historical Full GSM8K: Uniform 139/1319, Aggregate 248, Oracle 250, historical reconstruction 255, EIS+type 263, Role 268. Role–Aggregate의 기록된 p는 `.08498`, Role–EIS+type은 `.79755`. 구조적 allocation이 Uniform을 앞서는 사례는 있지만 모든 차이가 유의하다고 할 수 없다.
- A+C mini100: Uniform 54, A 55, A+C 61, 기록된 A+C 대 A paired p=`.145996`. historical cached DLM-Wanda50은 같은 100/protocol에서 62였다. 이는 end-to-end 서술값으로는 참고할 수 있지만, ranking/mask construction과 세부 allocation 차이가 있어 A+C signal 자체의 인과 효과나 열등함을 분리해 말할 수 없다.
- C1 coverage는 54, pooled는 50, Uniform은 54였고 established improvement가 없다. coverage를 weight capacity나 actual harm에 연결하는 인과 주장은 현재 근거가 없다.
- A/AC full evaluation은 A 107/AC 147 generation에서 중단되어 결과가 없다. 따라서 A+C superiority 또는 failure를 full-scale 결과로 쓰면 안 된다.

이 결과는 다음을 의미한다.

1. AR-derived baseline이 자동으로 약하다는 주장은 삭제한다. DSA와 EIS+type이 이미 반례다.
2. DLM-specific algorithm이 필요하다는 결론은 유지되지만, DLM-specificity는 모델 이름을 바꾸는 것으로 얻어지지 않는다. Sink-Aware/Quant-dLLM/Layer Collapse처럼 signal, operator, 또는 allocation mechanism이 명시되어야 한다.
3. A+C mini 결과의 61은 후보 proxy의 가능성을 보이는 탐색 결과일 뿐, static allocation algorithm의 established gain이 아니다.

## A+C의 novelty와 claim 경계

### 주장할 수 있는 좁은 내용

- shared masked query에서 context reveal 전후 **gold conditional log-odds endpoint error와 paired response error를 합친 후보 scoring signal**을 정의했다.
- 이 signal이 기존 A만 쓰는 ranking과 다른 mask를 만들 수 있는지 탐색했다.
- DLM의 bidirectional masked state에서 output response sensitivity를 static weight allocation에 연결하려는 연구 질문은 합리적이다.

### 현재 주장하면 안 되는 내용

- “DLM pruning에서 처음으로 trajectory/commit-aware objective를 제안했다.” FAIR-Calib, OPTD, TA-GRPO-d, Scheduling Thoughts, COPSD가 이미 commit/trajectory/order/early calibration을 각각 다룬다.
- “DLM 반복 denoising의 rollout KL을 최소화한다.” A+C는 rollout KL이 아니다. 실제 sparse rollout KL, NELBO, token/action disagreement를 측정하지 않는다.
- “A+C가 harmful commit을 보장해서 줄인다.” gold-vs-rest log-odds는 다른 wrong-token redistribution이나 position interaction을 놓칠 수 있다. actual-harm 보장은 금지한다.
- “AR 방법은 DLM에서 Uniform보다 나쁘다.” 프로젝트 DSA와 EIS+type 기록이 반례다.
- “첫 DLM static pruning/quantization method다.” Sink-Aware, Layer Collapse, Quant-dLLM, FAIR-Calib가 이미 있다.
- “coverage signal이 weight capacity를 결정한다.” C1은 현재 Uniform과 분리되지 않았다.
- “A+C가 best allocator다.” full A/AC result가 없고 mini difference도 paired p=`.145996`이다.

## 가장 새로울 가능성이 있는 빈자리

문헌을 확인한 범위에서 남는 빈자리는 **DLM의 masked-state response signal을 static, exact-budget, unstructured weight allocation의 per-unit marginal benefit과 연결해 검증하는 것**이다. 다만 이것은 “아직 찾지 못한 정확한 조합”이지 선행 부재의 증명이 아니며, A+C가 구조적으로 부적합하다는 뜻도 아니다. 다음은 필수조건이라기보다 가장 설득력 있는 검증 설계다.

1. A+C처럼 gold endpoint scalar만 써도 실제 allocation benefit을 예측하면 기여가 될 수 있다. 다만 full-vocabulary KL/logit reconstruction과 calibrated action disagreement는 wrong-token redistribution과 action flip을 점검하는 강한 control이므로 함께 두는 편이 안전하다.
2. decoder trajectory/commit state를 사용한다면 FAIR-Calib와 구별되는 단위가 필요하다. FAIR-Calib는 position prior와 PTQ hidden-MSE이고, A+C 후보는 weight-unit ranking이다. 이 차이는 가능한 차별점이지만, 명칭상의 차이만으로 충분하다고 미리 주장할 수 없다.
3. 동일 frozen rowwise Wanda, 동일 exact 50% budget, 동일 calibration/evaluation seed에서 Uniform, A-only, C-only, A+C, DSA, EIS, Sink-Aware를 비교하는 것이 권장된다. 모든 방법을 한 번에 이겨야 한다는 뜻이 아니라, 최소한 어떤 조건에서 이 signal이 유효한지 분리해 보여야 한다.
4. held-out static NELBO와 task correctness/action disagreement를 보거나, 같은 품질에서 비용·견고성 이득을 보고 proxy가 실제 deployment benefit과 맞는지 확인해야 한다. 단일 metric의 개선만으로 downstream value를 단정하지 않는다.

현재는 이 빈자리를 **가능한 연구 질문**으로만 기록한다. “최초” 또는 “causal downstream value”로 표현하지 않는다.

## 강한 저비용 baseline

정적 50% weight pruning을 계속한다면 최소 baseline 묶음은 다음이다.

1. historical cached **DLM-Wanda50** (동일 100/protocol 기록)
2. **rowwise Uniform**와 **layer-global Uniform**을 분리
3. 프로젝트에서 이미 좋은 **DSA**
4. **EIS/EIS+type** layer schedule (Layer Collapse와의 직접 충돌 확인)
5. 가능하면 동일 Wanda backend 위의 **Sink-Aware** reweighting
6. calibration fitness가 필요한 비교로 **LSA/EvoPress-style joint allocation**

이 묶음과 비교해 품질, 비용, 또는 견고성 중 어느 축에서 이득이 있는지 보여야 A+C의 실용적 의미를 판단할 수 있다. 반대로 이 baseline을 이기더라도 “DLM decoder commitment value”의 증명이 아니라, 해당 static protocol에서 allocation signal이 유효하다는 결론만 낼 수 있다.

## 비-commitment 대안 3개

### 대안 A: DLM-native layer schedule + 기존 Wanda

Layer Collapse의 EIS 관측과 프로젝트 EIS+type을 이용해 layer schedule을 고정하고, weight-level novelty를 주장하지 않는다. 구현/검증 비용이 가장 낮고, DLM-specific practical baseline으로 강하다. 단, 이미 EIS가 선행이므로 algorithmic novelty는 낮다.

### 대안 B: full-vocabulary masked-state loss를 쓰는 exact-budget joint allocation

A+C 대신 full-vocabulary KL/logit reconstruction 또는 mask/timestep-stratified output loss를 layer/block/unit score로 사용하고, EvoPress/LSA-style joint budget search와 비교한다. 이는 A+C의 wrong-token redistribution blind spot을 줄인다. 그러나 Quant-dLLM의 masked calibration, FAIR-Calib의 weighted teacher-forcing과 겹치므로 “새 loss” 자체보다는 exact 50% unstructured allocation에 초점을 둬야 한다.

### 대안 C: activation/sink/temporal structure를 allocator 입력으로 직접 결합

Sink-Aware의 평균 soft sink reweighting, DARE의 temporal drift, SparseD의 head/time structure를 static weight score에 넣어 DLM-specific allocation을 만든다. 직접 competitor와 차이가 선명하지만, 이미 공개된 각 signal을 단순히 합치는 것은 novelty가 약하다. 새로움은 결합 자체가 아니라 어떤 weight unit의 marginal loss를 정확히 예측하는지와 exact-budget gain에서만 생긴다.

## 결정 권고

- A+C를 현재 증거만으로 main algorithm novelty라고 쓰지 않는다. `candidate output-response proxy for static allocation`으로 기록하고, positive held-out result가 나오면 새 DLM signal+existing allocator의 알고리즘 기여 가능성을 다시 평가한다.
- commitment-value 방향은 FAIR-Calib와의 overlap 때문에 main claim으로 잡지 않는다. 계속 탐색하더라도 “weight-unit marginal allocator”라는 좁은 차이만 가설로 남기고, actual-harm 보장은 금지한다.
- 가장 먼저 값싼 비교를 할 후보는 DLM-Wanda50, rowwise/layer-global Uniform, DSA, EIS+type, Sink-Aware다. 이는 AR와 DLM의 성능 우열을 가정하지 않고 static task 자체를 평가한다.
- 새 논문 claim은 “DLM-specific method가 AR baseline보다 항상 낫다”가 아니라 “masked-state signal을 exact-budget static allocator로 사용할 때, 기존 DLM-specific/static baselines 대비 held-out metric에서 재현 가능한 이득이 있는가”로 쓴다.
- 조사하지 못한 OpenReview `HD7tuVakmR`의 제목은 확인 전 인용하지 않는다.

