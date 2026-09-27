# DLM pruning 아이디어 후보 목록 — 사용자 제공 문헌 통합
Date: 2026-09-17
Status: candidate backlog; not an experiment plan or launch
Source: 사용자 첨부 pasted-text.txt
Original attachment: /home/tmluser1/.codex/attachments/d23f115a-f796-4490-a070-b5898c3169e4/pasted-text.txt

## 기록 원칙
사용자 요청은 여섯 아이디어를 후보에 추가하는 것. 논문 관측 요약은 사용자 제공 자료를 바탕으로 기록했으며, 이번 등록에서는 원문/venue/모델 revision을 독립적으로 재검증하지 않았다. 실제 실험 설계 전 해당 원문과 코드 확인. 아래 pruning 연결은 가설이며 신규 결과나 novelty 확정이 아니다.
기존 context-response, progress-subspace, decoder-decision-aware 후보를 대체하지 않는다. 첨부의 1/2번 우선 검토 제안은 보존하되 실행 순서는 확정하지 않는다.

## C1. 상태별 active-set 교체와 고정 support 용량
Source: Universal Properties of Activation Sparsity in Modern Large Language Models
https://arxiv.org/html/2509.00454
- 제공 자료의 관측: LLaDA FFN에서 단계별 activation sparsity가 유사해도 활성 neuron 집합은 denoising 과정에서 달라짐.
- Hypothesis: 평균 활성량이 비슷해도 여러 상태에 걸쳐 서로 다른 성분을 쓰는 모듈은 고정 weight support에 더 많은 용량이 필요할 수 있음.
- Candidate measurement: 상태별 활성 에너지의 일정 비율을 보존하는 하나의 고정 channel support의 최소 크기/coverage curve. Active-set turnover는 진단, raw union count는 최종 점수로 고정하지 않음.
- Allocation link: 같은 평균 active-set 크기/energy/depth 조건에서 coverage 요구량이 큰 모듈의 pruning 취약성 확인 후 보호 기준 후보.
- Unresolved: dynamic activation sparsity→static weight sparsity 연결, FFN neuron→projection별 unstructured weight budget 연결, sample 수/threshold 영향. 기존 clean-corrupt aggregate 순위 유사성과 논리적으로 다른 측정.
- Algorithm extension: 상태별 coverage 제약 아래 단일 support/예산을 공동 선택하는 최적화 후보. 평균 중요도 하나로만 환원할 필요 없음.
- Status: 새 독립 후보, 미실행.

## C2. SAE feature 공간의 손상 보존
Source: DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders
https://arxiv.org/html/2602.05859
https://github.com/Xu0615/SAE4DLM
- 제공 자료의 관측: Dream/LLaDA 일부 초반층에서 SAE reconstruction으로 activation을 교체할 때 masked-token CE가 감소하는 구간.
- Hypothesis: 좌표 MSE보다 pruning으로 소실/오활성화되는 feature 표현이 배분에 추가 정보를 줄 수 있음.
- Candidate measurement: dense calibration에 맞춰 고정한 encoder z의 z(H_dense)와 z(H_pruned) 차이. Feature 소실과 새 오활성화를 별도 관찰.
- Allocation link: 실제 module pruning 후 공통 residual 측정 지점의 feature 손상 또는 marginal 손상 비용으로 배분.
- Controls / unknowns: 일반 PCA, normalized/raw reconstruction, feature scale/decoder norm, Top-K 경계, 우리 checkpoint/layer SAE 호환성. SAE feature가 자동으로 인과적 의미/중요도를 갖는다고 가정하지 않음.
- Algorithm extension: 상태별 feature coverage와 sparsity를 함께 최적화할 여지. 초기에는 C1과 결합하지 않고 독립 기여를 비교.
- Status: 새 독립 후보, 미실행. SAE는 calibration 측정기로만 사용 가능.

## C3. 추가 mask에 대한 문맥 이용 강건성
Source: Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models
https://arxiv.org/html/2511.21338
- 제공 자료의 관측: 추가 generation mask가 문맥 이용을 방해할 수 있고 mask-agnostic fine-tuning으로 완화.
- Hypothesis: pruning이 불필요한 추가 mask에 대한 취약성을 더 키우는 projection이 존재.
- Candidate measurement: 내용/근거/평가 query를 유지하고 추가 mask를 바꾼 paired input에서 pruning excess damage:
  [L_sparse(extra)−L_dense(extra)]−[L_sparse(base)−L_dense(base)].
- Allocation link: 원래 긴 입력 난이도와 pruning의 추가 손상을 구분하여 비용 후보로 사용.
- Controls / unknowns: context positions/attention/출력공간, 원래 정답성, length 영향. 모든 mask-count 신호를 제거하는 목표가 아님.
- Relationship: 현재 context-response는 실제 gold context 추가와 mask 감소를 함께 바꿈. C3는 불필요한 출력 mask를 늘리는 개입이므로 별도 후보. Clock 진행도 보존과 구분.
- Status: 독립 진단/방법 후보, 미실행.

## C4. 양방향 문맥 이용 기능 보존
Source: Induction in Both Directions: A Mechanistic Analysis of In-Context Learning in Masked Diffusion Language Models
https://arxiv.org/html/2607.15893
- 제공 자료의 관측: 작은 1–3층 attention-only masked DLM의 양방향 induction 회로. LLaDA/Dream 대규모 회로 검증으로 확대 해석하지 않음.
- Hypothesis: aggregate 손상이 비슷해도 pruning이 좌/우 문맥 이용을 다르게 손상시킬 수 있음.
- Candidate measurement: 같은 target/단서를 좌측 또는 우측에 둔 paired retrieval/복원 문제에서 actual projection pruning의 방향별 손상.
- Allocation link: 양방향 기능을 보존하는 비용/제약 후보. max/mean aggregation을 미리 정당화하지 않음.
- Controls / unknowns: 거리/위치/문제난이도 대조, 8B 모델 및 일반 downstream 전이. Attention 값만으로 중요도 판정하지 않음.
- Relationship: context-response의 방향별 확장으로 연결하되 별도 기능 가설로 보존.
- Status: 대규모 모델 적용성 확인이 필요한 후보, 미실행.

## C5. 보상 관계를 고려한 공동 배분
Source: The Hydra Effect: Emergent Self-repair in Language Model Computations
https://arxiv.org/html/2307.15771
- 제공 자료의 관측: AR Chinchilla 7B에서 attention 교란 후 후속 계산의 부분 보상.
- Hypothesis: projection A의 pruning 안전성이 보상하는 B의 생존에 의존할 수 있음.
- Candidate measurement: 동일 배경/입력에서 interaction I_AB=L(A,B)−L(A)−L(B)+L(empty).
- Allocation link: 강한 유해 상호작용 쌍을 함께 과도하게 자르지 않는 graph/제약 또는 후보 묶음의 실제 검증.
- Limits: positive interaction만으로 self-repair 메커니즘이 증명되지는 않음. O(G²) 비용은 후보 제한 필요. AR 근거이므로 DLM 고유성 자동 확보 아님.
- Relationship: 기존 Role decision audit/V2 sparse interaction 및 decoder-decision-aware 반복 배분과 연결. 이전 인접 projection gain 검사에서 보편 보상 근거는 없었고, context refresh mini는 실패. 이를 완전히 새 아이디어/검증된 방향으로 재포장하지 않음.
- Status: 기존 상호작용 후보의 선행 근거 및 구조적 확장, 미실행.

## C6. 토큰 의존성/정보 공개에 대한 응답 보존
Source: Breaking AR’s Sampling Bottleneck: Provable Acceleration via Diffusion Language Models
https://arxiv.org/abs/2505.21400
https://arxiv.org/html/2505.21400
- 제공 자료의 관측: 특정 가정/sampling schedule에서 sampling error를 token dependence, denoising iteration 수, predictor error와 연결. 정확한 theorem/상수/정의는 원문 확인 필요.
- Hypothesis: pruning이 다른 token 공개 후 조건부 예측을 갱신하는 기능을 손상.
- Candidate measurement: 동일 masked query의 context 전후 dense/sparse response 차이.
- Relationship: 이미 설계/실행한 context-response A+C vs A-only의 직접 관련 이론 후보. 별도 새 실험으로 중복 등록하지 않음.
- Limits: 논문의 MI는 데이터 token 의존성이지 layer hidden MI가 아님. 우리 gold-vs-rest response error가 그 theorem의 predictor-error 항/상계와 같다고 주장하지 않음. 현재 실험의 context 내용과 mask count 혼합 유지.
- Status: 기존 후보의 이론적 근거 보강. 연결 유도 미완료.

## 기존 후보와 함께 보는 목록
| ID | 후보 | 역할/상태 |
|---|---|---|
| C1 | active-set 교체 / 고정 support coverage | 새 독립 후보 |
| C2 | SAE feature 보존 | 새 독립 후보 |
| C3 | 추가 mask 강건성 | 새 독립 후보 |
| C4 | 양방향 문맥 이용 | 새 독립 후보, 모델 규모 전이 미확인 |
| C5 | 보상 의존성 / 공동 allocation | 기존 sparse interaction 계열 확장 |
| C6 | 조건부 의존성 / context-response | 기존 A+C 실험과 통합, 이론 근거 후보 |
| P1 | progress-associated subspace 보존 | 이전 clock 가설, 미검증 |
| P2 | decoder decision-aware iterative allocation | 이전 알고리즘 초안, 미검증 |

## Decision
6개 모두 후보군에 편입. C1/C2 독립 비교 가능성을 보존하고 처음부터 모두 결합하지 않는다.
현재 frozen 실험/평가 protocol은 변경하지 않으며 신규 GPU/SAE 학습/추가 평가를 시작하지 않았다.
기본 평가 지표는 기존 합의 WikiText NELBO/PPL bound, GSM8K는 추가 task 검증. 이번 후보 등록으로 평가 우선순위를 변경하지 않음.

## Related Notes
- [[Research/DLM-Pruning/Research-State]]
- [[Research/DLM-Pruning/Hypotheses/2026-09-16-Context-Response-Allocation-Proposal]]
- [[Experiments/2026-09-16-Context-Response50-GSM8K-Mini100]]
- [[Research/DLM-Pruning/Hypotheses/2026-09-17-Subliminal-Clocks-Pruning-Link]]
- [[Research/DLM-Pruning/Hypotheses/2026-09-17-Decoder-Decision-Aware-Iterative-Allocation]]
- [[Research/DLM-Pruning/Experiments/Role Exchange Prediction V2 Corrective Rerun]]
- [[Research/DLM-Pruning/Experiments/2026-09-12 Role Allocation Decision Audit]]

