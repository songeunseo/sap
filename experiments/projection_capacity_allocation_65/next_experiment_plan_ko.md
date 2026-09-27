# Oracle allocation 이후 실험 계획

작성일: 2026-09-09. 상태: 제안; 새 GPU 실험 실행 승인을 의미하지 않음.

## 확인한 출발점

- downstream.json: Uniform full GSM8K 139/1319, Capacity 250/1319.
- status.json: complete_and_audited. final_audit.json: verified.
- 실제 제거 개수: 두 모델 모두 4,536,008,704. nominal 65%, actual 64.9921123798%.
- curves/*.json에 sparsity별, state별 KL과 local_reconstruction_error가 이미 있음.
- local_reconstruction_error는 해당 Linear의 전체 token 출력 squared error / dense output squared energy이며 masked-only error가 아님.
- core.py allocator는 marginal damage/(0.05*N)으로 정렬하고 실제 row-floor count로 정확한 예산 도달 가능성을 검사함. 새 비교에서도 그대로 유지한다.

## 목적과 범위

먼저 oracle의 이득이 단순 depth/type 패턴 또는 local reconstruction으로 설명되는지 판별한다. 그 뒤에만 cheap functional-capacity estimator를 개발한다. Attention 전달 기반 V weight score는 별도 연구 후보이며 이번 allocation 실험에 결합하지 않는다.

## 공통 고정 조건

모델 revision, 224 targets와 순서, unweighted Standard Wanda ranking, 50/55/60/65/70/75% grid, 실제 제거 개수, historical generation/strict EM protocol을 유지한다. Historical artifact는 수정하지 않는다. 후속 산출물은 experiments/projection_capacity_followup_65/ 아래 별도 저장한다. Expensive jobs는 승인 후 tmux에서 실행한다.

## 1단계: GPU 없는 기존 데이터 진단

1. 224개 파일의 completeness, six-level grid, state IDs, finite values를 검사한다.
2. state-average D_g(r), E_g(r), 인접 increment, 비용/parameter를 계산한다. Raw negative increment는 보존한다.
3. E와 D의 상관을 level, marginal cost별로 보고한다. Projection type별 및 layer/type 효과를 조정한 관계도 함께 보고한다. 상관만으로 성공 판정을 내리지 않는다.
4. Reconstruction-only allocator에 기존 E curve를 입력해 allocation을 만든다. 기존 LSA의 재현이라고 부르지 않는다.
5. Oracle와의 assignment 차이, 실제 mask XOR, parameter-weighted budget 이동량을 보고한다. Oracle가 피한 expensive increment를 reconstruction이 선택하는지 확인한다.
6. 단일 anchor 가설은 r0=65%로 사전 고정한다. alpha_g=D_g(65)/E_g(65), Dhat_g(r)=alpha_g E_g(r). 분모가 0이면 임의 epsilon 튜닝 대신 해당 projection과 처리 필요성을 보고한다.
7. 8 calibration sequences를 deterministic 4/4로 분리해 양방향 cross-fit한다. Alpha와 predictor allocation은 construction fold에서만 산출하고 다른 fold의 oracle damage로 평가한다. Anchor level 자체는 예측 성능 평가에서 제외한다. 관측된 full GSM8K로 anchor나 식을 선택하지 않는다.
8. D/E 비율의 sparsity 의존성과 predicted marginal ranking을 검사한다. Held-out fold에서 reconstruction-only보다 additive oracle damage를 줄이지 못하면 single-anchor 방법 후보는 여기서 중단한다. 이 additive 평가는 full sparse 성능의 대체가 아니다.

## 2단계: 가장 먼저 비교할 jointly sparse 모델

다음 네 모델을 동결해 비교한다.

- Uniform: 기존 mask 재사용.
- Capacity oracle: 기존 mask 재사용.
- Reconstruction allocation: 1단계 E curve와 기존 allocator.
- EIS+type diagnostic control: Oracle가 각 projection type에 배정한 sparsity의 multiset은 유지하되, 같은 type 안에서 큰 sparsity를 early layer부터 내림차순 배치한다.

EIS+type은 oracle로부터 type별 budget과 sparsity 분포를 받은 강한 설명용 대조군이다. Cheap 독립 baseline이나 Layer Collapse 원문 재현으로 부르지 않는다. 동일 type의 weight shape와 level별 실제 count가 동일한지 먼저 확인한다. 동일하지 않으면 단순 permutation의 exact-budget 주장이 성립하지 않으므로 construction을 중단해 정확한 제약을 정한다. 실제 생성 mask로 모든 budget을 재검증한다.

이 대조군의 질문: projection별 개별 대응 관계가 필요한가, 아니면 같은 type별 budget을 depth 순으로 정렬해도 충분한가?

## 3단계: 평가와 다음 분기

기존 40-state held-out와 GSM8K 결과는 이미 본 결과다. 새 predictor의 반복 선택용 test set으로 쓰지 않는다. 새로운 disjoint WikiText-2 8 spans x 5 timesteps의 40 states를 만들고, corpus intervals 및 masks/digest를 고정한 뒤 네 모델을 동일 경로로 평가한다. 같은-path dense/sham 검증을 수행한다.

Primary comparison은 Capacity - EIS+type으로 사전 고정한다. Mean KL difference, paired state bootstrap 20,000회(seed 0), 8 sequence/5 timestep 평균을 보고한다. 같은 sequence의 states 의존성을 고려한 sequence-cluster bootstrap도 함께 보고하며 원래 oracle gate를 소급 변경하지 않는다. Capacity - Reconstruction은 사전 지정 secondary comparison이다. 두 비교를 모두 우월성 주장에 사용할 경우 다중비교 보정 결과도 보고한다.

두 새 comparator는 mini100 및 full1319를 모두 평가해 비교를 완성한다. 이번 목적은 새로운 predictor 후보 선별이 아니라 기존 oracle gain의 귀속 검증이므로 mini 결과로 comparator를 제외하지 않는다. 기존 Uniform/Capacity GSM8K prediction은 protocol 및 example hashes가 일치하면 재사용한다. Paired strict-EM transition counts와 exact McNemar 검정을 보고한다. 같은 GSM8K를 이미 본 한계와 선택 편향을 명시한다.

분기:

- Capacity가 EIS+type을 DLM과 downstream에서 일관되게 능가: 세부 projection 대응 관계를 예측할 근거 강화. Cheap predictor 단계로 진행.
- EIS+type과 구분되지 않음: Oracle-vs-uniform 성공은 유지되지만 세밀한 capacity 측정의 추가 가치는 미입증. Cheap predictor를 바로 확장하지 않는다. 동등하다고 단정하지도 않는다.
- Reconstruction만으로 oracle와 유사한 성능: Functional predictor 필요성 약화. 기존 reconstruction allocation과의 비교를 우선한다.
- DLM과 GSM8K 방향 불일치: Fidelity를 성능의 대리 지표로 삼지 않고 mismatch를 보고한다.

## 4단계: 조건부 predictor 연구

1단계에서 단일-anchor 가설이 살아남으면 그 predictor를 우선 검토한다. 그렇지 않더라도 propagation 필요성이 남으면 짧은 suffix 관측을 소규모 pilot으로 측정한다. 모든 224 projections의 소수 calibration states에서 block-exit와 한 후속 block 지점을 관측하고, final KL/marginal cost와의 관계 및 reconstruction 대비 추가 설명력을 검사한다. 이 두 지점은 개발용 비교이며 최종 평가 전에 하나를 동결한다. 마지막 block boundary 처리와 same-path sham을 명시한다.

Full backward는 사용하지 않는다. 임의 Fisher/confidence/Reveal 혼합이나 수동 module 보호를 추가하지 않는다. Raw distributions, redundancy, score/assignment/mask 변화, 실제 실패 모드 확인 순서를 지킨다. 신규 predictor를 선택했다면 별도의 미사용 평가 states를 확보한다.

## 논문 단계에서 필요한 비교

최종 후보가 살아남은 뒤 OWL, LSA, 원문 EIS의 공식 규칙을 확인하여 같은 protocol로 비교한다. Oracle-derived EIS+type 대조군으로 이 비교를 대체하지 않는다. 65%는 현재 confirmatory target으로 유지한다. 다른 sparsity와 Dream/SparseGPT 확장은 최종 규칙 동결 후 일반화 평가로 시행하며 65% 결과를 구제하는 tuning에 사용하지 않는다.

## 자원 및 산출물

우선 실행 단위는 1단계 CPU 진단이다. 다음 GPU 단위는 두 새 comparator의 held-out DLM 평가다. 시간은 기존 실행 속도만으로 확정하지 않고, 승인 후 짧은 timing pilot으로 추정한다. 새 full GSM8K 두 모델은 각각 기존 측정 7.8~10.6시간 수준이 참고치이나 shared GPU/생성 길이에 따라 달라질 수 있다.

후속 디렉터리에 config, source hashes, diagnostic tables, frozen allocations, mask manifests, new state verification, held-out per-state results, downstream paired results, report, logs를 보관한다. 본 계획 작성만으로 GPU job이나 새로운 방법 구현을 시작하지 않는다.
