# Dual-Role Reconstruction Allocation Diagnostic — Design

## 목적

Masked diffusion language model에서 같은 projection은 두 역할을 동시에 수행한다.
Masked position은 현재 예측 대상이고, unmasked position은 다른 masked token을 예측하는
확정 문맥이다. 전체 token을 합친 local reconstruction error가 두 역할 중 한쪽의
projection별 pruning 병목을 평균으로 가리는지 검증한다.

이번 단계는 **방법 성공을 주장하거나 full sparse model을 만드는 단계가 아니다.**
Standard Wanda candidate mask의 local error를 masked/unmasked로 분해하고, 사전 고정한
role-bottleneck curve가 aggregate reconstruction보다 oracle functional allocation을 더 잘
근사하는지 확인하는 analysis-only kill gate다.

## 기존 증거와 경계

- Masked/unmasked activation distribution은 다르지만 CGQ-Wanda statistic은 기존 Wanda와
  평균 Spearman 0.99738로 거의 중복됐다.
- Reveal/Remain 또는 confidence weighting을 강하게 적용해 weight mask를 바꾸면 downstream이
  단조 개선되지 않았다.
- 50% single-projection perturbation의 masked enrichment와 final KL 상관은 0.080,
  projection type 보정 후 0.191로 약했다.
- Aggregate local-reconstruction allocation은 full GSM8K에서 Uniform 139/1319보다 높은
  255/1319를 기록했다. 따라서 local reconstruction 자체는 allocation 신호로 살아 있다.
- 현재 structural diagnostic은 timestep-specific allocation을 지지하지 않았다.

따라서 이번 가설은 “masked token이 더 중요하다”가 아니다. 두 역할 중 더 취약한 쪽을
보호하는 **parameter-free minimax allocation**이 aggregate reconstruction의 averaging
failure를 줄이는지가 질문이다. Weight ranking, confidence, Reveal/Remain, gradient,
Fisher, attention sink 및 timestep별 mask는 변경하지 않는다.

## 고정 입력

- Model: `GSAI-ML/LLaDA-8B-Base`
- Revision: `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- Projection targets/order: 기존 32 blocks × 7 projections = 224
- Candidate masks: `experiments/projection_capacity_allocation_65/candidate_mask_manifest.json`
- Sparsity grid: 50/55/60/65/70/75%
- Local selector: 기존 80-state Standard Wanda ranking
- States: 기존 allocation calibration 8 sequences × 10 timesteps = 80
- Oracle diagnostic labels: 기존 per-state single-projection Dense||Sparse KL curve
- Target budget: 실제 Uniform-65 row-floor budget 4,536,008,704 / 6,979,321,856 weights

모든 source hash, state identity, module order/shape, mask hash, row-floor count를 실행 전에
검증한다. 기존 artifacts는 수정하거나 덮어쓰지 않는다.

## Role-conditioned reconstruction 측정

Projection `g`, sparsity `r`, state `s`의 dense output과 해당 Wanda mask output 차이를
`delta Y = Y_sparse - Y_dense`로 둔다. Persisted corruption mask로 token position을 나눈다.

```text
num_M[g,r,s] = sum_{p in masked, o} deltaY[p,o]^2
den_M[g,s]   = sum_{p in masked, o} Y_dense[p,o]^2
num_U[g,r,s] = sum_{p in unmasked, o} deltaY[p,o]^2
den_U[g,s]   = sum_{p in unmasked, o} Y_dense[p,o]^2
```

FP32에서 제곱하고 FP64 CPU scalar로 누적한다. 어떤 state에서도 한 group이 비거나
denominator가 0이면 epsilon을 임의 도입하지 않고 중단한다.

Construction sequence 집합 `S`에서 token을 pooling한 curve는 다음과 같다.

```text
E_M[g,r;S] = sum_s num_M / sum_s den_M
E_U[g,r;S] = sum_s num_U / sum_s den_U
E_all[g,r;S] = (sum_s num_M + sum_s num_U) / (sum_s den_M + sum_s den_U)
E_role[g,r;S] = max(E_M, E_U)
contrast[g,r;S] = abs(E_M - E_U) / (E_M + E_U)
```

`E_M=E_U=0`이면 contrast를 0으로 정의한다. 따라서 contrast 계산에 epsilon이나 scale
hyperparameter가 없다. `E_role=max(E_M,E_U)`는 미리 고정하며 mean, weighted mean, learned coefficient 또는
projection type별 공식을 결과를 보고 선택하지 않는다. Raw non-monotonic curve와 음의
marginal cost를 그대로 저장하고 envelope를 사용하지 않는다.

## 실행 방식과 numerical sanity

기존 candidate mask 여섯 개와 dense block-prefix를 재사용한다. 각 module/state에서
historical variant batch shape와 동일한 dense+6 batch를 target module까지 전달한 뒤,
target Linear output에서 group statistics를 계산하고 suffix 계산을 즉시 중단한다.
Full suffix logits나 functional KL을 다시 계산하지 않는다.

이 방식이 historical local reconstruction과 같은 값을 내는지 다음 sanity를 수행한다.

```text
E_reconstructed[g,r,s]
  = (num_M + num_U) / (den_M + den_U)
```

이를 기존 `local_reconstruction_error[g,r,s]`와 224×6×80 전체에서 비교한다.
최대 절대 오차가 `1e-7`보다 크거나 최대 상대 오차가 `1e-5`보다 크면 결과를 사용하지
않고 STOP한다. Dense parameter hash는 실행 전후 동일해야 한다.

GPU 작업은 GPU 0이 비어 있을 때 tmux에서만 실행하며 GPU 1은 사용하지 않는다.
Projection 단위 checkpoint와 provenance 검증을 지원한다.

## 분석 순서

### 1. Raw distributions

- `E_M`, `E_U`, `E_all`, `E_role`, `contrast`의 sparsity별 분포
- `E_M/E_U`와 marginal cost의 quantile/CV
- negative marginal 및 curve monotonicity violation
- layer, projection type, layer quartile, sequence, timestep breakdown
- 가장 큰/작은 role contrast projection-increment 각 20개

### 2. Group comparison과 안정성

- `E_M>E_U`, `E_U>E_M`, exact tie로 정의한 masked-dominant, unmasked-dominant, balanced projection 수
- Sequence별 및 timestep별 role contrast rank 안정성
- 4/4 sequence 양방향 construction/validation에서 `E_M`, `E_U`, `contrast` 재현성
- `attn_out`, `v_proj`, `ff_out`에 효과가 국소화되는지 정량화

### 3. Redundancy와 failure-mode validation

- `E_role`/`contrast`와 `E_all`, historical reconstruction curve, depth/type,
  Wanda activation statistic의 correlation
- Oracle functional marginal KL cost를 대상으로 construction sequences에서만 두 nested OLS를 적합한다.
  Baseline은 intercept, 31개 depth dummy, 6개 projection-type dummy와 `E_all` marginal
  cost/parameter를 사용한다. Extended model은 여기에 `(E_role-E_all)` marginal
  cost/parameter 하나를 추가한다. Reference categories는 block 0과 `attn_out`으로 고정한다.
- 두 모델의 계수를 validation oracle cost에 그대로 적용하고 module×increment RMSE 및
  sequence별 RMSE를 비교한다. Standardization은 construction rows에서만 수행하며 zero-variance
  feature는 사전 검증 실패로 처리한다.
- Aggregate reconstruction과 role allocation의 assigned sparsity, projection change count,
  exact nested-mask XOR 비교

### 4. Conditional allocation diagnostic

두 sequence 방향 각각에서 construction sequences만 사용해 다음 두 allocation을 만든다.

1. Aggregate: `E_all` marginal increase / additional parameter
2. Role bottleneck: `E_role` marginal increase / additional parameter

둘 다 50%에서 시작해 기존 exact discrete water-filling과 repository-order tie break를 사용하고
Uniform-65 actual row-floor budget을 정확히 맞춘다. Validation sequences의 기존 oracle
functional KL curves로 selected additive damage를 평가한다. Wanda masks가 전체 80 states로
고정되었다는 점 때문에 이는 fixed-mask-family conditional cross-validation이며 완전히 독립된
selector validation으로 부르지 않는다.

## 사전등록 kill gate

Role-bottleneck 경로는 다음을 모두 만족할 때만 **DIAGNOSTIC SUPPORTED**다.

1. Numerical reconstruction sanity와 dense-weight hash가 통과한다.
2. Role allocation이 Aggregate allocation보다 양방향 sequence fold 모두 낮은 validation
   additive oracle damage를 낸다.
3. 8개 held-out sequence의 paired damage 차이 `Role - Aggregate` bootstrap 95% CI가 전부 0 아래다.
4. 두 allocation의 exact nested-mask XOR가 전체 prunable weights의 1% 이상이다.
5. Extended OLS가 Baseline OLS보다 양방향 fold 모두 validation RMSE가 낮고, 8개 held-out
   sequence의 paired RMSE 차이 `Extended - Baseline` bootstrap 95% CI가 전부 0 아래다.

하나라도 실패하면 **NOT SUPPORTED**로 종료한다. Max 대신 mean 사용, group coefficient 조정,
grid/target 변경, 특정 module 보호, confidence/Reveal 정보 추가로 구제하지 않는다.

통과하더라도 이번 단계에서 full sparse model, held-out KL 또는 GSM8K를 실행하지 않는다.
그 결과를 검토한 후 official EIS/OWL 및 Aggregate reconstruction을 포함한 별도 full-model
사전 계획을 동결한다.

## 산출물

새 디렉터리 `experiments/dlm_dual_role_allocation/`에 다음을 저장한다.

- `config.json`
- `state_verification.json`
- `role_reconstruction_raw.json`
- `role_distributions.json`
- `redundancy_analysis.json`
- `crossfit_allocations.json`
- `decision.json`
- `logs/`

사람용 최종 보고서는 Obsidian의
`Research/DLM-Pruning/Experiments/` 아래에 작성한다. Repository에는 재현성과 provenance를
위한 machine-readable artifacts만 필수로 두며, 기존 실험 보고서를 덮어쓰지 않는다.

## 결과 해석 범위

- 통과 결과는 token-role-conditioned local reconstruction이 oracle allocation을 근사할
  가능성을 보인다는 뜻이지 downstream 개선 증거가 아니다.
- Random-corruption states를 사용하므로 actual reverse-trajectory 효과라고 부르지 않는다.
- AR control이 없으므로 masked/unmasked 현상이 DLM에만 존재한다고 주장하지 않는다.
- DLLMQuant/STaR-Quant가 token-state disparity를 사용한 선행연구라는 점을 명시하고,
  novelty는 이를 projection-wise weight sparsity allocation의 role bottleneck으로 연결하는 데 둔다.
