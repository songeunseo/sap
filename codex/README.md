# Codex 실험 기록

이 디렉터리는 Codex가 LLaDA-8B pruning과 denoising dynamics를 조사하면서 수행한 실험을 한곳에 정리한 기록이다. 설정, 실행 스크립트, 설계 문서, 원본 결과와 실패 로그를 실험별로 묶었다. 표의 수치는 콘솔 요약이 아니라 보존된 JSON과 sample JSONL에서 다시 추출했다.

## 한눈에 보기

| 디렉터리 | 질문 | 상태 | 핵심 결론 |
|---|---|---|---|
| [`temporal_rho/`](temporal_rho/) | 한 토큰의 오차가 denoising step을 지나며 증폭되는가? | Phase 1·2 완료 | 반복 수를 늘리자 중간 peak 가설은 기각됐고 후기 구간 영향이 더 컸다. |
| [`channel_3848/`](channel_3848/) | outlier channel 3848을 보호하면 75% Wanda의 GSM8K가 회복되는가? | 핵심 비교 완료 | activation은 회복됐지만 정확도는 1.95%에서 0.39%로 낮아져 단독 원인 가설은 지지되지 않았다. |
| [`time_risk_sensitivity/`](time_risk_sensitivity/) | timestep별 gradient 변동을 이용한 Time-Risk mask가 안정적인가? | Stage 1 NO-GO | split Spearman은 높았지만 split-mask Jaccard가 0.95 기준에 크게 못 미쳤다. |
| [`mean_dlm_sweep/`](mean_dlm_sweep/) | 평균 DLM gradient sensitivity만 사용한 pruning 성능은 어떤가? | 25/50/75% 완료 | WinoGrande에서는 기존 방법과 비슷했고, GSM8K는 75%에서 0%로 붕괴했다. |
| [`controlled_winogrande/`](controlled_winogrande/) | 같은 조건에서 Wanda, SparseGPT, Sink-Aware, Mean-DLM 중 무엇이 나은가? | 전체 비교 완료 | sparsity별 1위가 달랐고 격차가 작아 한 방법의 확실한 우위는 확인되지 않았다. |
| [`calibration_16x512/`](calibration_16x512/) | calibration token 수를 4배 늘리면 결과가 달라지는가? | 5/6 완료 후 중지 | 모든 paired 95% CI가 0을 포함해 확실한 calibration 효과는 없었다. |

## 공통 환경과 해석 방법

- 기본 모델: `GSAI-ML/LLaDA-8B-Base`
- 고정 revision: `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- pruning 대상: 32개 transformer block의 224개 linear module
- calibration 데이터: WikiText-2 train, seed 0
- 주 평가 GPU: NVIDIA RTX 5090 32GB
- WinoGrande 직접 비교: 전체 1,267문항, 5-shot, CFG 0, Monte Carlo 128, batch 8
- 표의 `±`는 정확도의 marginal standard error다. 동일 문항에서 두 모델의 변화를 비교할 때는 별도로 계산한 paired 95% confidence interval을 우선한다.
- 논문 표와 직접 실험은 절대 점수가 최대 6.32%p 달랐다. 따라서 논문 값은 참고치이고, 방법 비교에는 동일 환경에서 얻은 직접 실험값을 사용한다.

`results/` 아래의 대용량 mask, state, checkpoint와 로그는 Git에서 무시될 수 있지만 로컬 디스크에는 보존한다. 기존에 Git으로 추적하던 compact JSON, sample JSONL과 보고서는 계속 추적한다. 원본 로그와 JSON에 기록된 실행 당시 경로는 provenance이므로 재작성하지 않았다.

## 1. Temporal perturbation propagation

Dense LLaDA의 128-step deterministic denoising 과정에서 이미 확정된 한 토큰을 두 번째 후보로 바꾼 뒤, 다음 step에서 아직 mask인 위치들의 normalized hidden error가 어떻게 변하는지 측정했다. 주 지표는 `rho_temporal = e_s / e_t`이며 1보다 크면 unresolved representation에 대한 source-token 영향이 커진 것이다. 입력과 출력에서 측정되는 mask 집합이 달라지므로 operator norm이나 동일 좌표 contraction ratio로 해석하면 안 된다.

| 실행 | 표본 구성 | 대표 결과 | 판정 |
|---|---|---|---|
| Phase 1 | 1 sample × 1 perturbation, 3개 시점 | early 1.008, middle 1.057, late 0.932 | 중간 peak 가설을 잠정 지지했지만 시점당 `n=1` |
| Phase 2 | 4 samples × 3 perturbations, 5개 시점 | early 1.337, middle 1.140, late 1.793; 최대는 75% progress의 2.133 | 중간 peak 가설 기각, late 영향이 더 큼 |

Phase 2에서 모든 측정 시점의 평균 rho가 1보다 컸지만 confidence interval이 넓다. 이 실험은 propagation 진단이지 pruning 성능 실험이 아니다.

- [Phase 1 상세 보고서](temporal_rho/results/phase1/rho_profile_report.md)
- [Phase 2 상세 보고서](temporal_rho/results/phase2/rho_profile_report.md)
- 원본 구현 이력: `main` 브랜치의 `00c2339`, 최종 수정 `7b4a0df`

## 2. Channel 3848 인과 실험

### 2.1 전 채널 activation 측정

WikiText-2의 고정 8개 sequence에서 32 block × 4,096 residual channel을 전수 측정했다. 채널 번호를 선택 기준으로 사용하지 않고 정렬해도 channel 3848이 전체 1위였다.

| 위치 | Channel 3848 | 2위 | 비율 |
|---|---:|---:|---:|
| `q_proj` 입력 평균 절대 activation | 16.3971 | channel 753, 6.3279 | 2.59× |
| Post-block residual | 91.4623 | channel 753, 73.7790 | 1.24× |

이 결과는 dominant activation channel의 재현이지 downstream 과제에 대한 인과성을 뜻하지 않는다. [전 채널 측정 보고서](channel_3848/results/all_channel_profile_report.md)

### 2.2 Channel 보호 개입

75% Wanda baseline에서 channel 3848을 읽거나 쓰는 가중치를 복원하고 같은 module의 비보호 가중치를 동일 개수만큼 추가 제거했다. 따라서 global sparsity와 module별 prune count는 그대로다.

| 모델 | GSM8K 범위 | Flexible EM | 정답 수 | SE | Baseline 대비 |
|---|---:|---:|---:|---:|---:|
| Wanda 75% | 256 | 1.95% | 5/256 | 0.87%p | — |
| Wanda 75% + protect-3848 | 256 | 0.39% | 1/256 | 0.39%p | -1.56%p |

개입으로 637,166개 weight를 복원하고 같은 수를 보상 제거했으며, 224개 module 모두에서 불변식을 만족했다. 측정한 block 0/15/31에서 channel 3848 activation은 baseline 대비 각각 1.73%, 4.02%, 18.06% 증가했지만 GSM8K는 회복되지 않았다. 악화의 통계적 유의성까지 주장하지 않으며, 결론은 “channel 3848 단독 보호가 회복의 충분조건이 아니다”로 제한한다.

- [인과 실험 상세 결과](channel_3848/results/report.md)
- [설계](channel_3848/design.md) / [실행 계획](channel_3848/plan.md)
- `pruned_weights/`: Wanda 75% baseline과 두 보호 variant의 로컬 checkpoint

## 3. Time-Risk DLM gradient sensitivity

평균 DLM sensitivity `mu`에 timestep별 population standard deviation `sigma`를 더한 `mu + lambda·sigma` score가 calibration split 사이에서 안정적인 mask를 만드는지 검증했다. 기본 calibration은 8×256이며 timestep 10개, `lambda ∈ {0.25, 0.5, 1.0}`, decision sparsity `{0.50, 0.60, 0.70}`을 사용했다.

### Stage 결과

| 단계 | 결과 | 근거 |
|---|---|---|
| Stage 0 feasibility | GO | 예상 0.508 GPU h, peak allocated 16.16GiB, swap 증가 0, gradient 전부 finite/nonzero |
| Stage 1 첫 시도 | 운영 실패 | `torch.quantile()` 입력 tensor가 너무 커 artifact 발행 전 중단 |
| Stage 1 수정 후 | NO-GO | 32 block, 3,584 mask checksum audit는 통과했으나 reliability gate 실패 |

| Sparsity | Mean lambda-1 split Jaccard | 사전 기준 |
|---:|---:|---:|
| 50% | 0.792833 | ≥ 0.95 |
| 60% | 0.812856 | ≥ 0.95 |
| 70% | 0.838615 | ≥ 0.95 |

split `sigma` Spearman median은 0.884158로 0.5 기준을 통과했다. 그러나 실제 pruning decision mask는 split에 따라 너무 많이 달라졌으므로 Time-Risk 확장 실험은 중단했다. 이는 `lambda=0`인 Mean-DLM 자체의 무효 판정이 아니라, timestep risk 항을 안정적으로 더할 수 없다는 판정이다.

- [설정](time_risk_sensitivity/config.json)
- [상세 결과](time_risk_sensitivity/results/report.md)
- [설계](time_risk_sensitivity/design.md) / [실행 계획](time_risk_sensitivity/plan.md)

## 4. Mean-DLM sparsity sweep

Mean-DLM은 timestep 및 calibration state에 걸친 `weight² × gradient²` 평균인 `mu`만 사용한다. 각 output row에서 score가 가장 작은 weight를 지정 sparsity만큼 제거했다.

### GSM8K

모든 primary run은 동일한 앞쪽 256문항을 사용했다.

| Sparsity | 표본 | Flexible EM ± SE | Strict EM ± SE | 상태 |
|---:|---:|---:|---:|---|
| 25% | 256 | 65.63 ± 2.97 | 67.97 ± 2.92 | 완료 |
| 50% | 256 | 54.30 ± 3.12 | 54.69 ± 3.12 | 완료 |
| 60% | 1 | 0.00, SE 계산 불가 | 0.00, SE 계산 불가 | pipeline sanity only |
| 75% | 256 | 0.00 ± 0.00 | 0.00 ± 0.00 | 완료, 성능 붕괴 |

### WinoGrande

| Sparsity | Calibration | 표본 | Accuracy ± SE |
|---:|---|---:|---:|
| 25% | 8×256 | 1,267 | 74.43 ± 1.23 |
| 50% | 8×256 | 1,267 | 70.48 ± 1.28 |
| 75% | 8×256 | 1,267 | 50.91 ± 1.41 |

16문항 50% pilot은 87.5 ± 8.54였지만 표본이 너무 작아 primary 표에는 사용하지 않는다.

- [GSM8K runner](mean_dlm_sweep/run_gsm8k.sh)
- [WinoGrande runner](mean_dlm_sweep/run_winogrande.sh)
- [원본 결과](mean_dlm_sweep/results/)

## 5. 동일 조건 WinoGrande 비교

Dense와 네 pruning family를 WikiText-2 8×256, seed 0으로 직접 실행했다. Mean-DLM은 동일한 고수준 calibration 설정을 사용하지만 loader 경로가 달라 raw token span의 byte-identical 여부는 입증하지 못했다.

| Sparsity | Method | 직접 Accuracy ± SE | 논문 WinoG | 직접 − 논문 |
|---:|---|---:|---:|---:|
| Dense | Base | 74.59 ± 1.22 | 69.30 | +5.29 |
| 25% | **Wanda** | **74.51 ± 1.22** | 68.59 | +5.92 |
| 25% | Sink-Aware + Wanda | 74.35 ± 1.23 | 68.59 | +5.76 |
| 25% | SparseGPT | 73.88 ± 1.23 | 67.56 | +6.32 |
| 25% | Sink-Aware + SparseGPT | 73.95 ± 1.23 | 69.53 | +4.42 |
| 25% | Mean-DLM | 74.43 ± 1.23 | — | — |
| 50% | Wanda | 69.85 ± 1.29 | 64.56 | +5.29 |
| 50% | Sink-Aware + Wanda | 70.17 ± 1.29 | 65.27 | +4.90 |
| 50% | SparseGPT | 68.90 ± 1.30 | 64.64 | +4.26 |
| 50% | Sink-Aware + SparseGPT | 67.72 ± 1.31 | 65.82 | +1.90 |
| 50% | **Mean-DLM** | **70.48 ± 1.28** | — | — |
| 75% | Wanda | 50.59 ± 1.41 | 47.43 | +3.16 |
| 75% | Sink-Aware + Wanda | 50.67 ± 1.41 | 49.17 | +1.50 |
| 75% | **SparseGPT** | **51.46 ± 1.40** | 50.04 | +1.42 |
| 75% | Sink-Aware + SparseGPT | 49.96 ± 1.41 | 51.07 | -1.11 |
| 75% | Mean-DLM | 50.91 ± 1.41 | — | — |

sparsity별 점추정 1위는 25% Wanda, 50% Mean-DLM, 75% SparseGPT로 달랐다. 대부분의 차이는 marginal SE보다 작으며, 이 한 번의 실험으로 확실한 방법 우위를 주장하지 않는다. [상세 비교 보고서](controlled_winogrande/results/report.md)

## 6. Calibration 16×512 비교

Calibration을 8×256(2,048 tokens)에서 16×512(8,192 tokens)로 4배 늘리고 같은 WinoGrande 문항을 평가했다.

| Sparsity | Method | 8×256 | 16×512 | 변화 | Paired 95% CI |
|---:|---|---:|---:|---:|---:|
| 25% | Wanda | 74.51 ± 1.22 | 74.43 ± 1.23 | -0.08%p | [-0.94, +0.78] |
| 25% | SparseGPT | 73.88 ± 1.23 | 74.74 ± 1.22 | +0.87%p | [-0.34, +2.08] |
| 25% | Mean-DLM | 74.43 ± 1.23 | 74.27 ± 1.23 | -0.16%p | [-0.88, +0.57] |
| 50% | Wanda | 69.85 ± 1.29 | 70.17 ± 1.29 | +0.32%p | [-0.92, +1.55] |
| 50% | SparseGPT | 68.90 ± 1.30 | 69.06 ± 1.30 | +0.16%p | [-1.93, +2.25] |
| 50% | Mean-DLM | 70.48 ± 1.28 | 미실행 | — | — |

모든 완료 항목의 paired 95% CI가 0을 포함한다. 가장 큰 점추정 변화는 SparseGPT 25%의 +0.87%p지만 통계적으로 확실한 개선은 아니다. Mean-DLM 50%는 25% 평가 완료 직후 중지해 달라는 요청에 따라 시작하지 않았다.

- [16×512 설정](calibration_16x512/config.json)
- [재개 가능한 runner](calibration_16x512/run.sh)
- [원본 결과](calibration_16x512/results/)

## 종합 결론

1. **복잡한 pruning score가 WinoGrande 정확도에서 항상 우월하지 않았다.** 25%에서는 합리적인 방법들이 dense와 거의 같은 판단을 유지했고, 50/75%에서도 sparsity별 1위가 달랐다.
2. **Calibration 확대 효과는 현재 범위에서 작았다.** 4배 token 확대 후에도 확실한 변화가 없으므로 기존 성능 차이를 작은 calibration 하나로 설명하기 어렵다.
3. **Time-Risk 항은 mask 안정성 gate를 통과하지 못했다.** Gradient variance 순위 상관은 높았지만 실제 split mask가 충분히 일치하지 않았다.
4. **Channel 3848은 강한 activation outlier지만 단독 인과 원인은 아니었다.** 경로 보호와 activation 증가가 downstream 회복으로 이어지지 않았다.
5. **고 sparsity에서는 과제별 붕괴 양상이 다르다.** Mean-DLM 75%는 WinoGrande 약 50.9%를 유지했지만 GSM8K 256문항에서는 0%였다.

현재 결과는 한 모델, 주로 seed 0, 제한된 benchmark에 대한 것이다. 확실한 방법 우위를 주장하려면 calibration span과 seed를 완전히 통일하고 여러 benchmark에 대해 paired 반복 실험을 해야 한다.

## 재현 방법

저장소 루트 `/home/tmluser1/sap`에서 실행한다. 각 runner는 스스로 루트로 이동한다.

```bash
# 동일 조건 WinoGrande 전체 비교
bash codex/controlled_winogrande/run.sh

# Mean-DLM WinoGrande sweep
bash codex/mean_dlm_sweep/run_winogrande.sh full

# 16×512 calibration sweep 재개
# 완료된 5개 결과는 건너뛰고 미완료 Mean-DLM 50%부터 이어진다.
bash codex/calibration_16x512/run.sh
```

Time-Risk preflight는 다음처럼 재실행한다.

```bash
python dlm_gradient_sensitivity.py feasibility \
  --config codex/time_risk_sensitivity/config.json \
  --output codex/time_risk_sensitivity/results/stage0.json
```

Channel 3848 실험은 여러 장시간 gate와 checkpoint 수명주기를 포함하므로 단일 runner 대신 [실행 계획](channel_3848/plan.md)의 명령을 따른다. 대용량 결과나 checkpoint를 다시 생성하기 전에 필요한 디스크와 GPU 시간을 확인해야 한다.
