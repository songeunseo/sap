# LLaDA 채널 3848 보호 인과 실험 결과

## 발표용 핵심 결론

> **채널 3848은 Wanda pruning에서 무작위 채널보다 훨씬 많이 보존되는 특이 채널이지만, 75% sparsity에서 해당 경로를 완전히 보호해도 GSM8K 성능은 복구되지 않았다.**

- 256문항 GSM8K에서 Wanda 75% baseline은 **5/256 (1.95%)**, 채널 3848 보호 모델은 **1/256 (0.39%)**였다.
- 보호 개입은 계획대로 작동했다. 채널 경로에서 제거됐던 가중치 **637,166개를 복원**하고 같은 수를 다른 위치에서 제거해 전역 sparsity **75%를 유지**했다.
- 채널 3848 activation은 모든 측정 block에서 증가했지만 정확도는 **-1.56%p** 하락했다. 따라서 채널 3848의 보존은 GSM8K 성능 회복에 **충분조건이 아니다**.

## 1. 실험 질문과 설정

| 항목 | 설정 |
|---|---|
| 질문 | 75% Wanda pruning의 GSM8K 붕괴가 residual channel 3848 경로 제거 때문에 발생하는가? |
| 모델 | `GSAI-ML/LLaDA-8B-Base` |
| Baseline | Wanda unstructured pruning, 75% sparsity |
| Intervention | 32개 block, 224개 linear module에서 채널 3848을 읽거나 쓰는 가중치 보호 |
| 공정성 제약 | 복원한 수만큼 동일 module의 비보호 가중치를 추가 제거하여 module별 prune count 유지 |
| GSM8K | 동일한 앞쪽 64/256문항, task version 3, 기본 5-shot, temperature 0 |
| 생성 설정 | `gen_length=1024`, `steps=1024`, `block_length=1024`, `remasking=low_confidence` |
| Activation 진단 | 고정된 calibration sample 8개, block 0/15/31의 `mean(abs(channel 3848))` |

## 2. 핵심 GSM8K 결과

`flexible-extract`는 생성문에서 최종 숫자를 유연하게 추출한 정확도이고, `strict-match`는 요구된 답안 형식을 엄격히 만족한 정확도다.

| 평가 범위 | 모델 | Flexible EM | 정답 수 | 표준오차 | Strict EM | Baseline 대비 변화 |
|---:|---|---:|---:|---:|---:|---:|
| 64 | Wanda 75% | 1.56% | 1/64 | 1.56%p | 0.00% | — |
| 64 | Wanda 75% + protect-3848 | 0.00% | 0/64 | 0.00%p | 0.00% | **-1.56%p** |
| 256 | Wanda 75% | **1.95%** | **5/256** | 0.87%p | 0.00% | — |
| 256 | Wanda 75% + protect-3848 | **0.39%** | **1/256** | 0.39%p | 0.00% | **-1.56%p** |

### 발표 해석

- 작은 64문항 평가와 256문항 평가 모두에서 보호 모델이 baseline을 넘지 못했다.
- 256문항에서는 정답 수가 5개에서 1개로 줄어 상대적으로 80% 감소했다.
- 다만 정답 수가 매우 적고 confidence interval이 겹칠 수 있으므로, 이 결과만으로 “보호가 통계적으로 유의하게 해롭다”고 단정하지 않는다.
- 반면 예상했던 **회복 방향의 신호가 전혀 없기 때문에**, 채널 3848 단독 보호가 성능을 복구한다는 가설은 지지되지 않는다.

## 3. Repository reference와 현재 실험의 위치

아래 Dense와 Wanda 50% 값은 repository README의 전체 benchmark 참고값이다. 현재 75% 결과는 고정된 256문항 subset이므로 절대값을 직접 통계 비교하지 않고 성능 붕괴의 규모를 설명하는 용도로만 사용한다.

| 모델 | Sparsity | 평가 범위 | GSM8K Flexible EM | 용도 |
|---|---:|---|---:|---|
| Dense LLaDA | 0% | 전체 benchmark, repository reference | 69.29% | 정상 성능 참고 |
| Wanda | 50% | 전체 benchmark, repository reference | 57.01% | 중간 sparsity 참고 |
| Wanda | 75% | 현재 고정 subset 256개 | 1.95% | 인과 실험 baseline |
| Wanda + protect-3848 | 75% | 현재 고정 subset 256개 | 0.39% | 인과 intervention |

Dense 모델의 1문항 smoke test는 1/1 정답이었으며 evaluation pipeline 정상 동작 확인에만 사용했다.

## 4. 채널 보호 개입 검증

| 지표 | 결과 | 의미 |
|---|---:|---|
| 전체 pruning 대상 가중치 | 6,979,321,856 | 32 blocks × 7 linear modules |
| Baseline에서 제거된 가중치 | 5,234,491,392 | 정확히 75% |
| 채널 3848 보호 대상 | 1,703,936 | 전체 대상의 0.02441% |
| Baseline에서 이미 생존 | 1,066,770 (62.61%) | 개입 전에도 보존된 부분 |
| 개입으로 복원 | 637,166 (37.39%) | 실제 intervention 크기 |
| 보상 제거 | 637,166 | 복원 수와 동일 |
| 달라진 mask entry | 1,274,332 | 복원 + 보상 제거 |
| 전체 mask 변화율 | 0.01826% | 매우 국소적인 개입 |
| 최종 global sparsity | 75.00% | Baseline과 동일 |

모든 224개 대상 module에서 `복원 수 = 보상 제거 수`, `mask 차이 = 2 × 복원 수`, `baseline prune count = intervention prune count` 조건을 만족했다.

## 5. 채널 후보별 mask 진단

세 무작위 채널은 seed 0으로 선택됐으며 모두 동일한 Wanda baseline에서 독립적으로 계산됐다. 이 단계에서는 delta만 생성했고 GSM8K 평가는 실행하지 않았다.

| 채널 | 역할 | 이미 생존 | 새로 복원 | Mask 변화 수 | 전체 mask 변화율 |
|---:|---|---:|---:|---:|---:|
| **3848** | 실험 대상 | **1,066,770 (62.61%)** | **637,166 (37.39%)** | 1,274,332 | 0.01826% |
| 295 | Random control | 462,686 (27.15%) | 1,241,250 (72.85%) | 2,482,500 | 0.03557% |
| 809 | Random control | 483,705 (28.39%) | 1,220,231 (71.61%) | 2,440,462 | 0.03497% |
| 1316 | Random control | 443,298 (26.02%) | 1,260,638 (73.98%) | 2,521,276 | 0.03612% |

채널 3848은 random channel보다 baseline 생존율이 약 2.2–2.4배 높다. Wanda가 이 경로를 이미 상대적으로 중요하게 취급했다는 증거지만, 남은 경로까지 복원해도 downstream 정확도 회복으로 이어지지는 않았다.

## 6. 채널 3848 activation 결과

| Block | Dense | Wanda 75% | Baseline vs Dense | Protect-3848 | Protect vs Dense | Protect vs Baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28.512 | 24.280 | -14.84% | 24.700 | -13.37% | **+1.73%** |
| 15 | 94.187 | 49.936 | -46.98% | 51.946 | -44.85% | **+4.02%** |
| 31 | 25.474 | 22.516 | -11.61% | 26.582 | +4.35% | **+18.06%** |

보호 개입 후 세 block 모두에서 channel 3848 activation이 증가했다. 특히 마지막 block에서는 Dense보다 4.35% 높아졌지만 GSM8K 정확도는 회복되지 않았다. 즉 **activation 크기 회복과 task 정확도 회복은 동일하지 않다.**

## 7. 최종 판정

| 판정 항목 | 결과 |
|---|---|
| 개입 구현이 유효했는가? | **예.** 보호, 보상 pruning, sparsity 불변 조건을 모두 만족 |
| 채널 activation이 보존됐는가? | **예.** 측정한 모든 block에서 baseline 대비 증가 |
| GSM8K가 회복됐는가? | **아니오.** 256문항에서 -1.56%p, 5개 정답 → 1개 정답 |
| 인과 가설 | **지지되지 않음.** 채널 3848 보존은 GSM8K 회복의 충분조건이 아님 |
| 실험 분류 | **Case D — degradation**. 단, 악화의 통계적 유의성까지 주장하지 않음 |

가장 보수적인 해석은 다음과 같다.

1. 채널 3848은 Wanda가 무작위 채널보다 강하게 보존하는 특이 경로다.
2. 하지만 GSM8K 능력은 단일 residual channel이 아니라 여러 채널·레이어에 분산됐을 가능성이 높다.
3. 고정 sparsity를 위한 보상 pruning이 보호 효과를 상쇄했을 가능성도 배제할 수 없다.
4. 따라서 “채널 3848이 중요하다”와 “채널 3848만 살리면 능력이 돌아온다”는 서로 다른 주장이다. 본 실험은 후자를 지지하지 않는다.

## 8. 실행 범위와 한계

| 계획된 단계 | 상태 | 이유/용도 |
|---|---|---|
| Wanda 75% pruning 및 channel delta 생성 | 완료 | 3848 + random 3개 |
| Dense GSM8K smoke test | 완료 | pipeline sanity check만 수행 |
| Wanda / protect-3848, 동일 64문항 | 완료 | gross-regression check |
| Wanda / protect-3848, 동일 256문항 | 완료 | primary go/no-go 비교 |
| 전체 1,319문항 GSM8K | 미실행 | 256문항에서 회복 신호가 없고 모델당 약 14시간 필요 |
| Random channel GSM8K control | 미실행 | primary comparison에 회복 신호가 없어 execution gate에서 중단 |
| Sink-Aware 75% 비교 | 미실행 | primary comparison에 회복 신호가 없어 확장하지 않음 |

따라서 결론의 범위는 **Wanda 75%, 고정된 256문항, 채널 3848 단독 보호**에 한정한다. 전체 benchmark나 다른 pruning method에 일반화하지 않는다.

## 9. 발표용 한 문장

> “Wanda는 채널 3848 경로를 random channel보다 이미 두 배 이상 많이 보존했지만, 제거된 나머지 37.4%를 복원해 activation을 높여도 GSM8K는 1.95%에서 0.39%로 회복되지 않아, 이 채널 하나가 75% pruning 붕괴의 단독 원인은 아니라는 결과를 얻었습니다.”

## 원본 산출물

- GSM8K: `gsm8k-wanda75-{64,256}/`, `gsm8k-wanda75-protect3848-{64,256}/`
- Activation: `activation-dense.json`, `activation-wanda75.json`, `activation-wanda75-protect3848.json`
- Mask diagnostics: `wanda75-deltas/channel-{3848,295,809,1316}.json`
- 실행 로그: `wanda75-pruning.log`, `gsm8k-*.log`
