# EXP-004 — 디코딩 인지 토큰 중요도 방향 × 그래디언트 집계

## 0. 요약

- **최고 성능:** REVEAL-ABS, **767 / 1319 = 58.1501%**.
- REVEAL-ABS는 동결 UNIFORM-ABS보다 **+22문제 / +1.6679 pp** 높았지만 exact McNemar `p=0.0776532`였고, exploratory Holm 보정값은 `0.388266`이었다.
- 사전 지정 primary인 REVEAL-ABS vs REMAIN-ABS는 **+20문제 / +1.5163 pp**, discordant `73:53`, exact `p=0.0901229`였다. 따라서 REVEAL 방향은 **유망한 신호이지 확립된 우위가 아니다**.
- ABS는 두 token objective 모두에서 SQUARE보다 높았다. Holm 보정 후 유의한 비교는 **REVEAL-ABS vs REVEAL-SQUARE** 하나였다 (`+3.7908 pp`, Holm `p=0.00604355`).
- REVEAL-ABS는 Wanda보다 **+90문제 / +6.8234 pp**, SparseGPT보다 **+183문제 / +13.8741 pp** 높았지만, Dense보다는 **−171문제 / −12.9644 pp** 낮았다.
- 결론: 관찰 순위는 preregistered Outcome A와 일치하지만 token-weighting 개선은 통계적으로 확정되지 않았다. **ABS는 여전히 기준 aggregation**이다.

## 1. 목적 (Objective)

동일한 DLM-loss gradient pruning 절차에서 다음 transition에 곧 reveal/commit될 token과 계속 masked/unresolved로 남을 token 중 어느 쪽을 더 중요하게 보아야 하는지 확인한다. 동시에 token-level 중요도 도입 후에도 cross-state aggregation의 선호가 ABS인지, SQUARE로 뒤집히는지 검증한다.

실험이 바꾸는 것은 오직 masked-token loss에 곱하는 `alpha_(s,j)`뿐이다. 모델, pruning universe, sparsity, calibration states, timestep, corruption, DLM loss의 나머지 normalization, GSM8K protocol은 고정했다.

## 2. 가설 (Hypotheses)

- **H1 — REVEAL/COMMIT:** 다음 transition에 reveal되는 token은 이후 denoising context가 되므로, 이 token에 중요한 parameter를 보존하면 오차 전파를 줄일 수 있다. 예측: `REVEAL > UNIFORM`.
- **H2 — REMAIN/UNRESOLVED:** 계속 masked로 남는 token은 더 어렵고 추가 refinement가 필요하므로, 이 token에 중요한 parameter를 보존하는 편이 유리하다. 예측: `REMAIN > UNIFORM`.
- 어느 방향도 사전에 정답으로 가정하지 않았다.
- **Primary comparison:** `REVEAL-ABS vs REMAIN-ABS`.

## 3. 고정 설정 (Frozen Setup)

| 항목 | 설정 |
|---|---|
| 모델 | `GSAI-ML/LLaDA-8B-Base` |
| Revision | `0f2787f2d87eac5eed8a087d5ecd24277e6255b2` |
| dtype | BF16 |
| EXP-001 원본 run | `20260828T175010-2484545` |
| Calibration dataset | WikiText-2 train, 8 samples |
| Timesteps | `0.05, 0.15, ..., 0.95`의 10개 지점 |
| Calibration states | 8 × 10 = **80 frozen corrupted states** |
| Sequence length | 256 |
| Calibration seed | 0 |
| State SHA-256 | `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df` |
| Pruning | 50% unstructured, exact row-wise |
| Prunable universe | 32 blocks × 7 Linear matrices = **224 matrices**, 총 6,979,321,856 weights |
| Matrix types | `q_proj`, `k_proj`, `v_proj`, `attn_out`, `ff_proj`, `up_proj`, `ff_out` |

기존 EXP-001/002의 layer set, sparsity, row-wise semantics, revision, samples, corruptions 및 timestep을 변경하지 않았다.

## 4. 다음 reveal partition 정의

각 frozen state `s`에 대해 dense unpruned model을 한 번 실행하고, 현재 masked 위치의 argmax token과 softmax confidence를 저장했다.

```text
M_s = 현재 masked token 위치
steps_remaining = max(1, ceil(p_mask(t) * 256))
next_reveal_count = get_num_transfer_tokens(current_mask, steps_remaining)[0, 0]
R_s = 실제 low-confidence-remasking 규칙에서 다음에 transfer되는 top-confidence 위치
U_s = M_s \ R_s
```

이 partition은 dense model에서 한 번 계산해 동결했으며 ABS/SQUARE나 pruned model별로 다시 계산하지 않았다. 모든 state에서 `R_s ∩ U_s = ∅`, `R_s ∪ U_s = M_s`를 검증했다.

### 관측된 partition 통계

| 통계 | Masked | Reveal | Remain |
|---|---:|---:|---:|
| 전체 token 수 | 10,267 | 116 | 10,151 |
| 전체 masked 중 비율 | 100.0000% | 1.1298% | 98.8702% |
| state당 평균 | 128.3375 | 1.4500 | 126.8875 |
| state당 중앙값 | 133.0 | 1.0 | 131.5 |
| state당 최소–최대 | 7–245 | 1–2 | 6–243 |

- 80 states 중 reveal count 1인 state는 **44개**, reveal count 2인 state는 **36개**였다. top-1로 고정하지 않았으며 실제 schedule 결과를 사용했다.
- 전체 confidence: mean `0.374454`, min `0.011719`, p50 `0.209961`, p90 `0.984375`, p99 `1.000000`, max `1.000000`.
- Reveal confidence: mean `0.966477`, median `1.000000`, min `0.255859`, max `1.000000`.
- Remain confidence: mean `0.367689`, median `0.205078`, min `0.011719`, max `1.000000`.
- Reveal token이 전체 masked token의 약 1.13%뿐이라는 심한 불균형은 결과 해석에서 중요하다.

## 5. Token weighting과 DLM loss

```text
UNIFORM raw alpha:    reveal=1, remain=1
REVEAL-UP raw alpha:  reveal=2, remain=1
REMAIN-UP raw alpha:  reveal=1, remain=2

alpha_(s,j) = raw_alpha_(s,j) / mean_(k in M_s)(raw_alpha_(s,k))
mean_(j in M_s)(alpha_(s,j)) = 1
```

| 조건 | 정규화 전 비율 (R:U) | 정규화 후 reveal α 범위 | 정규화 후 remain α 범위 | state별 평균 α 범위 |
|---|---:|---:|---:|---:|
| UNIFORM | 1:1 | 1.000000–1.000000 | 1.000000–1.000000 | 1.0000000000000000–1.0000000000000000 |
| REVEAL | 2:1 | 1.750000–1.991803 | 0.875000–0.995902 | 0.9999999999999998–1.0000000000000002 |
| REMAIN | 1:2 | 0.501031–0.538462 | 1.002062–1.076923 | 0.9999999999999998–1.0000000000000002 |

공식 EXP-001 loss semantics를 유지한 실제 구현식은 다음과 같다.

```text
L_s(alpha) = sum_(j in M_s)[alpha_(s,j) * CE_(s,j)] / p_mask(t) / 256
d_(i,s) = -w_i * dL_s(alpha)/dw_i
```

동일 logits와 per-token CE graph에서 세 loss를 만들되, `torch.autograd.grad`로 condition별 gradient를 독립 계산했다. Parameter `.grad`는 항상 `None`으로 유지했고 state 및 condition 사이 gradient accumulation은 허용하지 않았다. Activation magnitude, Wanda score, confidence 연속 가중, timestep 추가 가중, sample/layer/parameter normalization은 넣지 않았다.

관측된 80-state 평균 loss는 UNIFORM `2.942043`, REVEAL `2.910066`, REMAIN `2.958977`였다. Alpha 평균은 동일하지만 token loss와 alpha의 상관 때문에 최종 loss 값까지 동일할 필요는 없다.

## 6. Cross-state aggregation과 mask 생성

```text
S_i^ABS    = mean_s |d_(i,s)|
S_i^SQUARE = mean_s d_(i,s)^2
```

Signed SUM은 EXP-002에서 `0 / 1319`였으므로 다시 평가하지 않았다. 각 matrix의 각 행에서 score 하위 50%를 정확히 prune했다. 네 신규 mask만 생성했고 UNIFORM-ABS/SQUARE는 동결 EXP-001 mask를 재사용했다.

| 방법 | Overall mask SHA-256 |
|---|---|
| REMAIN-ABS | `9f8260cc196a7cd93e340f216113c0419f8741dc693b4f2efcb3d3c76d924430` |
| REMAIN-SQUARE | `d189caaa05cd85a950da1b686a235294c9441aa8277d07de791b2200776fe286` |
| REVEAL-ABS | `7dc0b8e18cb60828505b0fceb2009745ea5751ba555141d36ee7d462d22f4071` |
| REVEAL-SQUARE | `f08891717b7c19e27f6946acb7c0d19ff02a9f9ea0073e183169a31fb9ee6225` |

Full score tensor는 영구 저장하지 않고 module-wise로 처리했다. 각 `*_scores.json`에는 module별 shape, score SHA-256, mean/std/min/max, row prune count, sparsity 및 연결된 mask hash가 남아 있다.

## 7. Sanity checks와 재현성 gate

| 검사 | 결과 |
|---|---|
| 동일 80 states | PASS |
| Dense partition이 ABS/SQUARE에 공통 | PASS |
| 모든 state/condition에서 mean(alpha)≈1 | PASS; 관측 범위 0.9999999999999998–1.0000000000000002 |
| REVEAL raw ratio 2:1 | PASS |
| REMAIN raw ratio 1:2 | PASS |
| Condition/state gradient 격리 | PASS; `torch.autograd.grad with parameter .grad always None` |
| Calibration 중 parameter update 없음 | PASS |
| 신규 mask exact row-wise 50% | PASS |
| Evaluation config hash 일치 | PASS; `1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add` |
| Packed payload 감사 | PASS; 4 × 224 = 896 files, 3,489,660,928 bytes |
| Per-example 감사 | PASS; 6 × 1319 = 7,914 aligned rows |
| 관련 회귀 테스트 | PASS; 170 tests |

### UNIFORM 재현성 deviation과 gate

초기 요구는 frozen EXP-001 mask와 bit-exact 일치였다. 그러나 변경하지 않은 EXP-001 공식 loss/backward/hook 경로의 block-0 control도 CUDA/compiled-backward 비결정성으로 bit-exact하지 않았다. GSM8K 결과를 보기 전에 control 최대치 × 1.25로 threshold를 고정했다.

| 항목 | ABS | SQUARE |
|---|---:|---:|
| 실제 global XOR | 0.053317% (3,721,168 / 6,979,321,856) | 0.064455% (4,498,508 / 6,979,321,856) |
| 고정 global threshold | 0.152123% | 0.152123% |

- Bit-exact: **아님**.
- 최대 실제 module XOR: `0.190413%`; 고정 module threshold: `0.249699%`.
- 두 global 값과 모든 module 값이 threshold 아래여서 calibrated gate를 통과했다. Gate 통과 전에는 GSM8K를 시작하지 않았다.

## 8. Mask diagnostics

### 8.1 전체 mask

| Mask 비교 | 다른 weight 수 | Global XOR | Spearman | Mask IoU | 보존 top-50% overlap |
|---|---:|---:|---:|---:|---:|
| REMAIN-ABS/REMAIN-SQUARE | 199,897,752 | 2.8641% | 0.992612 | 94.4365% | 97.1359% |
| REMAIN-ABS/UNIFORM-ABS | 6,626,172 | 0.0949% | 0.999994 | 99.8103% | 99.9051% |
| REMAIN-SQUARE/UNIFORM-SQUARE | 11,540,894 | 0.1654% | 0.999977 | 99.6699% | 99.8346% |
| REVEAL-ABS/REMAIN-ABS | 13,661,302 | 0.1957% | 0.999971 | 99.6093% | 99.8043% |
| REVEAL-ABS/REVEAL-SQUARE | 196,867,222 | 2.8207% | 0.992851 | 94.5180% | 97.1793% |
| REVEAL-ABS/UNIFORM-ABS | 9,671,968 | 0.1386% | 0.999987 | 99.7232% | 99.8614% |
| REVEAL-SQUARE/REMAIN-SQUARE | 27,361,746 | 0.3920% | 0.999847 | 99.2194% | 99.6080% |
| REVEAL-SQUARE/UNIFORM-SQUARE | 18,351,668 | 0.2629% | 0.999935 | 99.4757% | 99.7371% |

### 8.2 32개 layer별 분포 요약

아래 표는 각 비교의 32개 layer 값을 최소·중앙값·최대로 요약한다. 최대 XOR layer도 함께 표시했다.

| Mask 비교 | Layer XOR 최소 | 중앙값 | 최대 (layer) | Layer Spearman 최소 | 중앙값 | 최대 |
|---|---:|---:|---:|---:|---:|---:|
| REMAIN-ABS/REMAIN-SQUARE | 2.3328% | 2.7956% | 3.6525% (L28) | 0.987617 | 0.993182 | 0.995677 |
| REMAIN-ABS/UNIFORM-ABS | 0.0827% | 0.0917% | 0.1133% (L28) | 0.999990 | 0.999995 | 0.999996 |
| REMAIN-SQUARE/UNIFORM-SQUARE | 0.1140% | 0.1471% | 0.2420% (L28) | 0.999948 | 0.999984 | 0.999991 |
| REVEAL-ABS/REMAIN-ABS | 0.1570% | 0.1829% | 0.2595% (L28) | 0.999940 | 0.999977 | 0.999984 |
| REVEAL-ABS/REVEAL-SQUARE | 2.3134% | 2.7911% | 3.5287% (L28) | 0.988432 | 0.993260 | 0.995882 |
| REVEAL-ABS/UNIFORM-ABS | 0.1169% | 0.1299% | 0.1739% (L28) | 0.999974 | 0.999989 | 0.999992 |
| REVEAL-SQUARE/REMAIN-SQUARE | 0.2259% | 0.3498% | 0.6086% (L28) | 0.999639 | 0.999884 | 0.999955 |
| REVEAL-SQUARE/UNIFORM-SQUARE | 0.1642% | 0.2348% | 0.4084% (L1) | 0.999851 | 0.999948 | 0.999979 |

### 8.3 224개 matrix별 분포 요약

각 비교에는 32 blocks × 7 matrices = 224개의 matrix row가 있다. 전체 1,792개 상세 row는 JSON에 보존하고, 본문에는 비교별 분포와 최대 XOR matrix를 싣는다.

| Mask 비교 | Matrix XOR 최소 | 중앙값 | 최대 (layer/module) | Matrix Spearman 최소 | 중앙값 | 최대 |
|---|---:|---:|---:|---:|---:|---:|
| REMAIN-ABS/REMAIN-SQUARE | 1.7134% | 2.6911% | 4.7102% (L28/ff_out) | 0.983485 | 0.994220 | 0.998095 |
| REMAIN-ABS/UNIFORM-ABS | 0.0750% | 0.0899% | 0.1580% (L0/k_proj) | 0.999988 | 0.999995 | 0.999997 |
| REMAIN-SQUARE/UNIFORM-SQUARE | 0.1023% | 0.1413% | 0.2774% (L4/q_proj) | 0.999933 | 0.999984 | 0.999994 |
| REVEAL-ABS/REMAIN-ABS | 0.1428% | 0.1810% | 0.3094% (L0/k_proj) | 0.999918 | 0.999979 | 0.999988 |
| REVEAL-ABS/REVEAL-SQUARE | 1.6993% | 2.6729% | 4.5823% (L28/ff_out) | 0.984067 | 0.994479 | 0.998133 |
| REVEAL-ABS/UNIFORM-ABS | 0.1077% | 0.1291% | 0.2444% (L0/k_proj) | 0.999964 | 0.999989 | 0.999994 |
| REVEAL-SQUARE/REMAIN-SQUARE | 0.2021% | 0.3437% | 0.7485% (L1/attn_out) | 0.999478 | 0.999896 | 0.999973 |
| REVEAL-SQUARE/UNIFORM-SQUARE | 0.1457% | 0.2314% | 0.5180% (L1/attn_out) | 0.999775 | 0.999955 | 0.999988 |

### 8.4 7개 module type별 상세 진단

| Mask 비교 | Module type | 다른 weight 수 | XOR | Spearman | Mask IoU | 보존 overlap |
|---|---|---:|---:|---:|---:|---:|
| REMAIN-ABS/REMAIN-SQUARE | attn_out | 13,182,016 | 2.4553% | 0.995783 | 95.2107% | 97.5447% |
| REMAIN-ABS/REMAIN-SQUARE | ff_out | 50,868,488 | 3.1583% | 0.991798 | 93.8842% | 96.8417% |
| REMAIN-ABS/REMAIN-SQUARE | ff_proj | 49,664,970 | 3.0836% | 0.990476 | 94.0186% | 96.9164% |
| REMAIN-ABS/REMAIN-SQUARE | k_proj | 14,761,592 | 2.7496% | 0.994448 | 94.6515% | 97.2504% |
| REMAIN-ABS/REMAIN-SQUARE | q_proj | 14,381,080 | 2.6787% | 0.994732 | 94.7854% | 97.3213% |
| REMAIN-ABS/REMAIN-SQUARE | up_proj | 44,068,118 | 2.7361% | 0.992202 | 94.6777% | 97.2639% |
| REMAIN-ABS/REMAIN-SQUARE | v_proj | 12,971,488 | 2.4161% | 0.995564 | 95.2849% | 97.5839% |
| REMAIN-ABS/UNIFORM-ABS | attn_out | 498,762 | 0.0929% | 0.999995 | 99.8144% | 99.9071% |
| REMAIN-ABS/UNIFORM-ABS | ff_out | 1,563,376 | 0.0971% | 0.999994 | 99.8061% | 99.9029% |
| REMAIN-ABS/UNIFORM-ABS | ff_proj | 1,578,046 | 0.0980% | 0.999994 | 99.8042% | 99.9020% |
| REMAIN-ABS/UNIFORM-ABS | k_proj | 488,752 | 0.0910% | 0.999995 | 99.8181% | 99.9090% |
| REMAIN-ABS/UNIFORM-ABS | q_proj | 492,968 | 0.0918% | 0.999996 | 99.8165% | 99.9082% |
| REMAIN-ABS/UNIFORM-ABS | up_proj | 1,521,882 | 0.0945% | 0.999994 | 99.8112% | 99.9055% |
| REMAIN-ABS/UNIFORM-ABS | v_proj | 482,386 | 0.0899% | 0.999995 | 99.8205% | 99.9101% |
| REMAIN-SQUARE/UNIFORM-SQUARE | attn_out | 872,916 | 0.1626% | 0.999980 | 99.6754% | 99.8374% |
| REMAIN-SQUARE/UNIFORM-SQUARE | ff_out | 2,745,536 | 0.1705% | 0.999976 | 99.6597% | 99.8295% |
| REMAIN-SQUARE/UNIFORM-SQUARE | ff_proj | 2,754,466 | 0.1710% | 0.999973 | 99.6586% | 99.8290% |
| REMAIN-SQUARE/UNIFORM-SQUARE | k_proj | 867,200 | 0.1615% | 0.999980 | 99.6775% | 99.8385% |
| REMAIN-SQUARE/UNIFORM-SQUARE | q_proj | 871,564 | 0.1623% | 0.999980 | 99.6759% | 99.8377% |
| REMAIN-SQUARE/UNIFORM-SQUARE | up_proj | 2,592,348 | 0.1610% | 0.999977 | 99.6787% | 99.8390% |
| REMAIN-SQUARE/UNIFORM-SQUARE | v_proj | 836,864 | 0.1559% | 0.999980 | 99.6888% | 99.8441% |
| REVEAL-ABS/REMAIN-ABS | attn_out | 1,038,942 | 0.1935% | 0.999976 | 99.6137% | 99.8065% |
| REVEAL-ABS/REMAIN-ABS | ff_out | 3,313,056 | 0.2057% | 0.999970 | 99.5895% | 99.7943% |
| REVEAL-ABS/REMAIN-ABS | ff_proj | 3,201,822 | 0.1988% | 0.999966 | 99.6032% | 99.8012% |
| REVEAL-ABS/REMAIN-ABS | k_proj | 1,013,676 | 0.1888% | 0.999974 | 99.6231% | 99.8112% |
| REVEAL-ABS/REMAIN-ABS | q_proj | 1,013,464 | 0.1888% | 0.999975 | 99.6232% | 99.8112% |
| REVEAL-ABS/REMAIN-ABS | up_proj | 3,097,004 | 0.1923% | 0.999971 | 99.6162% | 99.8077% |
| REVEAL-ABS/REMAIN-ABS | v_proj | 983,338 | 0.1832% | 0.999977 | 99.6344% | 99.8168% |
| REVEAL-ABS/REVEAL-SQUARE | attn_out | 12,992,702 | 2.4201% | 0.995957 | 95.2777% | 97.5799% |
| REVEAL-ABS/REVEAL-SQUARE | ff_out | 50,230,406 | 3.1187% | 0.992036 | 93.9579% | 96.8813% |
| REVEAL-ABS/REVEAL-SQUARE | ff_proj | 48,999,786 | 3.0423% | 0.990754 | 94.0960% | 96.9577% |
| REVEAL-ABS/REVEAL-SQUARE | k_proj | 14,467,470 | 2.6948% | 0.994690 | 94.7548% | 97.3052% |
| REVEAL-ABS/REVEAL-SQUARE | q_proj | 14,065,686 | 2.6199% | 0.994995 | 94.8963% | 97.3801% |
| REVEAL-ABS/REVEAL-SQUARE | up_proj | 43,360,540 | 2.6922% | 0.992435 | 94.7603% | 97.3078% |
| REVEAL-ABS/REVEAL-SQUARE | v_proj | 12,750,632 | 2.3750% | 0.995738 | 95.3633% | 97.6250% |
| REVEAL-ABS/UNIFORM-ABS | attn_out | 732,436 | 0.1364% | 0.999989 | 99.7275% | 99.8636% |
| REVEAL-ABS/UNIFORM-ABS | ff_out | 2,313,566 | 0.1436% | 0.999986 | 99.7131% | 99.8564% |
| REVEAL-ABS/UNIFORM-ABS | ff_proj | 2,280,214 | 0.1416% | 0.999985 | 99.7173% | 99.8584% |
| REVEAL-ABS/UNIFORM-ABS | k_proj | 720,610 | 0.1342% | 0.999988 | 99.7319% | 99.8658% |
| REVEAL-ABS/UNIFORM-ABS | q_proj | 720,314 | 0.1342% | 0.999988 | 99.7320% | 99.8658% |
| REVEAL-ABS/UNIFORM-ABS | up_proj | 2,203,902 | 0.1368% | 0.999987 | 99.7267% | 99.8632% |
| REVEAL-ABS/UNIFORM-ABS | v_proj | 700,926 | 0.1306% | 0.999989 | 99.7392% | 99.8694% |
| REVEAL-SQUARE/REMAIN-SQUARE | attn_out | 2,081,862 | 0.3878% | 0.999872 | 99.2279% | 99.6122% |
| REVEAL-SQUARE/REMAIN-SQUARE | ff_out | 6,562,854 | 0.4075% | 0.999846 | 99.1888% | 99.5925% |
| REVEAL-SQUARE/REMAIN-SQUARE | ff_proj | 6,453,808 | 0.4007% | 0.999821 | 99.2021% | 99.5993% |
| REVEAL-SQUARE/REMAIN-SQUARE | k_proj | 2,077,958 | 0.3870% | 0.999861 | 99.2293% | 99.6130% |
| REVEAL-SQUARE/REMAIN-SQUARE | q_proj | 2,088,764 | 0.3891% | 0.999862 | 99.2253% | 99.6109% |
| REVEAL-SQUARE/REMAIN-SQUARE | up_proj | 6,130,456 | 0.3806% | 0.999848 | 99.2420% | 99.6194% |
| REVEAL-SQUARE/REMAIN-SQUARE | v_proj | 1,966,044 | 0.3662% | 0.999875 | 99.2706% | 99.6338% |
| REVEAL-SQUARE/UNIFORM-SQUARE | attn_out | 1,385,754 | 0.2581% | 0.999946 | 99.4853% | 99.7419% |
| REVEAL-SQUARE/UNIFORM-SQUARE | ff_out | 4,363,640 | 0.2709% | 0.999935 | 99.4598% | 99.7291% |
| REVEAL-SQUARE/UNIFORM-SQUARE | ff_proj | 4,356,086 | 0.2705% | 0.999924 | 99.4607% | 99.7295% |
| REVEAL-SQUARE/UNIFORM-SQUARE | k_proj | 1,402,686 | 0.2613% | 0.999939 | 99.4790% | 99.7387% |
| REVEAL-SQUARE/UNIFORM-SQUARE | q_proj | 1,405,366 | 0.2618% | 0.999939 | 99.4780% | 99.7382% |
| REVEAL-SQUARE/UNIFORM-SQUARE | up_proj | 4,117,036 | 0.2556% | 0.999936 | 99.4902% | 99.7444% |
| REVEAL-SQUARE/UNIFORM-SQUARE | v_proj | 1,321,100 | 0.2461% | 0.999946 | 99.5092% | 99.7539% |

핵심 관찰:

- Aggregation 변경은 REVEAL에서 `2.8207%`, REMAIN에서 `2.8641%`의 mask를 바꿨다.
- Token weighting만 바꾸면 UNIFORM 대비 XOR는 `0.0949%–0.2629%`에 그쳤다.
- REVEAL vs REMAIN XOR는 ABS `0.1957%`, SQUARE `0.3920%`였고 Spearman은 각각 `0.999971`, `0.999847`이었다.
- 즉 이 실험에서 aggregation 선택은 token 방향보다 pruning decision을 훨씬 더 크게 바꿨다. 이것은 관측된 연관이며 downstream 차이의 인과적 설명으로 단정하지 않는다.
- 전체 224 matrix, 32 layer, 7 module-type별 상세치는 `mask_diagnostics.json`의 2,112 rows에 저장돼 있다.

## 9. GSM8K 평가 protocol

| 항목 | 값 |
|---|---|
| Dataset | GSM8K full test, N=1319 |
| Harness | repository `LLaDAEvalHarness` |
| Prompt | 5-shot |
| Metric | lm-eval `exact_match,strict-match` |
| Temperature | 0 |
| Generation length | 256 |
| Block length | 256 |
| Denoising steps | 256 |
| Seeds | random 0; numpy/torch/few-shot 1234 |
| Evaluation config SHA-256 | `1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add` |

UNIFORM 결과와 모든 신규 방법은 동일한 example ID, document/prompt/target hash를 사용했다. Dense/Wanda/SparseGPT/DLM-SUM은 동일 EXP-002 evaluation protocol의 동결 결과다.

## 10. 전체 성능 비교: Dense, Wanda, SparseGPT 포함

표의 세 `대비` 열은 **정답 수 차이 / 정확도 차이(pp)** 형식이다.

| 구분 | 방법 | 희소도 | Calibration | 정답 / 1319 | 정확도 | Dense 대비 | Wanda 대비 | SparseGPT 대비 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 비희소 기준 | Dense | 0.0000% | 없음 | 938 / 1319 | 71.1145% | +0 / +0.0000 pp | +261 / +19.7877 pp | +354 / +26.8385 pp |
| EXP-004 신규 | REVEAL-ABS | 50.0000% | 동일 80 states | 767 / 1319 | 58.1501% | -171 / -12.9644 pp | +90 / +6.8234 pp | +183 / +13.8741 pp |
| EXP-004 신규 | REMAIN-ABS | 50.0000% | 동일 80 states | 747 / 1319 | 56.6338% | -191 / -14.4807 pp | +70 / +5.3071 pp | +163 / +12.3578 pp |
| 동결 대조군 | UNIFORM-ABS | 50.0000% | 동일 80 states | 745 / 1319 | 56.4822% | -193 / -14.6323 pp | +68 / +5.1554 pp | +161 / +12.2062 pp |
| EXP-004 신규 | REVEAL-SQUARE | 50.0000% | 동일 80 states | 717 / 1319 | 54.3594% | -221 / -16.7551 pp | +40 / +3.0326 pp | +133 / +10.0834 pp |
| EXP-004 신규 | REMAIN-SQUARE | 50.0000% | 동일 80 states | 709 / 1319 | 53.7528% | -229 / -17.3616 pp | +32 / +2.4261 pp | +125 / +9.4769 pp |
| 동결 대조군 | UNIFORM-SQUARE | 50.0000% | 동일 80 states | 701 / 1319 | 53.1463% | -237 / -17.9682 pp | +24 / +1.8196 pp | +117 / +8.8704 pp |
| 표준 참조 | Wanda | 50.0000% | clean 8×256 | 677 / 1319 | 51.3268% | -261 / -19.7877 pp | +0 / +0.0000 pp | +93 / +7.0508 pp |
| 표준 참조 | SparseGPT | 50.0002% | clean 8×256 | 584 / 1319 | 44.2760% | -354 / -26.8385 pp | -93 / -7.0508 pp | +0 / +0.0000 pp |
| EXP-002 기각 | DLM-SUM | 50.0000% | 동일 80 states | 0 / 1319 | 0.0000% | -938 / -71.1145 pp | -677 / -51.3268 pp | -584 / -44.2760 pp |

### 참조 baseline 해석 주의사항

- Dense는 pruning하지 않은 upper reference다.
- Wanda와 SparseGPT는 8개 clean WikiText-2 256-token span을 사용했다. DLM 계열의 80 corrupted states와 **calibration compute가 matched된 대조군이 아니다**.
- DLM/Wanda는 exact row-wise 50%지만, SparseGPT는 repository의 128-column block global threshold를 사용해 전체 희소도만 약 50.0002%이며 각 행이 정확히 50%일 필요는 없다.
- 신규 방법과 Wanda/SparseGPT의 표 차이는 descriptive reference다. 이 비교를 위한 추가 post-hoc McNemar test는 preregistered primary matrix에 넣지 않았다.
- EXP-002에서 DLM-SUM은 0%였고 이미 기각됐기 때문에 EXP-004에서 재실행하지 않았다.

## 11. EXP-004 factorial 결과와 질문별 답

### Q1 — Token importance가 UNIFORM보다 도움이 되었는가?

- ABS: REVEAL-ABS는 UNIFORM-ABS보다 `+22 / +1.6679 pp`였으나 exact `p=0.0776532`, Holm `p=0.388266`; REMAIN-ABS는 `+2 / +0.1516 pp`, exact `p=0.928492`, Holm `p=1`.
- SQUARE: REVEAL-SQUARE는 UNIFORM-SQUARE보다 `+16 / +1.2130 pp`, exact `p=0.181224`, Holm `p=0.724896`; REMAIN-SQUARE는 `+8 / +0.6065 pp`, exact `p=0.545534`, Holm `p=1`.
- 네 weighted variant 모두 관찰 정확도는 matching UNIFORM 이상이었지만, 어느 weighted-vs-UNIFORM 비교도 Holm 보정 후 유의하지 않았다. 따라서 simple binary weighting의 일반적 개선을 확립하지 못했다.

### Q2 — REVEAL과 REMAIN 중 어느 방향이 유용한가?

- ABS primary: REVEAL-ABS가 REMAIN-ABS보다 `+20 / +1.5163 pp`; discordant `73:53`; exact `p=0.0901229`.
- SQUARE exploratory: REVEAL-SQUARE가 REMAIN-SQUARE보다 `+8 / +0.6065 pp`; discordant `76:68`; exact `p=0.559821`, Holm `p=1`.
- 두 aggregation 모두 REVEAL > REMAIN 순위였지만 통계적으로 결정적이지 않다. H1은 **suggestive**, H2는 지지되지 않았다.

### Q3 — Token weighting이 aggregation 선호를 바꾸었는가?

- REVEAL: ABS가 SQUARE보다 `+50 / +3.7908 pp`; exact `p=0.000863365`, Holm `p=0.00604355`.
- REMAIN: ABS가 SQUARE보다 `+38 / +2.8810 pp`; exact `p=0.0136694`, Holm `p=0.0820165`.
- 두 token objective 모두 ABS > SQUARE였고 reversal은 없었다. Holm 보정 후에는 REVEAL 조건의 ABS 우위만 유의했다.

## 12. Paired 통계 전체

| 비교 (A vs B) | 둘 다 정답 | A만 정답 | B만 정답 | 둘 다 오답 | Discordant | A−B | Exact p | Holm p | 역할 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| REVEAL-ABS vs UNIFORM-ABS | 685 | 82 | 60 | 492 | 142 | +1.6679 pp | 0.0776532 | 0.388266 | Exploratory |
| REMAIN-ABS vs UNIFORM-ABS | 684 | 63 | 61 | 511 | 124 | +0.1516 pp | 0.928492 | 1 | Exploratory |
| REVEAL-ABS vs REMAIN-ABS | 694 | 73 | 53 | 499 | 126 | +1.5163 pp | 0.0901229 | — | Primary |
| REVEAL-SQUARE vs UNIFORM-SQUARE | 646 | 71 | 55 | 547 | 126 | +1.2130 pp | 0.181224 | 0.724896 | Exploratory |
| REMAIN-SQUARE vs UNIFORM-SQUARE | 638 | 71 | 63 | 547 | 134 | +0.6065 pp | 0.545534 | 1 | Exploratory |
| REVEAL-SQUARE vs REMAIN-SQUARE | 641 | 76 | 68 | 534 | 144 | +0.6065 pp | 0.559821 | 1 | Exploratory |
| REVEAL-ABS vs REVEAL-SQUARE | 633 | 134 | 84 | 468 | 218 | +3.7908 pp | 0.000863365 | 0.00604355 | Exploratory |
| REMAIN-ABS vs REMAIN-SQUARE | 615 | 132 | 94 | 478 | 226 | +2.8810 pp | 0.0136694 | 0.0820165 | Exploratory |

모든 비교는 동일한 1,319 examples에 대한 exact two-sided binomial McNemar test다. Primary 한 개는 사전 지정되어 보정하지 않았고, 나머지 7개 exploratory p-value에는 Holm 보정을 적용했다. Discordant pair가 실제 검정 정보를 제공하며 p-value만으로 인과를 주장하지 않는다.

Primary exact p=`0.0901229`로 alpha=0.05에 도달하지 않았다. Holm 보정 후 alpha=0.05를 통과한 exploratory 비교는 **REVEAL-ABS vs REVEAL-SQUARE**이다.

## 13. Preregistered outcome rules에 따른 해석

- ABS와 SQUARE 모두 관찰 순위는 `REVEAL > UNIFORM` 및 `REVEAL > REMAIN`이므로 형식상 **Outcome A ordering**이다.
- 그러나 primary가 유의하지 않고 weighted-vs-UNIFORM도 Holm 보정 후 유의하지 않으므로 “commitment/transition preservation이 입증됐다”고 결론 내릴 수 없다.
- 정확한 결론은 **REVEAL 방향의 약한·유망한 신호가 있었으나 simple 2:1 binary importance의 이득은 확정되지 않았다**이다.
- REVEAL과 REMAIN이 작은 차이만 낸 것은 reveal token이 1.13%뿐이고 score/mask가 거의 동일했다는 관찰과 양립한다. 다만 이것이 원인임을 이번 실험만으로 식별하지는 못한다.
- Aggregation에 대해서는 UNIFORM, REVEAL, REMAIN 모두 ABS > SQUARE였다. EXP-002의 ABS 선호는 유지됐고 token weighting interaction에 의한 reversal은 없었다.
- Confidence 자체가 보편적으로 중요하다거나 다른 ratio/benchmark에서도 REVEAL이 우월하다는 주장은 하지 않는다.

## 14. 실행 시간과 자원

- Scoring rerun: `3시간 7분 19초` (`11239.423`초).
- 네 신규 GSM8K 평가의 측정 시간 합: `44시간 27분 18초` (`160037.972`초).

| 방법 | 평가 시간 | 초 |
|---|---:|---:|
| REVEAL-ABS | 11시간 7분 15초 | 40034.725 |
| REVEAL-SQUARE | 11시간 6분 41초 | 40000.973 |
| REMAIN-ABS | 11시간 6분 45초 | 40004.902 |
| REMAIN-SQUARE | 11시간 6분 37초 | 39997.372 |

- Scoring peak CUDA allocated: `18,278,837,760` bytes; reserved: `20,126,367,744` bytes.
- Mask payload: 방법당 `872,415,232` bytes, 네 방법 총 3,489,660,928 bytes.
- Preflight 당시 가용 disk `4,880,248,832` bytes, 요구량(안전 여유 포함) `4,026,531,840` bytes로 통과했다.
- 장시간 scoring/evaluation은 전용 tmux에서 실행했다.

## 15. 구현 편차 및 실패 이력

1. 첫 scoring 시 PyTorch AOTAutograd가 compiled graph 반복 backward의 donated buffer를 거부했다. `torch._functorch.config.donated_buffer`만 비활성화해 세 독립 `autograd.grad`가 같은 forward graph를 재사용하게 했으며 loss/partition/pruning 의미는 바꾸지 않았다 (`31dca02`).
2. 첫 full scoring은 32/32 blocks를 끝냈지만 초기 bit-exact gate를 통과하지 못해 GSM8K 전에 중단됐다. 변경 없는 legacy block-0 control도 비결정성을 보였으므로 결과 관찰 전에 control 기반 threshold를 동결했다 (`22fa6af`).
3. Evaluation consumer에 남은 stale `exact=True` 조건 때문에 첫 evaluation launch가 model load 전에 종료됐다. Threshold-compliant `status=passed`를 따르도록 consumer만 수정하고 regression test 후 재실행했다 (`470f0e5`). 이 실패들에서는 GSM8K 결과가 관측되지 않았다.
4. Disk 제약 때문에 full score tensor는 module-wise 처리 후 폐기하고 hash/statistics/shape만 저장했다. Packed mask는 하나씩 write/checksum/readback했다.
5. UNIFORM mask는 bit-exact하지 않았으며 calibrated nondeterminism gate를 통과했다는 편차를 숨기지 않고 위에 정량 보고했다.
6. Permutation/random-token control, confidence-continuous weighting, ratio sweep, timestep/trajectory weighting, 추가 benchmark, 다른 sparsity는 넣지 않았다.

## 16. 산출물

- `config.json`: 동결 설정과 EXP-001/002 provenance
- `calibration_state_manifest.json`: 80 frozen states
- `token_partition_summary.json`: state별 masked/reveal/remain indices, prediction, confidence
- `token_weights_summary.json`: state별 raw/normalized alpha 및 score scale diagnostics
- `reveal_abs_scores.json`, `reveal_square_scores.json`, `remain_abs_scores.json`, `remain_square_scores.json`
- `reveal_abs_mask.json`, `reveal_square_mask.json`, `remain_abs_mask.json`, `remain_square_mask.json`
- `mask_payloads/`: 896 packed matrix masks, 총 3.49 GB; 대용량이라 Git에서는 제외하고 manifest/checksum으로 추적
- `mask_diagnostics.json`: global/layer/module-type/matrix별 2,112 diagnostics
- `gsm8k_per_example_results.jsonl`: 6 methods × 1,319 = 7,914 aligned rows
- `paired_comparisons.json`: 8 preregistered paired comparisons
- `logs/final.json`: machine-readable final summary
- `report.md`: 본 보고서

## 17. 최종 결정 (Decision)

1. 현재 최고 방법은 **REVEAL-ABS 58.1501%**다.
2. **ABS를 DLM-gradient pruning의 reference aggregation으로 유지**한다.
3. REVEAL weighting은 후속 검증 가치가 있는 신호지만 통계적으로 확정된 개선으로 취급하지 않는다.
4. Wanda/SparseGPT보다 높은 관찰 정확도는 확인했지만 calibration-compute-matched 우월성으로 확대 해석하지 않는다.
5. EXP-004는 완료됐다. **EXP-005 또는 다른 후속 실험은 자동 실행하지 않았다.**
