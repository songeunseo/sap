# Allocation 실패 패턴: 사후 offline 분석

## 설정과 범위

동일224 projection·실제 제거수4,536,008,704·동일mini100 예측 검증. 새GPU/성능실험 없음. 모든 비율은 실제 정수 mask count 기준.
Layer 구간은 parameter 가중 평균. Role의 기존 projection 단순평균과 혼동하지 않는다. Obsidian 연결 불가로 로컬 기록만 저장.

## 전체 allocation

|방법|mini100|B00–07|B08–15|B16–23|B24–31|최대 sparsity|layer ρ|
|---|---:|---:|---:|---:|---:|---:|---:|
|Uniform|12|64.99%|64.99%|64.99%|64.99%|65.00%|NA|
|Role|24|69.56%|64.03%|62.44%|63.93%|75.00%|-0.662|
|Aggregate|19|69.47%|63.93%|62.54%|64.03%|75.00%|-0.556|
|OWL|5|59.90%|66.45%|67.48%|66.13%|67.64%|0.540|
|dlp|4|56.50%|61.89%|65.80%|75.77%|84.52%|0.999|
|dsa|7|56.05%|66.58%|68.83%|68.50%|72.05%|0.383|
|alpha|1|53.04%|63.55%|64.77%|78.60%|85.67%|0.900|
|lsa|6|60.01%|62.00%|64.79%|73.16%|79.72%|0.997|

## Projection type

|방법|attn_out|ff_out|ff_proj|k_proj|q_proj|up_proj|v_proj|
|---|---:|---:|---:|---:|---:|---:|---:|
|Uniform|64.99%|65.00%|64.99%|64.99%|64.99%|64.99%|64.99%|
|Role|55.92%|66.09%|72.34%|69.06%|62.49%|61.86%|56.55%|
|Aggregate|55.93%|66.40%|72.18%|69.53%|62.96%|61.39%|56.55%|
|OWL|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|
|dlp|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|
|dsa|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|
|alpha|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|
|lsa|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|64.99%|

## 기존 functional vulnerability와 예산 방향

Top56은 D65 또는65→70 추가 제거 parameter당 KL cost 상위25%. 양수는 Uniform보다 그 집합에서 더 제거했다는 뜻. 실제 joint KL 증가량이 아니다.

|방법|density vs D65 ρ|density vs marginal cost ρ|Top56 D65 추가제거(M)|Top56 marginal 추가제거(M)|75%초과 projection 수|
|---|---:|---:|---:|---:|---:|
|Uniform|NA|NA|0.00|0.00|0|
|Role|0.108|0.576|54.57|-62.09|0|
|Aggregate|0.144|0.582|44.49|-64.60|0|
|OWL|-0.261|-0.389|27.48|6.42|0|
|dlp|-0.408|-0.589|96.89|82.35|28|
|dsa|-0.255|-0.405|48.52|37.83|0|
|alpha|-0.398|-0.534|142.73|84.04|42|
|lsa|-0.412|-0.597|66.84|60.72|14|

## D65 상위 projection

|projection|D65|Uniform|Role|Aggregate|OWL|dlp|dsa|alpha|lsa|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|block_31.ff_proj|0.05801|64.99%|75.00%|75.00%|62.06%|84.52%|69.43%|78.44%|79.71%|
|block_31.up_proj|0.03187|64.99%|75.00%|70.00%|62.06%|84.52%|69.43%|78.44%|79.71%|
|block_31.ff_out|0.01604|65.00%|75.00%|75.00%|62.05%|84.52%|69.43%|78.45%|79.72%|
|block_00.v_proj|0.00767|64.99%|54.98%|54.98%|51.64%|54.71%|56.05%|46.12%|59.77%|
|block_30.up_proj|0.00701|64.99%|64.99%|64.99%|65.41%|80.15%|68.60%|85.67%|77.71%|
|block_30.ff_proj|0.00689|64.99%|70.00%|70.00%|65.41%|80.15%|68.60%|85.67%|77.71%|
|block_29.ff_proj|0.00525|64.99%|64.99%|64.99%|66.46%|77.59%|68.31%|83.30%|74.78%|
|block_29.up_proj|0.00514|64.99%|59.99%|59.99%|66.46%|77.59%|68.31%|83.30%|74.78%|
|block_12.ff_proj|0.00495|64.99%|75.00%|75.00%|66.82%|61.91%|69.85%|62.45%|61.99%|
|block_11.ff_proj|0.00478|64.99%|70.00%|70.00%|66.80%|61.45%|70.26%|63.79%|61.69%|

## Paired mini100 결과

|방법|새 정답|소실 정답|exact p|Holm(4)|
|---|---:|---:|---:|---:|
|dlp|4|12|0.076813|0.230438|
|dsa|3|8|0.226562|0.359131|
|alpha|1|12|0.003418|0.013672|
|lsa|4|10|0.179565|0.359131|

## 해석상의 제한

- Post-hoc descriptive; no causal intervention.
- Capacity curves single-module dense-background; not joint-model damage.
- No extrapolation outside50-75; D65 and marginal sensitivity only characterize budget direction.
- Small selected set of methods, no cross-method accuracy correlation inference.
- Same mini repeatedly used for development; no fresh confirmatory holdout.
- Uniform density 상관은 정수 row-floor 차이만 반영하므로 NA로 처리한다.
- DSA NaN→0 blocks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
- Layer pattern, projection별 granularity, sparsity 범위, proxy가 함께 다르므로 Role의 성능 차이를 masked/unmasked 분리 하나로 귀속할 수 없다.
- 불량 candidate를 보고 curve나allocation을 수정하지 않았다.

## 핵심 해석과 다음 판단

1. 네 새 baseline 모두 early 보호/late 추가 제거 패턴. Role/Aggregate는 앞8block에서 더 많이 제거한다. 배분 방향 불일치가 공통 단서다.
2. DLP/Alpha/LSA는75%를 넘는 projection이 존재한다. 동일 global budget이 동일 local aggressiveness를 의미하지 않는다.
3. Role은 모든 late projection을 보호하지 않는다. D65 상위 B31 MLP 세 개는 모두75%다. 따라서 취약 late MLP 보호로 Role 성공을 단정할 수 없다.
4. Role은 type별로 attn_out/v를 상대적으로 보호한다. 네 baseline은 같은block 안7projection에 거의 동일 sparsity라 이런 차등 배분이 없다.
5. D65 절대 damage와 parameter당 marginal cost는 다른 기준이다. Role도 D65 상위56개에서 Uniform보다 더 제거하지만 marginal 상위56개에서는 덜 제거한다.
6. Aggregate도 비슷한 depth/type 구조를 가지며19점이다. 이 비교는 masked/unmasked 또는 max의 고유 효과를 식별하지 않는다.
7. 다음 개입으로 layer 총예산을 유지한 type 균일화가 within-layer allocation 효과를 분리하는 통제가 될 수 있다. 실행/후보 선택은 하지 않았다.
8. 이번 결과로 전체 AR allocation 실패나 특정부위 pruning의 인과효과를 확정하지 않는다.
