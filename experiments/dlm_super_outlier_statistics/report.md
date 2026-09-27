# LLaDA Super-Outlier Projection Statistics

## Hypothesis

Channel 3848 의존도가 projection별 pruning tolerance를 layer/type 및 local reconstruction 이상으로 설명하는지 검증한다. 동시에 3848을 제외한 나머지 channel 집중도가 더 일반적인 설명인지 비교한다.

## Actual Setup

Frozen LLaDA-8B-Base 80 corruption states에서 dense forward 통계를 수집하고, 기존 224×6 single-projection capacity curve와 결합했다. 모델 channel이나 weight는 제거하지 않았다.
분석 대상은 224 projections이며, residual channel과 직접 연결되는 read 통계는 q/k/v/ff_proj/up_proj 160개, write 통계는 attn_out/ff_out 64개다. Spearman 상관과 layer/type 고정효과 및 local reconstruction 통제 상관을 계산했고, 8 sequences와 32 layers를 cluster 단위로 재표집하여 2,000회 bootstrap CI를 얻었다.

## Results — observed statistics

Channel 3848 residual energy share는 layer 0 attention residual에서 0.6345, block-input 기준 최대 0.7219 (layer 5), 마지막 block output에서 0.0051였다.

단일 channel 지표는 raw D_g(65)와 강하게 연결됐지만 통제 후 사라졌다:

- input super share: raw rho=-0.5110, controlled rho=0.0423, 95% CI=[-0.2053, 0.2344]
- read contribution: raw rho=-0.4416, controlled rho=0.0531, 95% CI=[-0.1236, 0.2225]

반면 channel 3848을 제외한 입력 집중도는 통제 후에도 관계가 남았다:

- excluded dominant share vs D_g(65): rho=-0.3860, 95% CI=[-0.5514, -0.1364]
- excluded effective fraction vs D_g(65): rho=0.3598, 95% CI=[0.1007, 0.5535]
- excluded dominant share vs 65→70 marginal cost: rho=-0.3611, 95% CI=[-0.5167, -0.0860]
- excluded effective fraction vs 65→70 marginal cost: rho=0.2917, 95% CI=[0.0055, 0.4781]

## Main Associations

| Feature | Raw ρ with D(65) | Partial layer/type ρ | + reconstruction ρ |
|---|---:|---:|---:|
| input_excluded_dominant_share | -0.3308 | -0.5233 | -0.3860 |
| input_excluded_effective_fraction | 0.5223 | 0.4974 | 0.3598 |
| input_excluded_top1pct_share | -0.7176 | -0.4711 | -0.2991 |
| input_super_rank | 0.1865 | -0.2776 | -0.2938 |
| input_dominant_share | -0.4229 | 0.0325 | 0.2689 |
| write_super_cross_over_after | -0.1011 | 0.4476 | 0.2483 |
| input_effective_fraction | 0.4580 | 0.0529 | -0.2144 |
| input_super_token_argmax_fraction | -0.3041 | -0.0550 | 0.1534 |
| input_outlier7_ratio | -0.5747 | -0.1419 | 0.1387 |
| input_super_mean_sq_ratio | -0.3460 | -0.2515 | -0.1348 |
| input_top1pct_share | -0.5656 | -0.2159 | 0.0908 |
| write_super_branch_over_after | 0.5065 | 0.2124 | 0.0556 |
| read_super_component_over_output | -0.4416 | -0.0439 | 0.0531 |
| output_super_share | 0.0551 | 0.0836 | 0.0486 |
| input_super_share | -0.5110 | -0.1445 | 0.0423 |
| input_excluded_outlier7_ratio | -0.7445 | -0.1883 | 0.0289 |

## Interpretation

단일 super-outlier는 LLaDA의 depth/type 구조를 강하게 표시하지만, 그 자체가 projection tolerance를 독립적으로 설명한다는 증거는 없다. 더 흥미로운 관찰은 3848을 제거하고 계산한 나머지 channel 분포다. 집중도가 높은 read-projection일수록 D_g(65)와 65→70 marginal damage가 낮은 방향이어서, 이 데이터에서는 outlier가 많을수록 보호한다는 단순 OWL 직관보다 low-dimensional redundancy 해석과 더 잘 맞는다.

이 관계는 layer/type 및 reconstruction 통제와 cross-sequence split에서 유지되지만, 160개 read-projection에 한정된 다중 탐색 결과다. 인과관계나 downstream gain, 새 pruning 방법을 아직 입증하지 않는다.

## Decision

- channel 3848 단독 기반 pruning score/allocation: NOT SUPPORTED.
- channel 3848 제외 activation concentration: FOLLOW-UP DIAGNOSTIC WARRANTED.
- 다음 단계는 기존 Wanda score와의 redundancy 및 실제 mask 변화량을 측정하는 작은 diagnostic이다. 이 검증 전에는 새 allocation이나 full downstream 평가를 만들지 않는다.
