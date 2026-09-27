# Projection gain proxy 검증 결정

## Hypothesis / Setup

224 projection에 E(s)=rowwise Standard Wanda 제거 score² 합,
g=기존 projection-only50 signed ΔCE/E50을 적용했다.
고정 80 corrupted states, 8 sequences,256tokens,LLaDA dense background,
native batch1 physical pruning을 사용했다. 초중후반 block0/1/15/16/30/31,
선택6projections×30/70 및6blocks50/3pairs50의21조건을 측정했다.
224개50 mask 완전재현, dense CE 재현, 모든 조건 weight복원/CE복원,
종료 dense SHA 일치. 추가 forward 없이 dense baseline에서 block update
비율/cosine을 저장하고 leave-one-block-out CPU 예측을 비교했다.

## Result

- Signed g가120/224에서음수. 기존 CE에대한 sequence bootstrap의
  pointwise95% CI는198/224에서0을포함(음수16/양수10). 이CI는탐색적이며
  여러projection검정에대한동시보장이아니다.
- 선택곡선12개 예측잔차의 family-wise Bonferroni conditional95% CI는
  전부0포함. 일치/등가성이증명된것은아니다.
- 후반 ff_out70: block30 관측0.00615624/예측0.00393015,
  block31 관측0.0305605/예측0.0241076. 점추정은관측대비약36.2%/21.1%
  과소예측이지만두잔차CI는0포함. 초기q/k및중간q에서부호차이도관측.
- Block31 전체50 손상0.0206610 vs projection손상합0.0259570.
  잔차−0.0052960, 동시CI[−0.0100106,−0.00125139]. 나머지5블록CI는0포함.
- 인접pair0–1/15–16/30–31 잔차 +0.0013902/−0.000721705/−0.00150867,
  각각CI는0포함. 초기subadditivity서사를지지하지않음.
- Depth+type LOBO g-MSE1.00983808e−9; update추가1.01899922e−9
  (0.907%악화),cosine추가1.01514372e−9(0.525%악화).

## Interpretation

1점보정의일관성은확립못했다. 높은sparsity점추정오차만으로비대각
Hessian원인을확정하거나SparseGPT reconstruction으로교체할수없다.
Signed calibrationCE는양의손상곡선의단일배율로해석하기어렵다.
특히E는증가하지만g<0이면gE는감소하고한계비용은더음수로커져
convex nondecreasing 비용에기반한μ이분탐색조건이깨진다.
이문제는표본변동과CE개선이함께있을수있어음수자동clamp로해결하지않았다.

Block31내비가산성은projection단독합의한계를보여주지만현재결과는
초기block간redundancy가아니다. Dense50결과를full65 sparse composition으로
일반화하지않는다. 이번whole-block두feature가projection g를더잘예측한다는
근거가없으며projection-specific후보전체를기각하는결과는아니다.

## Decision

- 초기redundancy 서사/인접layer 할인항은추가하지않는다.
- Update/cosine으로실측g를대체하지않는다.
- 이번측정으로reconstruction교체또는two-point형태를확정하지않는다.
- 현재signed CE50 gE를증가비용으로취급한μ allocator/mini를진행하지않는다.
  우선손상target의정의와보정의불확실성처리가필요하다.
  Raw signed CE와g는그대로보존하고CE를KL/positiveCE로자동교체하지않는다.
- 다음설계후보: CE손상의안정적인측정/보정(CE변화의baseline선형성,
  신뢰도또는별도양의functionaldamage target을명시적으로선정).
  별도target변경시새비교설정으로기록해야하며이번결과에소급적용하지않는다.

## Limits

8개 in-calibration sequence와identity-selected6projections의진단이다.
Bootstrap은fixed proxy와기존pruning을조건으로하며12곡선의극단quantile은
2000resamples로근사했다. 손상near-zero에서g비율이나상대오차를과대해석하지
않는다. Full WikiText PPL/heldout/GSM8K와DLM-vs-AR대조를측정하지않았다.
