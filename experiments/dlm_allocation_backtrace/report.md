# 기존 allocation의 후반 pruning 역추적

## Objective

OWL, DLP, AlphaPruning, LSA가 후반 레이어를 더 자르는 경로를 원시 통계,
projection→block 집계, importance 해석, sparsity 매핑으로 분리한다.
실제 allocation 생성 원인과 GSM8K 실패의 인과 원인은 구분한다.

## Setup / verification

- 고정 LLaDA-8B-Base revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2.
- 기존 80-state activation, 224 LSA metrics, Alpha fit summaries, super-outlier
  input/output energy, 최신 sequential65 manifests 재사용.
- CPU에서 checkpoint를 한 projection씩 읽어 Wanda score, count 및 norm을 계산.
  새 GPU, 모델 forward, SVD, pruning mask, allocation 성능 평가 없음.
- 6개 checkpoint shard SHA와 총254개 출처 해시 확인. 기존 DLP 평균 및 OWL
  threshold/count 수치 재현(상대 허용오차2e-6; count 절대 최소 허용5).
- 곱분해/공개매핑대수식/bootstrap 축 3tests 통과. 별도 verifier에서 rank,
  곱분해, 원시 통계–CE 상관 독립 재계산. verification.json 확인.
- 주 분해 CPU140.70초. 추가 CPU 검증/스펙트럼 에너지 계산 별도.

## 1. 수식이 배분 방향과 폭을 어떻게 결정했나

기본 block OWL/DLP/LSA의 정수 반올림 전 수식은 아래 형태다.

    s_l = S + sign * 2λ * (q_l - mean(q)) / (max(q) - min(q))

| 방법 | q | sign | 이상 sparsity 전체 폭 | 관측된 원시 통계 패턴 |
|---|---|---:|---:|---|
| OWL | block threshold를 초과하는 Wanda score 비율 | - |16%p|초기 높음, depth rho=-.540|
| DLP public mean path | pooled Wanda score 평균 | + |30%p|후반 높음, rho=+.999|
| LSA basic | 7개 projection surrogate 평균의 절댓값 | + |20%p|후반 높음, rho=+.997|

위 단순화와 실제 반올림 allocation의 최대 차이는 각각 .0186/.0164/.0181%p로,
정수 예산 보정이 큰 전후반 패턴을 만든 것은 아니다. 실제 원시통계→sparsity
rho는 OWL -.999817, DLP/LSA +1.0이다.

Alpha는 block 평균 α를 minmax 후 [.7,1.3]에 선형 매핑하고 파라미터 예산으로
재스케일한다. 순서는 보존되며 현재 전체 sparsity 폭은39.55%p다. 초기 낮은 α를
보호하고 후반 높은 α에 sparsity를 더 배정한다.

중요: minmax는 통계 차이의 절대 크기나 신뢰도를 보지 않는다. 양의 affine
변환 q'=a*q+b를 해도 최종 배분은 그대로다. 작은 차이라도 정해진 폭으로
확대될 수 있다는 구조이지, 현재 차이가 noise라는 증거는 아니다.

## 2. DLP: 증가폭은 activation 규모가 지배, 순서는 규모 제거 후에도 일부 유지

각 projection에서 아래 식을 수치적으로 정확히 분해했다.

    mean(|W| sqrt(A)) = mean(|W|) * mean(sqrt(A)) * alignment

alignment는 열별 weight 평균과 activation RMS 사이 정렬에 따른 잔여 배율이다.
Block에서도 파라미터 가중 평균 weight/RMS와 잔여 배율로 정확한 곱분해를 했다.

| 통계상 변환 | 후반8 / 초기8 평균 비율 | depth rho |
|---|---:|---:|
| 원래 score |7.405|+.9989|
| projection activation RMS 규모로 나눔 |1.100|+.9633|
| projection weight 규모로 나눔 |6.729|+.9978|
| 두 규모 모두로 나눔 |.954|-.9003|

Block log-score의 선형 depth slope는 .087262이다. 정확한 로그 곱분해에서
activation .083491(95.68%), weight .005831(6.68%), 잔여 -.002060(-2.36%)로
합산된다. 이 비율은 **관측된 log-depth 기울기의 대수적 분해**이지 성능 원인의
95.68%를 설명한다거나 인과적 설명력이라는 의미가 아니다.

결론: 원시 score가 후반으로 커지는 크기 차이는 activation 규모가 대부분이다.
그러나 activation 규모만 없애도 순서 rho는 .963으로 높다. 따라서 규모 정규화만
하면 후반 pruning이 자동으로 없어지거나 성능이 좋아진다고 말할 수 없다.
더구나 minmax 매핑은 축소된 통계 차이를 다시 고정 폭으로 늘릴 수 있다.

## 3. OWL: 단일 channel3848이나 block 공통 threshold만으로 설명되지 않음

모델 채널을 제거한 것이 아니라 score 통계에서만 해당 열을 제외했다.
3848은 residual channel을 직접 읽는 q/k/v/up/ff_proj에만 적용하고,
attn_out/ff_out의 같은 숫자 열을 동일 의미로 취급하지 않았다.

| OWL outlier 비율 계산 | B00–07 | B08–15 | B16–23 | B24–31 | depth rho |
|---|---:|---:|---:|---:|---:|
| 원래 block threshold |1.2634%|.4057%|.2718%|.4475%|-.5396|
| 3848 제외, 원 threshold 유지 |1.2553%|.3924%|.2586%|.4384%|-.5396|
| 3848 제외, threshold 재계산 |1.3140%|.4060%|.2654%|.4407%|-.5532|
| projection RMS 규모 표준화 후 block threshold |1.1680%|.3530%|.2697%|.4679%|-.4065|
| 각 projection 자체 평균으로 threshold 설정 |1.1918%|.3523%|.2644%|.4246%|-.4835|

초기8 block에서 3848 연결은 전체 OWL outlier count의 block평균0.99%만
차지했다. 초기 보호 패턴은 그 열을 제외해도 유지된다. 이는 그 채널의
functional 중요도가 작다는 뜻이 아니라, **OWL count 패턴을 그 하나가
직접 만들어냈다는 설명이 성립하지 않는다**는 뜻이다.

outlier 구성도 바뀐다. 초기 count 중 q21.13%, k20.84%, ff_proj21.93%,
up17.57%였지만 후반에서는 ff_out이50.15%를 차지했다. 같은 block 비율이
어떤 종류의 연결로 구성되는지가 depth에 따라 다르다. 이것이 성능의 원인인지,
type-aware 보호가 유리한지는 여기서 평가하지 않았다.

## 4. LSA: 규모 효과는 크지만 단순 규모·행렬 크기만의 현상은 아님

| 통계상 변환 후 block 집계 | 후반8 / 초기8 | depth rho |
|---|---:|---:|
| raw surrogate |42.230|+.9974|
| projection parameter 수로 나눔 |51.831|+.9956|
| projection 입력 채널당 energy로 나눔 |1.916|+.8284|
| projection 출력 energy로 나눔 |2.760|+.6932|

출력 energy는 기존80state에서 projection의 dense 출력 제곱합을 state 평균한 값.
위 상대값은 원시 통계를 해석하기 위한 진단이며 새 추천 score가 아니다.
Raw metric과 출력 energy의 projection rho=.833, 입력 energy와는 .807.
각 type 내부에서도 raw metric depth rho=.982~1.000이고, 어떤 한 type을
block 평균에서 빼도 depth rho=.997~.999가 유지된다.

따라서 하나의 큰 ff_out이 block 통계를 오염시켰다는 설명은 맞지 않는다.
전체적으로 후반에서 raw surrogate가 증가한다. 에너지로 나누면 증가폭이 크게
줄지만 패턴이 사라지지는 않는다. 입력/출력 규모와 나머지 구조의 영향이 함께 있다.

이 통계를 **큰 값→더 많이 prune**으로 변환하는 공개 basic mapping이 후반
pruning을 직접 만든다. 출력 상대값으로 바꾸는 것만으로 기능적 손상 예측이
좋아진다는 증거도 없었다(다음 표).

## 5. Alpha: 한 projection의 이상치가 아니라 넓은 depth 패턴

Block α 평균: 1.995 → 2.578 → 2.646 → 3.413, depth rho=.8999.
모든7type에서 depth rho가 양수(.588~.940). 어떤 type을 제외해도 block α의
depth rho=.859~.922로 유지된다. 따라서 단일 type 제거로 없어질 패턴이 아니다.

저장된 λmax와 CPU Frobenius energy로 stable rank=||W||F²/λmax를 계산했다.
초기→후반 quartile 평균은153.33,249.14,255.37,223.08이다. 최대고유모드의
energy 비중 평균은1.506%,.543%,.467%,.647%다. 초기 집중도는 상대적으로
높지만 depth 전체에 단조 관계는 아니다. Block α–stable rank rho=.346,
projection rho=.425; type별 .049~.769로 차이가 크다.

따라서 **낮은 α=낮은 stable rank=불필요한 레이어**로 바로 등치할 수 없다.
Weight spectral mode는 activation channel3848과도 별개다.
α fit D 범위 .027~.135, fitting tail 표본374~3099. 이 요약만으로 α 추정의
안정성/불안정성을 확정할 수 없으며, 전체 스펙트럼의 tail 재적합은 하지 않았다.

## 6. 기존 기능 손상과 연결되는가

기존 native batch1, 고정 Wanda50 mask, projection 하나씩 dense 배경에서
측정한 signed masked-gold CE 변화와 비교했다. Block target은7개 개별
projection 손상의 평균이며 joint block 손상이 아니다.

| Block proxy | CE 변화 rho | 조건부 탐색적95% CI |
|---|---:|---|
| OWL outlier 비율 |-.100|[-.285,.149]|
| DLP mean |+.444|[.093,.609]|
| LSA raw |+.434|[.089,.596]|
| LSA output-relative |+.199|[-.108,.358]|
| Alpha α |+.309|[.044,.433]|

8 sequence cluster,1000 bootstrap,seed2026, proxy는 고정. 다중 비교 미보정인
탐색적 구간으로 유의한 후보를 선택하는 검정에 쓰지 않는다. 모델/통계 추정
불확실성 전체를 반영하지 않는다. CE 변화는 음수도 그대로 보존한다.

큰 DLP/LSA/α 값에 더 많은 pruning을 배정하는 방향은 이50% 진단에서
더 큰 signed CE 손상과 함께 움직인다. **문제가 원시 통계의 무정보성보다
통계→중복성/보호 필요성 해석에 있을 수 있다**는 단서다. 규모 신호라고 해서
그 자체가 쓸모없다는 뜻은 아니다. 정규화는 손상과 관련된 정보도 제거할 수 있다.

## Conclusion / Decision

1. 같은 후반 pruning이어도 원인이 동일하지 않다.
2. DLP는 크기 차이가 activation 규모에 크게 의존한다. LSA는 에너지 효과 외에도
   넓은 type 공통 depth 패턴이 남는다.
3. OWL의 초기 outlier 집중은3848 제외나 자체 threshold로 없어지지 않는다.
4. Alpha는 초기 낮은 α를 보호하는 설계가 원인이지만, α를 redundancy로 해석할
   근거 또는 반대로 배분할 근거가 이번 스펙트럼 요약만으로 확정되지는 않는다.
5. 매핑은 통계의 불확실성/차이 크기와 무관하게 정해진16/20/30/약40%p 폭을
   부여한다. 순서 문제와 배분 폭 문제는 분리해서 다뤄야 한다.

새 generation/마스크/튜닝은 실행하지 않는다. 이 분석은 allocation 생성경로의
역추적이며,65% jointly sparse GSM8K 실패 원인의 인과적 증명이나 DLM-vs-AR
특유성의 증명이 아니다. 전체 eigenvalue 재적합은 남은 측정 공백이다.

## Artifacts

- results.json: 전체 분해/상관/조건부CI/source SHA
- collection.json, projection_features.csv: projection별 score와 outlier contribution
- spectral_concentration.json: α·stable rank·최대모드 energy
- verification.json: 독립 검증과 output SHA
- analyze.py, verify.py, test_core.py, run.log, verify.log
