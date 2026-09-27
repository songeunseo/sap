# 기존 scalar A+C를 발전시키기 위한 이론 문헌 조사

- Date: 2026-09-22
- Status: literature review / mathematical derivation / untested method candidates
- Obsidian: 이번 작업 시작 때 get_sync_status와 Research-State read_note 모두 정상 응답.
- 실행 범위: 문헌 검토와 작은 유한공간 산식 검산만 수행. 모델 forward, GPU 실험, pruning, GSM8K/NELBO 평가 없음.
- 관련 코드: experiments/dlm_context_response50/core.py, run.py, config.json
- 세부 조사: [그래프](/home/tmluser1/sap/research/theory_ac_graph_2026-09-22.md), [조건부 확률](/home/tmluser1/sap/research/theory_ac_conditionals_2026-09-22.md), [제어 이론](/home/tmluser1/sap/research/theory_ac_dynamics_2026-09-22.md)

## 판단

기존 A+C에서 발전시킬 가장 유용한 부분은 **C의 입력 쌍을 어떤 분포와 어떤 크기의 문맥 변화로 구성하는가**이다. 기존 scalar readout이나 allocation solver를 모두 교체할 필요는 없다.

이번 조사에서 우선순위를 다음처럼 잡는다.

1. **여러 mask 변화 규모에서 dense–sparse 반응 오차를 측정하는 A+C**: Boolean noise operator로 pair를 설계하면 C가 어떤 차수의 문맥 의존성 오류를 강조하는지 정확히 해석할 수 있다. 이론과 구현 사이의 연결이 가장 명확한 새 탐구 방향이다.
2. **네 개의 연결된 reveal 상태를 사용하는 A+C**: 두 문맥 단서를 각각/함께 공개해 반응을 비교한다. 상호작용 손상을 관찰하기 좋은 작은 설계이며, 기본 버전은 새 loss 항 없이 구성 가능하다.
3. probability/Bregman readout, observability/adjoint, graph sparsification은 보조 후보다. 현 단계에서 더 우수한 main method라는 근거가 없다.

이 우선순위는 성공 확률의 측정값이 아니다. 새 pair 분포가 실제 decoding과 멀어질 위험 때문에 noise-operator 설계는 탐구 우선순위이며, 기존 A+C를 대체하도록 확정하지 않았다. 그래프 조사 에이전트는 이 이유로 noise 설계를 우선 진단용으로 둘 것을 권했다. Root는 분포와 스펙트럼의 연결을 명시적으로 검증할 수 있다는 점에서 method 후보로 남긴다.

## 1. 출발점과 기존 증거

정답 토큰 y에 대한 scalar:

\[
f_X(x)=\log\frac{p_X(y\mid x)}{1-p_X(y\mid x)},\quad
e(x)=f_M(x)-f_D(x).
\]

\[
A=\mathbb E[(e(x_0)^2+e(x_1)^2)/2],\qquad
C=\mathbb E[(e(x_1)-e(x_0))^2].
\]

D는 dense, M은 static sparse model. 같은 query 위치를 계속 masked로 두고 x1에서 다른 일부 gold token을 추가 공개한다.

실제 bank는 8개 WikiText span × 10개 mask 설정의 80 pairs이며, reveal 수는 현재 조건에 따라 1–13개다. **여러 mask rate를 본 것과 여러 크기의 문맥 변화를 본 것은 다르다.**

48/52% layer probe를 Uniform50 배경에서 측정하고, marginal cost 순위를 45–55% sparsity로 매핑한다. 마지막 정수 DP는 budget rounding 목적이다. AC 자체를 전역 최적화하는 solver로 설명하지 않는다.

관측된 같은 native pipeline mini100은 Uniform54, A55, AC61이다. full AC는 사용자 요청으로 중단되었고, 독립 NELBO 결과는 없다. 다른 cached pipeline의 Uniform62와 섞어 최고 성능을 주장하지 않는다. 이 결과는 response preservation을 계속 연구할 이유이지만 아래 후보의 실험 근거는 아니다.

기존의 직접 유도:

\[
u=(e_0+e_1)/2,\quad v=(e_1-e_0)/2
\quad\Longrightarrow\quad A+C=\mathbb E[u^2+5v^2].
\]

즉 C는 완벽한 endpoint matching에 없는 정보를 만들어내기보다, 제한된 static mask가 어떤 오류를 더 줄여야 하는지 정한다. 새 후보에도 같은 기준을 적용한다.

## 2. AlphaPruning에서 가져올 연구 방식

[AlphaPruning](https://arxiv.org/html/2410.10912) §3–4와 Appendix E를 원래 HT-SR 논문들과 대조했다. weight spectrum의 shape를 학습 상태와 연결하고, 이를 layer sparsity에 매핑하는 구조다. 원래 이론에서 특정 sparsity allocation이 최적이라는 정리가 바로 나오지는 않는다.

원래 근거는 [Mahoney–Martin, ICML2019](https://proceedings.mlr.press/v97/mahoney19a.html)의 스펙트럼/self-regularization 관찰과 [Martin–Peng–Mahoney, Nature Communications2021](https://arxiv.org/html/2002.06716)의 pretrained model quality 분석이다.

우리에게 적용할 연구 구조:

> 관측 가능한 DLM 기능 → 그 기능의 손상을 읽는 이론적 좌표 → static budget allocation → 독립적인 성능 검증.

복잡한 optimizer가 필수는 아니다. 대신 이론이 설명하는 측정량과 실제 보존하려는 기능 사이의 연결이 있어야 한다. 임의 통계량을 논문 이름으로 포장하는 것으로는 부족하다.

## 3. 우선 후보: 여러 mask 변화 규모에서의 반응 보존

### 3.1 이론 출처와 설계 변경

[BKS1999](https://arxiv.org/abs/math/9811157)의 noise sensitivity와 [O'Donnell, Analysis of Boolean Functions](https://arxiv.org/abs/2105.10386) §2.4, §8.2–8.4의 product-space noise operator가 핵심 근거다. 여기서 빌리는 것은 correlated input의 차이와 Fourier/ANOVA 차수 사이의 항등식이다. percolation이나 Boolean 분류 결과를 DLM 성능 정리로 옮기지는 않는다.

최근 [Noise Stability of Transformer Models, ICLR2026](https://arxiv.org/html/2602.08287)도 Transformer 분석/학습에 noise stability를 활용한다. 그러므로 “noise stability를 Transformer에 처음 적용”한다는 주장은 불가능하다. 이 논문의 §6은 model output stability를 학습시키며, Appendix J.3의 언어 실험은 작은 모델 설정이다. 이를 LLaDA pruning의 성능 근거로 사용하지 않는다.

**우리 adaptation**은 자연 문장의 token value를 랜덤 단어로 바꾸는 대신 visible/masked bit를 조작하고, model 자체의 안정성보다 **dense–sparse 오차의 문맥 의존성**을 제어하는 것이다.

### 3.2 정확한 구성

한 clean sequence와 계속 masked로 유지할 query Q를 고정한다. 나머지 eligible context token의 공개 여부를 v∈{0,1}^n으로 표시한다.

1. phase별 공개 확률 p∈(0,1)를 정하고 v를 independent Bernoulli(p)로 샘플한다.
2. v'의 각 bit는 확률 ρ로 v와 같게 유지하고, 나머지는 새 Bernoulli(p)로 다시 뽑는다.
3. 입력 x(v), x(v') 모두 같은 clean sequence의 mask corruption이며 Q는 공개하지 않는다.
4. 두 입력에서 기존 f와 e를 그대로 계산한다.

두 mask의 주변분포는 동일하다. 작은 ρ는 더 큰 mask 재배치를 만든다. 총 mask 수가 정확히 같다는 뜻은 아니고, v→v'가 단조로운 reveal trajectory라는 뜻도 아니다. reveal과 re-mask가 모두 가능하다.

### 3.3 무엇을 측정하는지 유도

고정 sequence/query의 residual e를 p-biased orthonormal basis에 전개하자.

\[
e(v)=\sum_{S\subseteq[n]}\hat e_p(S)\phi_{p,S}(v),\qquad
E_k=\sum_{|S|=k}\hat e_p(S)^2.
\]

여기서 k는 **mask bit에 대한 함수 의존성의 차수**다. 실제 추론 단계 수나 의미적 reasoning depth가 아니다.

\[
A_p=\mathbb E_v[e(v)^2]=\sum_{k=0}^n E_k.
\]

위에서 정의한 정확한 correlated-resampling coupling에서는

\[
\boxed{
C_{p,\rho}
=\mathbb E[(e(v')-e(v))^2]
=2\sum_{k=1}^n(1-\rho^k)E_k.
}
\]

증명은 같은 주변분포로부터 C=2(E[e²]−E[ee'])를 쓰고 noise operator의 고유값 ρ^k를 대입하면 된다.

따라서 C가 어떤 종류의 오류를 더 강조하는지는 pair의 변화 규모에 의해 결정된다. 단계 크기를 정규화하면

\[
\widetilde C_{p,\rho}=
\frac{C_{p,\rho}}{2(1-\rho)}
=\sum_{k\ge1}(1+\rho+\cdots+\rho^{k-1})E_k.
\]

- ρ=0이면 비상수 residual variance를 읽는다.
- ρ→1이면 차수 k에 비례하는 오류 가중치에 가까워진다.
- 작은 변화에 대한 response matching을 강하게 정규화하면 고차 문맥 의존성 오류를 더 강조한다.
- 이것은 고차 interaction을 모두 보존하는 것이 좋다는 결론이 아니다. 어떤 spectrum이 실제 성능에 중요한지는 아직 모른다.

**기존 gold-reveal pair에는 이 식을 그대로 적용할 수 없다.** endpoint mask 분포가 달라지고, coupling도 다르다. 이 항등식을 쓰려면 위 bank를 별도로 만들어야 한다.

### 3.4 제안 objective

\[
\boxed{
L_{\mathrm{scale}}(M)=
\mathbb E_{p}\!\left[
A_p(M)+
\lambda\mathbb E_{\rho\sim\nu}
\widetilde C_{p,\rho}(M)
\right].
}
\]

첫 비교에서는 기존 scalar f, surviving weights, Wanda ranks, sparsity budget, allocator를 유지한다. 새로운 것은 pair coupling과 ρ 분포다. 먼저 두 개의 명시된 변화 규모만 사용하면 충분하다. 구체적 ρ·λ는 아직 확정하거나 실행하지 않았다.

ρ를 여러 개 썼다는 사실 자체가 장점은 아니다. 비교하려는 오류 차수를 어떻게 가중하는지 미리 설명하고, λ만 조절한 baseline과 구분해야 한다. raw C와 normalized C도 같은 방식으로 평가하지 않으면 단순 loss scale 효과가 섞인다.

**가설:** 한 종류의 작은 reveal에 맞춘 allocation은 다른 문맥 재배치 크기에서 response fidelity를 잃을 수 있다. 여러 규모에서 residual의 문맥 의존성을 제한하면 static mask가 더 넓은 masked conditional family를 보존할 수 있다.

**실패할 이유:** stationary re-mask pairs가 실제 생성 중의 단조로운 reveal보다 덜 관련될 수 있다. 큰 재배치는 평범한 endpoint matching에 가까워질 수 있고, 작은 재배치의 정규화는 희귀한 큰 오차에 민감하다. gold-vs-rest scalar의 wrong-token blind spot도 남는다.

**계산:** 한 pair는 여전히 endpoint 두 개다. pair 총수를 고정해 규모별로 나누면 forward 수를 반드시 늘릴 필요는 없다. 그 대신 규모별 표본 수가 줄어 추정 variance가 증가한다. 기존 bank와 새 bank를 모두 추가하면 비용은 실제 endpoint 수에 비례해 증가한다.

**중요한 차이:** teacher의 noise-stability scalar만 따라 맞추면 안 된다. 서로 다른 token dependence가 같은 aggregate stability를 가질 수 있다. 위 식은 동일 입력의 e=fM−fD에 적용하여 teacher–student 대응을 유지한다.

## 4. 두 번째 후보: 네 상태에서 조건부 반응을 비교

### 4.1 이론 출처

[Shapley–Taylor, ICML2020](https://proceedings.mlr.press/v119/sundararajan20a.html)는 set function의 mixed finite difference를 interaction 분석에 사용한다. [Faith-Shap, JMLR2023](https://jmlr.org/papers/v24/22-0202.html)는 coalition 분포와 근사 기준에 따라 interaction attribution이 달라짐을 분명히 한다.

우리는 전체 Shapley 값을 계산하거나 그 attribution 공리를 새 loss의 보장으로 주장하지 않는다. 빌리는 것은 두 evidence group의 결합 효과를 구분하는 유한차분이다.

### 4.2 실제 비교

기본 visible context S와 disjoint reveal group R1,R2를 고르고, 같은 masked query에 대해 네 상태를 만든다.

| 상태 | 추가 공개된 문맥 |
|---|---|
| x00 | 없음 |
| x10 | R1 |
| x01 | R2 |
| x11 | R1와 R2 |

\[
I_X=f_X(x_{11})-f_X(x_{10})-f_X(x_{01})+f_X(x_{00}).
\]

이는 R2의 효과가 R1의 존재에 따라 얼마나 달라지는지다. 예컨대 수량과 단가를 각각 봤을 때와 함께 봤을 때의 반응을 분리하는 형태다. 이 예시는 설명용이며 모델 관측 결과가 아니다.

기존 C도 여러 token을 함께 reveal하므로 interaction 효과를 포함할 수 있다. 차이는 **각각의 효과와 함께 공개했을 때의 효과를 분리해서 볼 수 있느냐**다.

### 4.3 처음부터 고차 loss를 추가할 필요는 없다

\[
A_\square=\tfrac14(e_{00}^2+e_{10}^2+e_{01}^2+e_{11}^2)
\]

\[
C_\square=\tfrac14[
(e_{10}-e_{00})^2+(e_{01}-e_{00})^2+
(e_{11}-e_{10})^2+(e_{11}-e_{01})^2].
\]

첫 candidate는 그냥 A_square+C_square다. 네 상태를 기존과 같은 방식으로 읽고, 실제로 하나의 reveal group을 추가하는 네 edge의 response를 보존한다.

네 residual의 orthogonal decomposition을 e_ab=u+αs_a+βs_b+γs_as_b, s0=−1,s1=1로 쓰면,

\[
A_\square=u^2+\alpha^2+\beta^2+\gamma^2,\qquad
C_\square=2\alpha^2+2\beta^2+4\gamma^2.
\]

따라서 A_square+C_square=u²+3α²+3β²+5γ²이다. 별도 interaction penalty (I_M−I_D)²=16γ²는 **이미 존재하는 interaction mode의 가중치를 더 올리는 것**이다. 무조건 추가하면 이중 강조가 될 수 있다.

실제로 다른 문맥 단서의 효과를 무시해도 x00와 x11에서 오차가 상쇄되는 경우가 있다. 네 상태는 그 상쇄를 발견할 수 있다. 그러나 네 endpoint만 사용하는 A_square도 새 정보를 얻으므로 반드시 같은 bank에서 A_square와 AC_square를 비교해야 한다.

### 4.4 직접적인 반대 근거와 한계

[A Unified Approach to Interpreting KD for LLMs via Interactions](https://arxiv.org/html/2607.08776) §3–5는 Harsanyi interaction을 이용해 KD를 분석하고 Complex Interaction Penalty를 제안한다. 모든 복잡한 interaction의 무조건적 보존을 지지하는 결과가 아니다. 강한 penalty가 유용한 interaction도 억제할 수 있다는 trade-off도 보고한다.

따라서 “DLM이 복잡하니 고차 interaction을 많이 보존하자”를 가설로 삼지 않는다. 먼저 어떤 reveal interaction의 손상이 독립 성능과 연결되는지 확인해야 한다.

추가 한계:

- R1/R2가 여러 token이면 group interaction은 token-order 2와 같지 않다.
- f가 mask 개수의 비선형 함수이기만 해도 I가 0이 아닐 수 있다. 이를 순수 semantic interaction이라고 부르지 않는다.
- e10−e01은 equal-size R1/R2일 때 additive count-only effect를 상쇄하지만, mask 위치와 evidence 내용의 영향까지 분리하지 않는다.
- input-token interaction과 layer-pruning interaction은 다른 대상이다. 이전 cross-layer quadratic contribution이 작았다는 결과를 반박하는 주장이 아니다.
- 두 reveal 경로의 scalar increment 총합 차이는 항등적으로 0이다. 이것을 path consistency/curl loss로 제안하면 아무 것도 측정하지 못한다.

**계산:** 네 상태는 네 forward가 필요하다. 2개의 독립 pair도 네 상태이므로 같은 endpoint 예산에서 재배치할 수 있다. 그러나 독립 문맥 표본 수가 줄어드는 trade-off가 있다.

## 5. 읽어봤지만 main method로 채택하지 않은 방향

| 방향 | 얻은 것 | 우선 채택하지 않는 이유 |
|---|---|---|
| Manifold/Dirichlet graph | 기존 C 자체가 residual graph energy라는 정확한 해석 | graph 이름을 붙이는 것만으로 새 방법이 되지 않음. 같은 문장·query·visible set은 같은 입력 |
| Effective resistance / BSS | 이미 정의한 graph proxy를 sparse edge로 근사하는 보장 | 모델 forward 비용은 주로 node에 있음. edge thinning만으로 이를 줄인다는 보장 없음 |
| Proper scoring / Bregman | probability-space와 logit-space가 강조하는 오차 차이를 구분 | teacher logit matching도 정확하면 같은 probability 복원. Brier가 pruning에 우월하다는 결론 없음 |
| Martingale / conditional compatibility | 진짜 posterior의 조건부 평균과 reveal law를 엄밀히 구분 | 고정 gold 문장에서 위치만 바꿔 평균내는 것은 posterior-content expectation이 아님 |
| Balanced truncation / observability | 현재 작은 오차와 미래 목표에 큰 오차를 구분하는 관점 | transformer weight deletion은 stable LTI state reduction이 아님 |
| Adjoint / goal-oriented estimation | 후보 변화의 end-to-end 영향을 근사하는 방법 | 지금 AC는 이미 최종 출력으로 측정. 같은 output의 Jacobian을 붙여도 새 정보가 생기지 않음 |
| Contraction | 특정 smooth trajectory의 local robustness 설명 | hard decode가 branch를 바꾸고 호출 사이에 hidden state가 그대로 전달되지 않음 |
| Optimal design / sensor placement | 중복이 적고 feature coverage가 좋은 calibration bank | feature model이나 submodularity 가정이 아직 없음. 더 좋은 allocation의 직접 근거 아님 |
| Smart Predict-then-Optimize | proxy MSE보다 최종 선택의 regret을 목표로 삼는 관점 | 실현할 true decision cost가 추가로 필요. 기존 commitment/rollout 제안과 중복 |

현재 gold token y를 sample 이후에 고르는 scalar는, reveal 전에 고정한 event의 posterior와도 구분해야 한다. **p(Y=y)의 martingale 정리를 observed-gold readout에 자동 적용하지 않는다.**

## 6. 최소 비교 설계: 아직 실행하지 않은 제안

후보를 한꺼번에 합치지 않는다.

**Noise-operator candidate를 먼저 검토하는 경우:**

1. 같은 새 endpoint bank에서 A-only, full-output KL, 기존 scalar AC 형태를 비교한다.
2. endpoint 주변분포를 유지하면서 coupling 규모만 바꿔 비교한다. 여러 규모와 한 규모의 총 endpoint/forward 예산을 맞춘다.
3. natural nested-reveal AC는 원래 control로 유지한다. bank 변경 효과와 loss 변경 효과를 각각 보고한다.
4. exact mask의 held-out response와 독립 quality를 확인한다. calibration spectrum만 좋아지면 method 개선으로 판정하지 않는다.

**네 상태 candidate의 경우:**

- A_square → A_square+C_square가 핵심 비교.
- 별도 mixed interaction penalty는 첫 버전에 넣지 않는다.
- 같은 네 상태에서 비교해야 추가 context 관측 효과를 C의 효과로 오인하지 않는다.
- 임의의 wrong token 치환이나 teacher confidence 기반 sampling은 첫 비교에 섞지 않는다.

모델/revision, corpus, seed, query, sparsity scope, surviving weights, decoding, sample 수는 기록해 통제한다. 새 bank는 기존 설정의 변경이므로 구현/실행 전에 명확히 알린다.

**기각 기준:** 같은 예산에서 endpoint matching/일반 KL보다 나은 독립 quality가 없고, 새로운 response 구조가 실패 사례를 더 설명하지도 못하면 이 확장의 우선순위를 내린다. 더 복잡한 loss로 계속 덮지 않는다.

## 7. 신규성 판단

이미 있는 것:

- derivative/response matching, Dirichlet regularization, Boolean noise stability, Shapley/Harsanyi finite differences.
- Transformer noise-stability regularization과 interaction 기반 KD.
- DLM의 masked conditional prediction이라는 문제 구조.

아직 검증할 연구 기여:

> DLM pruning에서 한 static mask가 보존해야 하는 conditional response를 mask perturbation의 구조와 규모로 정의하고, 그 선택이 어떤 기능 손상을 줄이며 어떤 allocation을 만드는지 보이는 것.

이는 논문으로 발전 가능한 구성이다. 모든 AR 방법이 Uniform보다 나쁘거나, 식이 오직 DLM에서만 정의되어야 할 필요는 없다. 반대로 새로운 이론 이름을 붙였다는 이유만으로 독창성이 확보되지는 않는다. 이론적 측정 정의, 실제 손상 패턴, 독립 성능이 연결되어야 한다.

## 8. 문헌 범위와 읽기 깊이

중복 AlphaPruning을 제외해 **논문 34편 + 이론서 1권의 관련 부분**을 검토한 묶음이다. 전부 전체 정독했다는 뜻이 아니다. H는 관련 method/theorem/appendix까지 읽음, T는 핵심 결과/적용 조건 중심, S는 서지·요약 중심이다. 이론서는 지정 절만 읽었다. 각 에이전트 노트의 접근 제한/수정 기록을 함께 본다.

### Root: 이론의 출발점, interaction, noise operator, 의사결정

| # | 문헌 / primary source | 읽은 범위 | 설계에 기여한 점 |
|---|---|---|---|
| 1 | [AlphaPruning (2024)](https://arxiv.org/html/2410.10912) | H: §3–4, allocation ablation | 이론→측정→allocation→실험의 연결 |
| 2 | [Traditional and Heavy Tailed Self Regularization (2019)](https://proceedings.mlr.press/v97/mahoney19a.html) | H: phase model, §2/4 | 원래 HT 이론의 적용 범위 |
| 3 | [Predicting trends in quality… (2021)](https://arxiv.org/html/2002.06716) | H: shape/scale methods, §2.3 | quality correlation과 pruning 중요도의 차이 |
| 4 | [Smart “Predict, then Optimize”](https://arxiv.org/abs/1710.08005) | H: SPO/SPO+, Proposition 3 | 예측량보다 선택 결과의 regret |
| 5 | [Near-Optimal Sensor Placements in Gaussian Processes (2008)](https://www.jmlr.org/papers/volume9/krause08a/krause08a.pdf) | H: §3–4, Theorem 7 | sampling coverage; 일반적인 monotonicity 가정 금지 |
| 6 | [Combinatorial Algorithms for Optimal Design (2019)](https://proceedings.mlr.press/v99/madan19a.html) | T: D/A optimality, local search theorem | calibration 설계와 mask search 분리 |
| 7 | [Shapley–Taylor Interaction Index (2020)](https://proceedings.mlr.press/v119/sundararajan20a.html) | H: finite difference, axioms, theorem | 4-state interaction |
| 8 | [Faith-Shap (2023)](https://jmlr.org/papers/v24/22-0202.html) | H: weighted approximation, counterexample | 기준 coalition/분포의 중요성 |
| 9 | [Interpreting LLM KD via Interactions (2026)](https://arxiv.org/html/2607.08776) | H: §3–5, CIP approximation | 직접 선행연구 및 무조건 보존에 대한 반대 근거 |
| 10 | [Noise Sensitivity of Boolean Functions… (1999)](https://arxiv.org/abs/math/9811157) | T: §1.1/1.2/1.5, spectral setup | input perturbation과 interaction order |
| 11 | [Analysis of Boolean Functions](https://arxiv.org/abs/2105.10386) | H: Thm2.49, Prop8.28, §8.4 | biased product mask의 정확한 noise identity |
| 12 | [Noise Stability of Transformer Models (2026)](https://arxiv.org/html/2602.08287) | H: §3–6, AppD/J.3 | closest Transformer prior, 규모와 empirical 한계 |

Noise Stability 논문의 본문 Lemma 1은 log의 밑/부호 표기가 Appendix D와 일치하지 않는다. 여기서 사용한 식은 해당 bound를 복사하지 않고 O'Donnell의 product-space 항등식에서 직접 유도했다. 또한 그 논문의 random-token regularizer의 ρ parameterization을 우리의 bit-retention ρ와 동일시하지 않는다.

### Graph/design 에이전트: AlphaPruning 중복 제외 7편

| # | 문헌 / primary source | 검토 중심 |
|---|---|---|
| 13 | [Belkin et al., Manifold Regularization (2006)](https://jmlr.org/papers/volume7/belkin06a/belkin06a.pdf) | Dirichlet objective, representer 조건 |
| 14 | [Spielman–Srivastava, Effective Resistances (2008)](https://www.cs.cornell.edu/~abrahao/tdg/papers/p563.pdf) | all-vector quadratic-form approximation |
| 15 | [Batson–Spielman–Srivastava, Twice-Ramanujan Sparsifiers](https://doi.org/10.1137/090772873) | weighted sparse graph theorem |
| 16 | [Führ–Pesenson, Poincaré and Plancherel–Pólya inequalities](https://arxiv.org/abs/1108.5637) | connectedness/bandlimited assumptions |
| 17 | [Anis–Gadde–Ortega, Graph Spectral Proxies](https://doi.org/10.1109/TSP.2016.2546233) | uniqueness and sampling criterion |
| 18 | [Kiefer–Wolfowitz, Equivalence of Two Extremum Problems](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/B8B0626C11F52B0FD8C67C5D54BDDD43/S0008414X00010002a.pdf/equivalence_of_two_extremum_problems.pdf) | D/G-optimality, short theorem |
| 19 | [Dereziński et al., Minimax Experimental Design](https://www.stat.berkeley.edu/~mmahoney/pubs/derezinski19b_colt19.pdf) | volume/leverage sampling 조건 |

### Conditional/probability 에이전트: 8편

| # | 문헌 / primary source | 검토 중심 |
|---|---|---|
| 20 | [Gneiting–Raftery, Strictly Proper Scoring Rules (2007)](https://doi.org/10.1198/016214506000001437) | binary proper score/regret |
| 21 | [Banerjee et al., Clustering with Bregman Divergences (2005)](https://jmlr.org/papers/v6/banerjee05b.html) | expectation representative, decomposition |
| 22 | [Banerjee–Guo–Wang, Optimality of Conditional Expectation (2005)](https://doi.org/10.1109/TIT.2005.850145) | conditional Bayes target |
| 23 | [Fong–Holmes–Walker, Martingale posterior distributions](https://arxiv.org/abs/2103.15671) | filtration/c.i.d. 조건 |
| 24 | [Probabilistically Masked LM (2020)](https://arxiv.org/abs/2004.11579) | mask averaging/random order |
| 25 | [Any-Order AR Models the Right Way (2022)](https://arxiv.org/abs/2205.13554) | mask lattice, path distribution |
| 26 | [Consistent Diffusion Language Models (2026)](https://arxiv.org/abs/2605.00161) | stochastic bridge/anchor, factorized marginals |
| 27 | [Abernethy–Frongillo, Scoring Rules for Linear Properties (2012)](https://proceedings.mlr.press/v23/abernethy12.html) | elicited property와 loss geometry |

### Control/model-reduction 에이전트: 8편

| # | 문헌 / primary source | 검토 중심 |
|---|---|---|
| 28 | [Moore, PCA in Linear Systems (1981)](https://algos.inesc-id.pt/projects/mor4less/Moore_81.pdf) | controllability/observability |
| 29 | [Glover, Optimal Hankel-Norm Approximations (1984)](https://doi.org/10.1080/00207178408933239) | classical approximation result; full-text access 범위는 세부 노트 참조 |
| 30 | [Lall–Marsden–Glavaski, Nonlinear Balanced Truncation (2002)](https://www.cds.caltech.edu/~marsden/bib/2002/06-LaMaGl2002/) | empirical input-output model reduction |
| 31 | [Condon–Ivanov, Empirical Balanced Truncation (2004)](https://doi.org/10.1007/s00332-004-0617-5) | nonlinear empirical construction |
| 32 | [Kawano–Scherpen, Empirical Differential Gramians](https://arxiv.org/abs/1902.09836) | trajectory-local differential system |
| 33 | [Becker–Rannacher, Goal-Oriented Error Estimation (2001)](https://www.cambridge.org/core/journals/acta-numerica/article/abs/an-optimal-control-approach-to-a-posteriori-error-estimation-in-finite-element-methods/5C67A03F528C6FA69F37A97DF5C3BE19) | adjoint-weighted residual; access 범위는 세부 노트 참조 |
| 34 | [Lohmiller–Slotine, Contraction Analysis (1998)](https://web.mit.edu/nsl/www/preprints/contraction.pdf) | local differential stability |
| 35 | [Tran–Rüffer–Kellett, Discrete-Time Convergence](https://arxiv.org/abs/1612.05327) | incremental/contraction 조건 |

초록/검색 결과만 확인한 Sobol, Efron–Stein, 일부 context/head pruning 자료와 식별이 끝나지 않은 OpenReview 항목은 이 35개에서 제외했다. 더 많이 검색했다는 것을 더 많이 정독했다는 뜻으로 세지 않았다.

## 9. 산식 검산과 기록

표준 라이브러리만 사용한 CPU 유한공간 검산:
- 3-bit Bernoulli(p=.3), residual degrees 0/1/2/3를 혼합.
- ρ=.25: 직접 기대값 C=1.2121875, spectral expression 동일.
- ρ=.75: C=.4890625, spectral expression 동일.
- 최대 부동소수점 오차 약 1.2e−15.
- four-state decomposition: A=.78, C=2.46, 직접식/분해식 일치.

이 숫자는 검산용 임의 함수의 값이다. **LLaDA 통계, pruning 결과, 또는 새 실험 성능이 아니다.**

Decision: 기존 A+C를 reference로 유지한다. 이번에 새로 구체화한 두 방향을 idea candidate로 기록한다. 가장 큰 가치가 있는 다음 질문은 “C의 어떤 문맥 변화 구조가 실제로 손상되며, 그 보존이 endpoint fidelity보다 추가적인 독립 성능을 설명하는가?”이다.

