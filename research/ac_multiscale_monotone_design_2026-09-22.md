# Multiscale A+C: monotone reveal 이론과 실행 가능한 설계 초안

Date: 2026-09-22
Status: proposed, unimplemented, no model experiment
Scope: 기존 scalar A+C의 pair distribution/objective 설계. GPU, model forward, pruning mask 생성, benchmark 실행 없음.
Prerequisite: Obsidian get_sync_status와 Research-State read_note 성공. 최근 연구 결과와 기존 core.py/config를 확인.
Predecessor: [이전 이론 조사](/home/tmluser1/sap/research/theory_ac_upgrade_synthesis_2026-09-22.md)

## Decision

이전 stationary correlated-mask 후보를 **단조로운 gold-reveal 상태들의 다중 규모 반응 보존**으로 발전시킨다. 현재 main draft는 다음 구성이다.

1. 동일 clean sequence와 고정 masked query set에서 nested reveal chain을 만든다.
2. 공개 확률의 log-odds를 기준으로 phase 간격을 정한다.
3. 동일 상태들을 작은/중간/큰 phase 간격으로 연결한다. 각 규모에서 모든 상태가 정확히 한 번 참여하는 matching을 쓴다.
4. endpoint A와 평균 multiscale C를 계산한다. λ=1을 시작안으로 고정하고, 기존 scalar/weight ranking/예산 backend를 첫 대조에서 유지한다.
5. 최종 sparse mask의 독립 성능을 평가하기 전에는 더 좋은 pruning 방법으로 주장하지 않는다.

새로운 closed-form sparsity proxy나 spectrum fit을 제안하는 것이 아니다. **dense–sparse 출력 오차가 문맥 공개 과정에서 어떻게 변하는지**를 직접 평가하는 목적함수다. 이론은 pair 설계와 측정 대상을 설명한다.

## 1. 전제와 현재 구현

D는 dense, M은 고정 sparse mask가 적용된 모델이다. clean sequence z와 평가할 query q를 고정한다.

\[
f_X(v)=\log\frac{p_X(z_q\mid x(v))}{1-p_X(z_q\mid x(v))},
\qquad e_M(v)=f_M(v)-f_D(v).
\]

v는 **eligible context token의 visible/masked bit**다. query는 항상 masked이고, 고정 prompt/background가 있으면 모든 상태에서 동일하게 유지한다. 따라서 p는 eligible context의 공개 확률이며, query와 prompt를 포함한 전체 입력의 실제 공개 비율과 구분한다.

기존 code는 한 before state에서 고정 개수의 token을 랜덤 reveal하고, 남은 모든 masked 위치를 공통 query로 평가한다. n_reveal=min(round(.05*256), max(1, masked_count//4)) 계열의 설정이다. 이 구현은 아래 independent Bernoulli process와 동일하지 않다.

관측된 native mini100: Uniform54 / A55 / AC61. 새 설계의 성능 근거는 아니다. 다른 cached Uniform62와 섞지 않는다. 새 설계가 기존 61을 넘는다는 예측도 현재 근거가 없다.

## 2. 기존 stationary 제안의 적용 범위

고정 p의 product mask에서 각 bit를 확률 ρ로 유지하고 나머지를 Bernoulli(p)로 재샘플하면,

\[
C_{p,\rho}
=\mathbb E[(e(v')-e(v))^2]
=2\sum_{S\ne\emptyset}(1-\rho^{|S|})\hat e_p(S)^2.
\]

근거: [O'Donnell](https://arxiv.org/abs/2105.10386), Chapter 8, Proposition 8.28. 이는 정당한 calibration coupling이지만 단일 monotone decoding trajectory가 아니다.

기존 답변의 “여러 ρ를 섞자”에는 세 미완성 사항이 있었다.

- scale mixing이 어느 오류를 더 강조하는지 결정되지 않았다.
- stationary re-mask coupling의 생성 관련성이 불명확했다.
- 각 scale에서 새로운 state를 더 계산하면 loss 효과와 calibration coverage 효과가 섞인다.

아래 설계는 monotone coupling, 균형 matching, 동일 bank 대조로 이 세 문제를 구체화한다. 이론은 설계 가능성을 보이며, 최적 scale이나 GSM8K 향상을 보장하지 않는다.

## 3. Monotone reveal에서도 정확한 분석이 가능하다

### 3.1 생성 규칙

각 eligible context 위치 i에 독립적인 U_i~Uniform(0,1)를 한 번 뽑고,

\[
V_i(p)=1\{U_i\le p\},\quad 0<p<1
\]

로 정의한다. p<r이면 V(p)≤V(r)이므로 gold token을 공개하기만 한다. 독립 좌표의 joint distribution은

\[
\Pr(0,0)=1-r,\quad
\Pr(0,1)=r-p,\quad
\Pr(1,1)=p,\quad
\Pr(1,0)=0.
\]

이것은 **내용과 무관한 무작위 reveal order**다. confidence-based LLaDA decoding이나 model-generated token을 사용하는 실제 rollout과 같지는 않다.

### 3.2 일반화된 noise identity의 근거

추가로 확인한 [Heidari–Pradhan–Venkataramanan, Boolean Functions with Biased Inputs (2019)](https://arxiv.org/abs/1901.10576)의 §III Lemma 1, Eq.13–15는 서로 다른 입력 bias p,r의 joint product distribution에서도 cross-correlation을 biased Fourier coefficient로 표현한다. 그 증명은 orthogonality와 coordinate independence를 사용한다. 같은 곱 전개를 real-valued residual에 적용한다.

정규화된 basis를

\[
\phi_{p,S}(v)=\prod_{i\in S}
\frac{v_i-p}{\sqrt{p(1-p)}}
\]

로 두고, a_{p,S}=E[e(V(p))φ_{p,S}(V(p))]라 하자. 한 coordinate의 correlation은

\[
\rho(p,r)=\sqrt{\frac{p(1-r)}{r(1-p)}}.
\]

좌표 독립성에 따라 서로 다른 subset S,T의 cross term은 0이고, 같은 subset에는 ρ^{|S|}가 남는다. 따라서

\[
\boxed{
C_{p,r}
=\mathbb E[(e(V(r))-e(V(p)))^2]
=\sum_S
\left[
a_{p,S}^2+a_{r,S}^2
-2\rho(p,r)^{|S|}a_{p,S}a_{r,S}
\right].
}
\]

이것은 Transformer 선형화나 작은 pruning 가정 없이 성립하는 finite-function identity다. 단, **정의한 independent product coupling**이라는 조건은 필요하다.

각 S의 항은 다음처럼 항상 음수가 아닌 두 부분으로 나뉜다.

\[
\frac{1-\rho^{|S|}}2(a_{p,S}+a_{r,S})^2
+
\frac{1+\rho^{|S|}}2(a_{p,S}-a_{r,S})^2.
\]

두 부분은 서로 다른 관점이다.

- 여러 context 위치에 의존하는 residual 성분의 크기.
- phase가 바뀔 때 그 성분의 biased-basis coefficient가 달라지는 정도.

p와 r에서 basis 자체도 달라지므로 두 번째를 곧바로 semantic circuit의 변화로 해석하지 않는다. 이것은 phase에 따른 residual 함수의 좌표 표현 변화다.

기존 stationary 식처럼 하나의 E_k만 남는 것이 아니다. **서로 다른 phase에 동일 spectrum이 있다고 가정하지 않는다.**

### 3.3 공개 비율의 log-odds가 자연스러운 scale 좌표

\[
s(p)=\log\frac{p}{1-p},\qquad
h=s(r)-s(p)>0
\]

라 하면

\[
\boxed{\rho(p,r)=e^{-h/2}}.
\]

같은 h의 reveal interval은 시작 phase가 달라도 같은 coordinate correlation을 갖는다. 따라서 “13개 token reveal”을 모든 phase에서 동일한 변화 크기로 취급하기보다, **공개 확률의 log-odds 거리**로 scale을 정의할 근거가 생긴다.

이 s는 모델의 confidence log-odds f와 다른 값이다. 하나는 mask sampling의 공개 확률, 다른 하나는 정답 token 확률이다.

h가 같아도 공개되는 token 수는 같지 않다. 기대 reveal 수는 n(r-p)이고, 실제 값은 랜덤이다. p=0 또는 1에서는 basis가 퇴화하므로 내부 구간만 사용한다.

## 4. 같은 state bank를 여러 규모로 연결한다

### 4.1 Nested chain

K=2^m개 phase를 잡는다.

\[
s_j=s_{\min}+j\Delta s,\qquad
p_j=\operatorname{sigmoid}(s_j),\quad j=0,\ldots,K-1.
\]

같은 U를 공유하는 x_0,…,x_{K-1}을 만든다. 모든 상태에서 같은 query set Q를 평가한다.

읽기 쉬운 K=8 예시의 scale set은 D={1,2,4}다.

| scale | 연결되는 phase index |
|---|---|
| 1 | (0,1), (2,3), (4,5), (6,7) |
| 2 | (0,2), (1,3), (4,6), (5,7) |
| 4 | (0,4), (1,5), (2,6), (3,7) |

각 pair는 더 공개된 쪽으로 향하는 자연스러운 gold-reveal 비교다. d=2^b에서 bit b가 0인 j를 j+d와 연결한다.

이 matching을 선택한 이유:

- 각 규모에서 모든 node가 정확히 한 번 참여한다.
- 모든 scale은 K/2개 pair를 갖는다.
- 같은 d 안에서는 h=dΔs이므로 correlation ρ=e^{-dΔs/2}가 같다.
- 추가 scale의 C는 **이미 계산한 출력의 차이**로 만들 수 있다.
- phase 자체가 A 또는 C에 참여하는 총 가중치가 scale별로 달라지지 않는다.

이진 matching이 모든 가능한 reveal 연결 중 최적이라는 주장은 하지 않는다. 해석과 통제가 쉬운 완결된 첫 설계다.

### 4.2 새로운 objective

각 sequence/reveal-chain/query에서 e_j=f_M(x_j)-f_D(x_j)라 하자.

\[
A_{\rm chain}=\frac1K\sum_j e_j^2,
\]

\[
C_d=\frac{2}{K}\sum_{j:\,j\&d=0}(e_{j+d}-e_j)^2,
\]

\[
\boxed{
L_{\rm multi}(M)
=A_{\rm chain}(M)
+\lambda\sum_{d\in D}\alpha_d C_d(M),\quad
\alpha_d\ge0,\quad\sum_d\alpha_d=1.
}
\]

첫 설계는 α_d=1/m, λ=1이다. 원래 A+C의 상대 coefficient를 유지하고, C 예산을 여러 scale에 나누는 선택이다. 이것이 최적 coefficient라는 이론은 없다.

query 내 평균 → chain 평균 → sequence 균등 평균 순서로 집계한다. query 수가 많거나 scale edge가 많다는 이유로 특정 sequence를 더 가중하지 않는다.

**최소 대조군:** 같은 x_j와 f_D cache를 사용하는 A-only, A+C_1, A+mean_d C_d. historical AC는 별도 reference이고 이 새 bank의 A+C_1과 같은 실험이 아니다.

### 4.3 왜 C의 coefficient만 키운 것과 다른가

K=8에서 두 residual 패턴을 보자. 값은 설명용이며 LLaDA 관측값이 아니다.

| 패턴 | e_0,…,e_7 | A | C_1 | C_2 | C_4 |
|---|---|---:|---:|---:|---:|
| 일정한 편차 | 1,1,1,1,1,1,1,1 | 1 | 0 | 0 | 0 |
| phase에 따라 방향이 달라지는 편차 | 1,1,−1,−1,1,1,−1,−1 | 1 | 0 | 4 | 0 |

모든 λ에 대해 A+λC_1은 두 패턴을 똑같이 본다. A+(C_1+C_2+C_4)/3은 각각 1과 7/3이다.

따라서 **추가 scale은 기존 짧은 pairing의 scalar coefficient 변경으로 표현되지 않는 오류 선호도를 만든다.** A가 두 패턴의 개별 endpoint 오차를 놓친다는 뜻은 아니다. 두 패턴의 총 A가 같을 때 어떤 형태를 더 부담스럽게 볼지 달라지는 것이다.

이 예시는 현재 disjoint matching과의 차이를 보인다. 모든 인접 edge를 사용하는 connected-path C까지 구별 불가능하다는 주장은 아니다. 그래서 A+all-adjacent-C도 강한 추가 control로 포함한다.

## 5. 무엇을 보장할 수 있고, 무엇은 못 하는가

### 5.1 Objective scale과 endpoint exposure

각 matching이 node를 한 번씩 사용하므로 모든 C_d의 quadratic matrix diagonal은 2/K이고 trace는 2다. α 합을 1로 두면 multiscale에서도 같아진다.

또한

\[
0\le C_{\rm multi}\le4A_{\rm chain},
\qquad
A_{\rm chain}\le L_{\rm multi}\le(1+4\lambda)A_{\rm chain}.
\]

scale 수를 늘렸다는 이유만으로 C의 총 계수가 m배 증가하지 않는다. 다만 특정 error pattern에 부여하는 가중치는 의도적으로 바뀐다.

### 5.2 Chain 상의 residual drift 제어

균등 scale matching graph는 phase-index m-dimensional hypercube다. 여기서는 **phase index의 bit**를 쓰는 것이며, §3의 **input-mask bit**와 다른 Fourier 공간이다.

phase-index Walsh mode S의 C_multi multiplier는 4|S|/m이다. 따라서

\[
\frac4m\operatorname{Var}_j(e_j)
\le C_{\rm multi}\le4\operatorname{Var}_j(e_j).
\]

하나의 matching만으로는 여러 disconnected component마다 서로 다른 상수 오차가 남을 수 있다. 모든 dyadic scale을 쓰면 connected graph가 되어 chain 전체의 비상수 residual을 측정한다.

이는 A가 제공하지 못하던 완전한 정보를 새로 만든다는 보장이 아니다. 유한 sparse mask 선택에서 **phase에 따라 달라지는 오류를 추가로 강조하는 norm**이라는 설명이다.

### 5.3 Dense가 보인 반응 방향의 뒤집힘

pair distribution은 scale를 α로 고르고 해당 matching edge를 균등 선택하는 것으로 정의한다. Δ_D=f_D(x_j)-f_D(x_i), Δ_M=f_M(x_j)-f_M(x_i)라 하자.

\[
\Pr\left[
\Delta_M\Delta_D\le0,\ |\Delta_D|\ge\gamma
\right]
\le
\frac{C_{\rm multi}}{\gamma^2},\quad\gamma>0.
\]

방향이 반대로 바뀌고 teacher response 크기가 γ 이상이면 |Δ_M−Δ_D|≥γ이므로 얻는 직접적인 bound다.

보존 대상은 **dense의 정답 log-odds 반응 방향**이다. dense가 정답인지, final answer가 맞는지, reveal token 선택이 유지되는지까지 보장하지 않는다. 이것을 GSM8K accuracy bound로 부르지 않는다.

### 5.4 Perfect matching limit와 metric bounds의 의미

모든 endpoint에서 e=0이면 모든 C도 0이다. 반응 loss는 perfect endpoint matching보다 더 많은 정답 정보를 갖는 것이 아니다. 제한된 sparsity/finite calibration에서 error allocation을 바꾸는 역할이다.

또 logit error e에 대해 sigmoid의 1/4-Lipschitz 성질로 endpoint binary-probability MSE≤A/16이라는 보조 bound는 가능하지만, gold-vs-rest scalar가 wrong-token ordering까지 보존하는 것은 아니다.

## 6. 왜 전체 pair를 다 쓰지 않는가

모든 K(K−1)/2 pair를 똑같이 쓰면

\[
C_{\rm all}
=\frac{2K}{K-1}\operatorname{Var}_j(e_j).
\]

어떤 phase가 이웃인지, 어떤 interval이 짧은지 정보가 사라지고 단순한 global residual variance가 된다. 더 많은 pair가 더 의미 있는 objective라는 결론은 아니다.

선택한 dyadic matching은 같은 총 node/edge 가중치를 유지하면서 interval 규모를 구별한다. 그러나 “분산 penalty보다 더 좋아야 한다”는 이론은 없으므로 A+C_all을 저비용 control로 둔다. 둘 다 동일한 cached outputs에서 계산할 수 있다.

## 7. 무엇을 정규화하지 않기로 했는가

이전 stationary 초안에는 C/[2(1−ρ)]가 있었다. 이는 small-step limit에서 고차 input dependence를 강하게 강조한다.

현재 monotone main draft에는 이 정규화를 넣지 않는다.

- cross-phase coefficient도 변하므로 stationary 차수 해석을 그대로 적용할 수 없다.
- ρ→1에서 rare nonzero transitions를 1/(1−ρ)로 나누면 sampling variance 문제가 생긴다.
- 높은 차수의 오류에 더 큰 penalty를 주어야 한다는 empirical evidence가 아직 없다.

이번에는 C의 raw squared log-odds response 단위를 유지하고, 각 scale의 edge 개수/endpoint exposure만 맞춘다. response magnitude가 큰 scale이 결과를 주도할 수 있으므로 C_d의 값, marginal allocation rank, interval별 teacher response를 따로 보고한다. 이것이 문제로 관측되면 normalization을 별도 설계 축으로 검토한다.

## 8. Allocation으로 연결하는 완결된 첫 알고리즘

Formal target:

\[
\min_{M\in\mathcal M_B}L_{\rm multi}(M),
\]

여기서 \(\mathcal M_B\)는 기존 fixed Wanda ranks/surviving weights와 정확한 global budget B를 따르는 physical masks다.

첫 comparison에서는 objective의 기여를 분리하기 위해 다음 기존 backend를 유지할 수 있다.

1. Uniform50 mask와 within-row Wanda order를 freeze.
2. 새 nested chain bank와 Q를 미리 생성해 freeze.
3. Dense의 node별 scalar f_D를 한 번 cache.
4. block l의 48/52% probe를 동일 jointly sparse Uniform50 background에서 physical mask로 평가.
5. 같은 probe 출력에서 A, 각 C_d, L을 모두 계산.
6. signed marginal cost
   \[
   g_l=\frac{L(M_{l,52})-L(M_{l,48})}
   {N_{\rm pruned}(M_{l,52})-N_{\rm pruned}(M_{l,48})}
   \]
   를 기존 rank mapping/45–55% allocation/exact-count rounding에 넣는다.
7. 완성된 전체 sparse mask를 같은 목적함수와 독립 평가에서 다시 측정한다.

이 rank mapping이 위 constrained optimization을 정확히 푼다는 주장은 하지 않는다. 48/52의 marginal이 45/55나 다른 layer 변경과 가산적으로 합쳐진다는 가정도 보장되지 않는다. 기존 probe-extrapolation audit의 한계는 유지한다.

향후 정확한 hard-mask exchange backend를 사용해도 objective 정의는 동일하다. 그러나 새 C와 새 solver를 한 번에 바꿔 gain의 원인을 섞지 않는다.

복잡도: N_chains·K번의 state forward가 한 complete candidate의 주요 비용이다. 모든 C_d를 추가 계산하는 비용은 O(N_chains·K·log K·|Q|)의 scalar 연산이며 transformer backward가 필요 없다. 기존 bank에 새 chain을 추가하면 forward 비용은 증가한다. “같은 새 bank를 여러 방식으로 연결”하는 비교에서만 forward 수가 동일하다.

## 9. 아직 실행하지 않은 V1 설계 초안

| 항목 | 설계 결정 |
|---|---|
| readout | 기존 scalar gold-vs-rest log-odds |
| context | clean calibration sequence의 gold reveal |
| queries | chain 생성 전에 선택한 고정 masked Q |
| phase | logit(p) 균등 grid; p는 eligible context의 공개 확률 |
| scale | K=2^m에서 d=1,2,4,…,K/2 |
| pair | dyadic matching; lower→higher visibility |
| objective | A+mean_d C_d, λ=1 |
| backend | 첫 대조에서는 기존 backend 공통 유지 |
| controls | A-only, A+C_1, A+all-adjacent-C, A+C_all; historical AC 별도 |
| evaluation | development/held-out response와 독립 task quality 구분 |

K=8은 설명과 최소 구성의 예시다. p_min/p_max, Q 크기, chain 수, corpus sample 수와 총 forward 예산은 실행 전에 고정해야 한다. 이번 작업은 이론 설계 요청이며, 실행 설정 확정이나 새 실험 시작이 아니다.

중요한 sampling 제약:

- “한 pair에서 최소 1 token reveal”을 강제하거나 zero-change pair를 버리면 위 joint distribution이 바뀐다. zero-change는 정당한 0 response로 포함한다.
- 정확한 mask 개수를 맞추면 product distribution이 아니므로 §3의 식을 그대로 주장할 수 없다.
- confidence-dependent reveal이나 generated-token rollout도 같은 identity의 적용 대상이 아니다.
- Q를 chain 결과를 본 뒤 임의로 고르면 sampling conditioning이 생긴다. upfront selection이 가장 명확하다.
- Q가 크면 late-phase 실제 mask 비율을 크게 바꾼다. eligible visibility p와 전체 mask 개수를 둘 다 기록한다.
- 많은 edge를 동일 node/chain에서 만들었다고 독립 표본 수가 증가한 것은 아니다. uncertainty는 clean sequence/chain 단위를 고려한다.
- 현재 query 대상 gold label은 고정 clean sequence의 readout index다. posterior martingale 정리를 주장하지 않는다.

## 10. 가설 / 관측 / 실패 조건

**Hypothesis H1.** 짧은 context change의 response error만으로 결정한 mask가 더 긴 reveal interval에서 systematic drift를 보일 수 있다.

**Hypothesis H2.** endpoint fidelity/총 C weight를 통제한 상태에서 여러 interval의 response fidelity를 반영하면 더 나은 allocation을 선택할 수 있다.

**현재 Result.** finite toy algebra에서 identity와 graph normalization만 확인했다. 실제 DLM에서 H1/H2를 측정하지 않았다.

**Interpretation.** 성공한다면 “정적 mask가 여러 진행도 사이의 반응을 함께 보존해야 한다”는 설계 근거가 된다. 실패하면 scale 설계, scalar readout, teacher-state relevance, allocation approximation을 구분한다.

다음 결과는 설계 가치를 낮춘다.

- all-adjacent 또는 all-pair variance control이 같은 비용에서 동일하거나 더 좋음.
- 단일-scale λ 조절만으로 모든 independent gain이 재현됨.
- C_multi는 개선되지만 held-out response/독립 task quality가 개선되지 않음.
- 새 bank의 A-only가 모든 gain을 설명함.
- gold-reveal bank에서는 좋아지지만 생성 context에서 이득이 사라짐.

DLM 특화는 mask가 점차 공개되는 conditional family와 static-mask allocation 문제의 결합에 있다. 일반 masked model에도 적용 가능하다. DLM-only 수학이나 모든 기존 방법의 Uniform 열세를 요구하지 않는다.

## 11. 동일 phase의 기존 후보를 유지하는 대안

stationary ρ-correlated pair도 공통 parent에서 두 번 독립적으로 reveal하는 sibling pair로 생성할 수 있다.

목표 child visibility p와 correlation ρ에 대해

\[
a=\frac{\rho p}{1-p+\rho p},\qquad b=p(1-\rho).
\]

parent B_i~Bernoulli(a)를 만들고, B_i=0인 위치를 각 branch에서 독립적으로 확률 b로 공개한다. 두 child의 marginal은 p이고 correlation은 ρ다. 각 branch는 parent로부터 monotone reveal이다.

\[
C_{p,\rho}=2\,\mathbb E_B\operatorname{Var}(e(V)\mid B).
\]

이것은 **고정 gold 문장에서 reveal 위치를 바꾸는 mask-randomness variance**다. token 내용의 posterior expectation이나 tower consistency가 아니다.

두 child를 서로 비교하는 것은 하나의 실제 monotone trajectory transition이 아니므로, 이 sibling 방식은 phase를 고정한 보조 control로 남긴다. main objective에 추가 항으로 합치지 않는다.

## 12. 출처, 신규성, 검산

직접 근거:
- [O'Donnell, Analysis of Boolean Functions](https://arxiv.org/abs/2105.10386): product-space basis/noise operator, Chapter 8.
- [Heidari et al., Boolean Functions with Biased Inputs](https://arxiv.org/abs/1901.10576): 서로 다른 bias의 correlated input, Lemma 1/Eq.13–15.
- [Noise Stability of Transformer Models](https://arxiv.org/html/2602.08287): closest Transformer stability-analysis/training prior, §6.

기존 noise identity, cross-bias identity, graph quadratic form을 새로운 수학 정리로 주장하지 않는다. 이 문서의 기여는 그것들을 **gold-reveal calibration과 동일 예산 allocation의 검증 가능한 설계**로 연결하는 제안이다.

검산 파일:
- [Python](/home/tmluser1/sap/research/ac_multiscale_design_checks_2026-09-22.py)
- [JSON](/home/tmluser1/sap/research/ac_multiscale_design_checks_2026-09-22.json)

확인:
- 4-bit arbitrary nonlinear residual, 4개 p<r 조건에서 직접 finite expectation과 cross-bias/PSD formula 일치.
- 최대 차이 4.44e−16.
- K=8 temporal-index Walsh 8개 mode의 C eigenvalue 일치.
- node별 matching 참여 수, C≤4A, spectral-gap bound, all-pair variance identity 확인.
- 같은 A/C_short이지만 C_multi가 다른 toy pattern 확인.
- sibling construction joint distribution 차이 1.12e−16 이하.

이 값들은 모델 실험 결과가 아니다.

