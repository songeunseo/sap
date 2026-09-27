# Subliminal Clocks와 pruning allocation 연결
Date: 2026-09-17
Status: literature-grounded hypothesis, untested
Source: [Subliminal Clocks: Latent Time Modelling in Diffusion Language Models, v2](https://arxiv.org/html/2607.01774v2)
사용자가 제공한 요약과 원문 §4–5, Appendix C–E를 확인했다.

## Source findings
논문 모델은 LLaDA-1.5와 Dream이며 우리 LLaDA-8B-Base와 동일 checkpoint가 아니다. 입력을 고정한 진행도 평균벡터 차이 개입으로 confidence/entropy가 변한다. Norm-matched control은 empirical activation covariance에서 추출한 Gaussian 방향이다. Mean-vector PCA subspace 효과와 early-depth correction을 보고하지만 pruning 손상은 측정하지 않는다. Appendix C에서 Dream GSM8K는base68.8→target100에서23.1; 저자들의 qualitative inspection은 조기EOS를 지적한다. Confidence 증가=성능 증가가 아니다. 마지막층 linear τ readout이 약해질 수 있어 PCA 좌표를 정확한 scalar clock값으로 등치하지 않는다.

## Relation to current evidence
새 외부 근거는 “진행도 관련 특정 방향에 대한 개입이 출력에 영향을 준다”는 것이다. 우리 후반layer budget 교환 결과나D31 direction×receiver 손상이 이 방향 때문에 발생했다는 증거는 아니다. 기존 timestep-profile/reconstruction 보정의 미지지 결과를 소급해 뒤집지 않는다. 다만 scalar timestep weighting과 식별된 저차원 방향의 실제 pruning 손상은 측정 대상이 달라 구체적 새가설을 세울 수 있다.

## Proposed method hypothesis
같은pruning예산에서, 뒤계산을 통과하고도 출력직전 residual에 남는 progress-associated subspace 손상을 적게 만드는 allocation이 유리할 수 있다.

고정 입력x에서 unit g의 실제Wanda후보mask를적용한 residual차이:
delta_g(x)=h_L^{pruned_g}(x)−h_L^{dense}(x).
Dense calibration으로 별도 추정한 진행도관련orthonormal basis U_L, P_L=U_L U_Lᵀ.
단순proxy: d_g(s)=E_{x,j masked} ||U_Lᵀ delta_g^j(x;s)||².
이 값은 scalar τ 오차가 아니라 특정subspace 오차다. 다른층수복을거친최종차이를써 local손상을자동할인하는임의depth계수를추가하지않는다. 처음에는출력직전한지점을고정하고,우리Base에서subspace/steering효과재현이안되면논문의layer29를자동대입하지않는다.

Allocation: 같은Wanda순위/전체50%에서 g의48/52비용차이를추가제거수로나눈signed marginal로사용→기존bounded rank45–55→exactbudget. 기술적으로완성가능한후보지만기존전체reconstruction 대비추가가치미확인. Subspace추정은기능손상및downstream로선택하지않고분리된dense문서에서고정한다.

## A discriminating intervention before claiming this mechanism
입력mask수/내용을고정하고 pruning 후 h_P에서 progress subspace성분만동일입력dense h_D의성분으로교체:
h_rescue=h_P+P_L(h_D−h_P).
실측NELBO에서 L(h_P)−L(h_rescue)를계산. 부호그대로보고하고 confidence만높아진것을성공으로판정안함. Dense vector가필요한이개입은원인가설진단이며배포시방법이아니다.
같은rank·실제개입norm을맞춘 random/complement direction교정,일반activation상위PC교정과비교. Progress형방향만의recovery가있는지확인하되semantic purity나정확한mediation비율로주장안함.
전체최적화나모든interaction을규명해야만작은screen을할수있다는필수요건으로삼지않는다.

## Confound to control in estimating the basis
전체token평균은masked/unmasked의혼합비율변화만으로이동할수있다. 같은maskcount에서mask배치/문장을바꾼generalization,masked위치만의means및role-balancedmeans대조가필요하다. 이들은새방법에rolemax를부활시키는것이아니라basis추정의진단이다. Nominalmask확률대신실제응답window tau를사용하며prompt는제외한다. 기존8문장×10states는100bin기반의논문전체재현으로부르지않는다.

## Current A+C experiment interpretation
현재x−→x+는goldcontext추가와maskcount감소가동시에일어난다. 따라서 response오차의이득이있어도content utilization인지progress-relatedconfidence인지분리불가. 현재A+C를clock-preservingmethod로소급해이름바꾸지않는다. 입력mask고정된내부개입은이차이를다루는별도설계이다.
진행중인동결실험은유지. 00:24KST 확인:32/32probe완료,Uniform mini62/100진행,A/AC대기. 아직성능결과없음.

## Decision
문헌과현재통계의연결을명시한별도후보로저장. 새GPU수집/steering/probe/PCA/mini실험은시작하지않음. 기존context_response50 pipeline계속.

