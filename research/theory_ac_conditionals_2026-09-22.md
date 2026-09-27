# Theory review for the original scalar A+C

Date: 2026-09-22  
Scope: conditional-distribution compatibility, proper scoring/Bregman regret,
Doob martingales over reveal filtrations, and random-order masked prediction.  
No model, GPU, mask, or experiment was run for this review.  No Obsidian note
was written.

## Question and exact starting point

The object under review is the existing scalar objective in
`experiments/dlm_context_response50`, not the later full-vocabulary or DKD
extension.  For each natural nested pair, the code evaluates a fixed masked
query set at a `before` state and an `after` state obtained by randomly
revealing clean tokens.  The scalar readout is the gold-vs-rest log odds

`r_X^t = z_X^t(g) - logsumexp(z_X^t(w), w != g)`,

where `t=0,1` is the pair endpoint, `g` is the corpus token at the query, and
`X` is dense `D` or sparse `M`.  The error is `e_t=r_M^t-r_D^t`, and the
current code uses

`A = E[(e_0^2+e_1^2)/2]`,  `C = E[(e_1-e_0)^2]`,  `AC=A+C`.

The frozen bank has 8 WikiText spans and 80 pairs.  Each pair reveals a random
subset of currently masked positions (value-independent in `make_pairs`) and
keeps the remaining masked queries shared.  Current development evidence is
Uniform 54, A 55, A+C 61 on GSM8K mini-100; A+C versus A has exact paired
McNemar `p=.145996`.  NELBO was not measured.  These numbers are development
evidence, not a capability claim.

The narrow theoretical question is whether conditional-probability theory
supports a better *scalar* pair/loss construction.  The answer is negative as
an automatic design rule.  Proper-score theory applies to a true event or
target distribution, whereas this code regresses a sparse teacher-relative
logit.  A logit teacher target can already encode the same binary probability
exactly, and a reveal-position average on one fixed gold completion is not a
conditional-content expectation.  The literature therefore supplies validity
checks and counterexamples, not a new main scalar loss.

## Primary literature, findings, and applicability

### 1. Gneiting and Raftery, “Strictly Proper Scoring Rules, Prediction, and Estimation” (JASA 2007)

Primary: <https://doi.org/10.1198/016214506000001437> and the author PDF
<https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf>.

The paper’s Sections 2–3 characterize proper scores through a convex entropy
and its supporting hyperplanes.  Table 1 and the categorical examples give the
quadratic/Brier score and logarithmic score; the Brier divergence is squared
Euclidean distance, while the logarithmic score has KL divergence as its
regret.  In the categorical case, truthful distribution reporting uniquely
minimizes expected regret for a strictly proper score.  This would apply when
the event or target distribution is the population truth.  It does not make
`(sigmoid(r_M)-sigmoid(r_D))^2` automatically better than
`(r_M-r_D)^2`: the latter is teacher regression in a coordinate from which an
exact binary probability can be recovered.  Log odds remains a valid evidence
coordinate and may be useful for ranking; squared log-odds error is simply not
a proper score for the underlying probability.  The paper is forecasting
theory, not a pruning method, and our dense teacher is only a surrogate target.

### 2. Banerjee, Merugu, Dhillon, and Ghosh, “Clustering with Bregman Divergences” (JMLR 2005)

Primary: <https://jmlr.org/papers/v6/banerjee05b.html>; full paper
<https://jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf>.

Definition 1 and Examples 1–2 define Bregman divergence, squared Euclidean
distance, and KL.  Proposition 1 (pp. 1709–1710) proves that the expectation
is the unique representative minimizing expected Bregman loss.  Theorem 1,
Eq. (3), and pp. 1713–1714 decompose total Bregman information into between
and within partition information.  This supplies the conditional version we
need: if a random child posterior `P_1` has conditional mean `P_0` given the
parent sigma-algebra, then for any Bregman divergence

`E[D_phi(P_1,q_0)|F_0] = E[D_phi(P_1,P_0)|F_0] + D_phi(P_0,q_0)`.

The first term is irreducible reveal information; only the second depends on
the parent prediction.  Therefore child variation is not automatically an
error to suppress.  The paper is a clustering/quantization analysis, not
DLM or pruning; the conditional identity is a mathematical transfer, not an
empirical static-mask result.

### 3. Banerjee, Guo, and Wang, “On the Optimality of Conditional Expectation as a Bregman Predictor” (IEEE TIT 2005)

Primary record and manuscript: <https://ecommons.cornell.edu/entities/publication/4d8302b3-7259-4910-955b-c86d5564c556>; DOI
<https://doi.org/10.1109/TIT.2005.850145>.

The theorem states that `E[X|Z]` is optimal for every Bregman loss, with
least-squares prediction as a special case, and that Bregman losses are
essentially exhaustive under mild assumptions.  This is the cleanest reason
to regard a binary probability endpoint as the appropriate scalar forecast
coordinate: if the target is a conditional probability, its conditional
expectation is the population optimum under squared probability/Brier loss.
It also gives a no-go result for a purportedly universal new scalar geometry:
unless the new term encodes a different deployment target or finite-model
constraint, it cannot beat the true conditional expectation in population.
The paper does not study distributions over token sequences, masked
denoising, or frozen weights.  Its result applies to the binary projection
only and does not repair the missing wrong-token relation in gold-vs-rest.

### 4. Fong, Holmes, and Walker, “Martingale posterior distributions” (JRSS B 2023; arXiv 2103.15671)

Primary: <https://arxiv.org/html/2103.15671> and published article
<https://academic.oup.com/jrsssb/article/85/5/1357/7597700>.

Section 2.1, Theorem 1, Eq. (2.3), states the Doob posterior-mean martingale
property `E[theta_N | Y_{1:N-1}]=theta_{N-1}`.  Section 3.2, Eqs. (3.1)–(3.2),
defines conditionally identically distributed (c.i.d.) predictive sequences:
the conditional expectation of the next predictive distribution equals the
current predictive distribution.  They emphasize that predictive coherence
requires the update law and the sampling filtration, not merely a sequence
of observed states.  This maps to DLM reveals only when parent contexts and child *contents* are
sampled from a specified joint/conditional law.  Randomizing reveal positions
inside one already fixed gold sequence samples positions conditional on that
completion; it is not a draw from the conditional distribution of possible
child contents.  One child per unrelated state cannot estimate the tower
expectation.  A teacher chosen adaptively from token content, or a bank
reweighted by teacher confidence, changes the effective law and invalidates a
martingale claim.
The work concerns Bayesian/predictive uncertainty, not pruning; it provides a
pair-bank validity test, not a frozen-mask guarantee.

### 5. Liao, Jiang, and Liu, “Probabilistically Masked Language Model Capable of Autoregressive Generation in Arbitrary Word Order” (ACL 2020)

Primary: <https://arxiv.org/html/2004.11579>.

Section 2.3 Eq. (2) writes the usual masked objective and explicitly notes the
masked-token conditional-independence assumption.  Section 3.1 Eqs. (3)–(6)
averages over a probabilistic mask, while Section 3.2 Eq. (8) and Appendix A
Eqs. (9)–(13) prove that the uniform mask-ratio objective is equivalent to an
autoregressive model averaged over permutations.  The result supports a random, predeclared reveal order in a model trained over
the corresponding data/mask distribution; it does not support arbitrary
hand-picked pairs, or reveal-position randomization on one fixed completion,
as interchangeable samples from one joint distribution.  Their u-PMLM is trained from scratch and
evaluated for generation/NLU (Sections 4.2–4.3, Tables 2–7), not pruned.
The paper’s independence assumption is also a warning: per-query binary
posteriors can be coherent marginally while not defining a coherent joint
distribution over all masked queries.  Consequently, scalar A+C can be a
useful local functional surrogate but cannot be called a joint posterior
consistency proof.

### 6. Shih, Sadigh, and Ermon, “Training and Inference on Any-Order Autoregressive Models the Right Way” (NeurIPS 2022)

Primary: <https://arxiv.org/html/2205.13554>.

Section 2.1 Eqs. (3)–(4) represents arbitrary conditional inference by
univariate conditionals on a mask lattice.  Section 3.1 Definitions 1–2 and
Eqs. (5)–(6) show that a decomposition protocol induces an edge distribution
`D_{M,w}`; training should match the downstream mask distribution and path
frequency.  Algorithm 1 samples a mask and a data point, then traverses its
declared path.  Figure 3 and Tables 1–3 show ablations in which protocol and
mask-distribution changes improve likelihood, but these are model-training
results.  Section 5 explicitly warns that conditional estimates along paths
can be biased even when they remain normalized (lines 318–321 in the HTML).
For our bank, this supports recording reveal position, parent mask, query set,
and path frequency.  It does not justify treating branches from one fixed gold
completion as draws from a conditional content law, adaptive pair selection,
or an AO-ARM training improvement transferred to a frozen static weight mask.

### 7. Amin et al., “Consistent Diffusion Language Models” (ICML 2026)

Primary: <https://arxiv.org/html/2605.00161>.

Section 3.2 Definition 3.2 Eq. (3) defines a consistency operator as the
expected per-position prediction after an exact stochastic reverse bridge.
Definition 3.3 Eqs. (4)–(5) imposes path consistency in expectation under a
strictly proper divergence.  Proposition 3.4 proves the per-token posterior
marginal is a fixed point and says a max-step boundary anchor is needed for
uniqueness; the text explicitly notes degenerate unanchored fixed points and
that the optimum is factorized marginals, not the joint posterior.  Section
4.4 reports divergence ablations: forward KL is unbiased but unstable in their
training setup, backward KL is mode-seeking, and JSD trades stability and
diversity.  Appendix D.1 Eqs. (8)–(12) relates the anchored max-step limit to
MDLM/NELBO.  CDLM is trained, not pruned, and its exact reverse bridge is
unavailable to our frozen LLaDA calibration bank.  The safe transfer is only
the principle “expectation over a declared stochastic bridge plus an anchor”;
it is not a theorem that pairwise scalar response selection preserves DLM
quality.

### 8. Abernethy and Frongillo, “A Characterization of Scoring Rules for Linear Properties” (COLT 2012)

Primary: <https://proceedings.mlr.press/v23/abernethy12.html>.

The paper characterizes proper losses that elicit a linear property of a
distribution and proves that the resulting rules are Bregman divergences for
convex functions.  This sharpens the readout boundary: the binary probability
`P(Y=g|F)` is a valid scalar property, while the gold-vs-rest log odds is a
nonlinear reparameterization.  Squaring differences in that reparameterized
coordinate is not automatically a proper loss for the underlying probability.
The paper is about elicitation and prediction markets, with no DLM, masked
states, or pruning experiment.  It therefore supports choosing and labeling
the scalar property carefully, but cannot establish that probability-space
A+C will select a better static mask.

## What the theory says about the current log-odds pair

Let `X_Q` be the random token at a masked query and let `F_0` be the visible
parent context.  After a valid stochastic reveal, `F_1` is a larger sigma
algebra.  The full categorical posterior is

`P_t(v)=P(X_Q=v | F_t)`.

For every fixed vocabulary candidate `c` chosen before the reveal, a genuine
joint distribution and nested filtration give the coordinate-wise tower
identity

`E[P_1(c) | F_0] = P_0(c)`.                              (T1)

The full vector is the safest object for stating this identity.  The current
code instead sets `g` to the observed clean token for each sample and then
projects to `P_t(g)` or its teacher log odds.  Across samples, `g=X_Q` is a
future-measurable random index correlated with the hidden query value; it is
not a fixed coordinate `c`.  The tower identity for every fixed `c` therefore
cannot be silently applied to the sample-specific gold projection.  The
current gold-vs-rest readout must be labeled a teacher-relative,
label-conditioned scalar surrogate.  It is not a martingale coordinate.

Even for a fixed `c`, the tower property for probabilities does **not** imply
`E[logit(P_1(c))|F_0]=logit(P_0(c))`.  A concrete valid counterexample is
`P_0(c)=.6`, with `P_1(c)=.9` or `.3` at equal probability.  The child
probabilities average to `.6`, but the child logits average to
`(logit(.9)+logit(.3))/2 ≈ .675`, whereas `logit(.6)≈.405`.  Thus a log-odds
increment is an evidence coordinate, not a martingale innovation.  Jensen
curvature is enough to create drift even for a perfect posterior.

The current C is still meaningful as a *teacher-relative response* term:
`e_1-e_0` compares the sparse model's change in the evidence coordinate with
the dense model's change.  It should not be described as preserving a
martingale.  It can also over-emphasize high-confidence states because a fixed
probability error maps to a larger logit error near 0 or 1.  The implementation
randomizes reveal positions inside each fixed calibration state, but those
states already share a selected gold completion and the bank has essentially
one child per parent.  Neither fact supplies the missing expectation over
possible child contents in (T1).

For any Bregman divergence `D_phi`, the conditional identity is

`E[D_phi(P_1,q_0)|F_0]`
` = E[D_phi(P_1,P_0)|F_0] + D_phi(P_0,q_0)`.                 (T2)

This identity is valid only when `P_1` is a random child posterior under the
specified conditional law and `P_0` is its conditional mean.  Averaging reveal
positions on one fixed gold sequence does not establish those premises.  When
they do hold, the first term is the value/variance of the reveal and is
independent of the parent forecast.  Exact endpoint matching already has the
Bayes target; a probability-space transform is not automatically better than
exact logit teacher matching.  A response term can change finite-sample or
finite-capacity mask selection, but it has no general population-improvement
theorem.  This is the main negative result of the literature review.

## Optional scalar challenger, not a martingale construction

The previous `A_prob+C_prob` idea is downgraded.  It is not a justified main
candidate, and the name “martingale-centered” is incorrect for the current
bank.  It can be retained as a low-priority teacher-regression ablation only
after the sampling law is made explicit.  It keeps the binary gold-vs-rest
property, frozen Wanda support, exact-budget static allocator, and no rollout,
vector/DKD term, or new optimizer.

### What a valid conditional-compatibility test would require

A tower test must start with a fixed candidate token `c` chosen before the
query completion is sampled, or with the full categorical posterior vector.
It must then sample parent contexts and child contents from a specified joint
or conditional law.  Suitable sources could be a known probabilistic bridge,
a separately sampled data continuation under a declared model, or a synthetic
joint distribution used only for a diagnostic.  The current clean calibration
bank supplies none of these: each span has one fixed gold completion and
randomizing which masked position is revealed samples positions conditional on
that completion.  Averaging those children estimates reveal-position/content
sensitivity for that sequence, not `E[P_1|F_0]`.

If one nevertheless groups several reveal positions from the same fixed gold
state, report only the descriptive quantities

`V_D = E_parent Var_k[p_D^{1,k}]`,

and the analogous sparse/teacher response decomposition.  Do not call
`mean_k p_D^{1,k}-p_D^0` teacher martingale defect or interpret it as
incoherence.  A large value can reflect the fixed sequence, position choice,
mask geometry, or teacher behavior; it has no tower interpretation without a
conditional-content sampling law.

### Optional teacher-relative probability ablation

For the existing label-conditioned scalar readout, define

`p_X^t=sigmoid(r_X^t)`,  `a_t=p_M^t-p_D^t`.

On a grouped fixed-sequence bank, define the algebraic branch summaries

`bar_a_1 = K^{-1} sum_k a_{1,k}`,

`C_mean = (bar_a_1-a_0)^2`,

`C_var = K^{-1} sum_k (a_{1,k}-bar_a_1)^2`,

`C_pos = E_parent[C_mean+C_var]`,

and

`A_prob = E_parent[(a_0^2+K^{-1}sum_k a_{1,k}^2)/2]`.

With equal child weighting, `C_pos` is exactly the grouped expansion of the
one-child probability response error

`E[((p_M^1-p_M^0)-(p_D^1-p_D^0))^2]`.

Here `C_mean` and `C_var` are only a predictable-looking-versus-branch
sensitivity decomposition; they are not conditional drift and innovation
unless the missing content-sampling premises have been established.  The
optional fixed-coefficient ablation is `A_prob+C_pos`, compared against the
existing `A_logit+C_logit` on the same bank.  It has no proper-scoring
superiority claim: both are regressions to an approximate dense teacher, and
exact logit matching can recover the same binary probability.  If corpus labels
are scored directly, binary Brier/log score may be reported as a separate
true-event calibration diagnostic, not substituted silently for the teacher
objective.

The strongest prerequisite for any martingale-language result is therefore a
full-vector or fixed-`c` compatibility diagnostic first.  The sample-specific
gold projection can be reported separately as the project’s original scalar
surrogate, but it cannot validate the probability-space theory.

## Static-mask transfer and minimum comparison

Any optional scalar ablation still uses one static mask, original weights,
exact 50% count, frozen within-row Wanda ordering, and the same hard-mask
exchange/allocator as all other arms. It adds no runtime reveal policy or
adaptive inference. A teacher-relative loss on calibration states does not
prove preservation of an unseen denoising trajectory, and a lower scalar
surrogate does not imply lower NELBO or higher GSM8K.

The minimum defensible comparison is:

| arm | scalar objective | bank |
|---|---|---|
| Uniform | none | current natural parent/child states |
| A reference | `A_logit` | current natural bank |
| existing AC | `A_logit+C_logit` | current natural bank |
| optional scalar ablation | `A_prob+C_pos` | same fixed-sequence bank, labeled teacher regression |
| compatibility diagnostic | full-vector or fixed-`c` coordinates, `V_D`, label Brier/log score | only if a valid sampling law is available |

Do not call the optional fixed-sequence branch average a tower test. Keep the
candidate pool, total calibration forwards, exact budget, and support family
fixed. Freeze masks before independent evaluation. The primary decision is a
predeclared held-out downstream metric at equal sparsity and measured compute;
NELBO is complementary and may disagree with capability. The optional scalar
ablation earns no claim if it only lowers its teacher-regression surrogate, if
all arms improve equally under a solver change, or if its result depends on a
fixed-sequence reveal-position average with no document-held-out effect.

## Novelty boundary and decision

Proper scoring, Bregman conditional expectation, Doob/c.i.d. coherence,
random-order masked conditionals, and stochastic diffusion consistency are
established. `A_prob+C_pos` is a coordinate/teacher-regression ablation, not a
new scoring-rule theorem, martingale result, or arbitrary-order model. The
current natural bank cannot support a population conditional-compatibility
claim because it lacks fixed-candidate/full-vector sampling over possible
child contents. No new main pruning candidate follows from this literature
review.

Decision: retain existing scalar `A_logit+C_logit` as the project reference;
do not promote the prior “martingale-centered probability response” idea. If a
future low-cost theory audit is authorized, first use full-vector or fixed-`c`
coordinates with an actually specified conditional-content law, and report the
sample-specific gold projection separately. Same-count swaps remain a distinct
confound diagnostic. Do not call logit C a martingale term, do not interpret
fixed-gold reveal-position means as teacher incoherence, and do not infer a
joint conditional distribution from scalar queries.

## Sources

1. Gneiting & Raftery (2007), <https://doi.org/10.1198/016214506000001437>.
2. Banerjee et al. (2005), <https://jmlr.org/papers/v6/banerjee05b.html>.
3. Banerjee, Guo & Wang (2005), <https://doi.org/10.1109/TIT.2005.850145>.
4. Fong, Holmes & Walker (2023), <https://arxiv.org/html/2103.15671>.
5. Liao, Jiang & Liu (2020), <https://arxiv.org/html/2004.11579>.
6. Shih, Sadigh & Ermon (2022), <https://arxiv.org/html/2205.13554>.
7. Amin et al. (2026), <https://arxiv.org/html/2605.00161>.
8. Abernethy & Frongillo (2012), <https://proceedings.mlr.press/v23/abernethy12.html>.
