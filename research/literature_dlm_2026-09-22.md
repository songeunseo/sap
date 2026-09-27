# DLM pruning literature reassessment for the A+C design

Date: 2026-09-22  
Scope: primary-source literature review and design reasoning only. No model
forward, mask generation, GPU job, NELBO run, or GSM8K evaluation was run.

## Research question and current evidence boundary

The current native 50% mini comparison is Uniform 54, A-only 55, and A+C 61
on GSM8K mini-100. A+C versus A has 9 rescued and 3 regressed examples but an
exact paired McNemar p-value of 0.145996. A+C has no independent NELBO result.
The current A+C bank contains natural nested gold-reveal pairs: the query
positions are shared, but the revealed endpoint has both more visible evidence
and fewer masked positions. The scalar readout is gold-vs-rest log-odds
endpoint error plus its paired change. The current CPU audit found A--A+C cost
Spearman 0.942815, C--A Spearman 0.153592, and 27/32 ideal layer-rate changes;
these are in-calibration allocation diagnostics, not held-out evidence.

The older Reveal-KL branch already used a dense denoising trajectory bank and
did not show a strong independent gain (reported mini results: Reveal 19,
masked 16, Uniform 12, paired p=0.508). Therefore “use a trajectory bank” or
“weight by denoising time” is not a new recommendation. The question is whether
the bank should contain *generated and deliberately corrupted visible contexts*
that test prediction revision, rather than only clean/gold context reveals.

## Primary literature findings

### A. Context brittleness and revision are measurable in generated states

**CoRe: Context-Robust Remasking for Diffusion Language Models**, Zhai et al.,
arXiv:2602.04096v3, 2026. Primary source:
<https://arxiv.org/html/2602.04096>.

This is a decoding method, not pruning. In Sec. 3--4, the authors define a
partially unmasked generated state `y^(t)`, mask a candidate set `S_t` of
already visible tokens, and evaluate the old token under the perturbed context.
Their instability score is Eq. (2),

`ell_i = -log p_theta(Y_i = y_i^(t) | tilde y^(t))`,

and the worst-case perturbation objective is Eq. (8). Algorithm 1 uses a
margin-based candidate set, one additional forward pass, and revises the
largest-instability positions. Appendix C, Eqs. (the robust score and its
lower-bound argument), establishes only that the measured score lower-bounds
the worst-case instability for the selected subset; it does not establish a
static-weight importance score.

The experiment is on LLaDA-8B-Base with 128 steps, length 512, greedy
decoding, candidate size `m=32`, revision interval `E=8`, and one revised token
per invocation (Sec. 5.1, Algorithm 1). The compute-matched controls are random
remasking and margin remasking. Under the same low-confidence base, Table 2
reports GSM8K 51.40/52.69, HumanEval 12.20/17.07, and MBPP 15.60/24.80 for
base/CoRe, while random and margin controls are nearly unchanged. The
candidate-size and interval ablation (Table 4) peaks around `m=32,E=8`; `m=64`
degrades, consistent with excessive masking destroying the context used to
judge brittleness. The paper explicitly limits the claim to internal
structural consistency and says it does not guarantee factual correctness
(Sec. 6). This is strong evidence that a generated-context response change is
meaningful for DLM inference, but it is not evidence that a permanent weight
mask should preserve that response.

**Application to A+C.** The transferable measurement is the *state/action
pair*: a context that a model generated, and the same query after a targeted
visible-context perturbation. The non-transferable parts are CoRe's remasking
decision, its margin screen, and its inference-time extra forward pass. In our
setting the perturbation should be used once during calibration to rank static
weight capacity; it cannot be described as a CoRe implementation or as an
inference-time revision method.

### B. Standard MDLMs do not reliably score corrupted visible tokens

**Corrective Diffusion Language Models**, Zhang et al., arXiv:2512.15596v2,
2026. Primary source: <https://arxiv.org/html/2512.15596>.

This is post-training/fine-tuning of DLMs, not static pruning. Sec. 2.1,
Eq. (the absorbing objective), points out that standard MDLM training applies
loss only where `z_i = mask`; logits at visible positions are ignored. Sec. 2.2,
Eqs. (confidence and remasking), formalizes the usual refinement rule. The
authors' Code Revision Benchmark (CRB, Sec. 3) injects type-preserving,
executable code corruptions and tests localization plus iterative correction.

The key observed result is negative for the unmodified base behavior: in Sec.
4.1, confidence gaps between clean and erroneous visible tokens are small and
Top-1 error localization is weak, even though Top-5 is better. The proposed
absorbing--uniform mixture explicitly supervises visible corrupted tokens and
improves correction. On LLaDA-2.0-mini, Table 3 shows that the correction-aware
model benefits more from remasking on high-entropy ParallelBench tasks (for
example Shuffle fixed/remask 0.45/0.64 versus the standard MDLM 0.47/0.38).
Appendix E.2 holds the decoding schedule fixed and reports MBPP: MDLM versus
corrective model is 0.106/0.131 Pass@1 under vanilla decoding and 0.108/0.141
under ReMDM (Table 6). Appendix D also shows that training with too many
simultaneous updates creates a train-test distribution mismatch and worsens the
quality estimator.

**Application to A+C.** This gives a sharper reason to add generated/corrupted
states: the dense model's clean gold-reveal responses do not expose the model
to the visible-but-wrong contexts that a sparse model can create. The paper
does not imply that a base LLaDA checkpoint has a reliable error detector, nor
that a static mask can learn correction ability without fine-tuning. The safe
claim is only that the calibration bank should test sensitivity to such states,
and that a static mask selected from clean states may fail there.

### C. Path consistency is a training principle, not a pruning criterion

**Consistent Diffusion Language Models**, Amin et al., arXiv:2605.00161v2, 2026.
Primary source: <https://arxiv.org/html/2605.00161>.

The paper introduces Multi-Path Discrete Consistency (MPDC) and trains CDLMs;
it does not prune a frozen LLaDA checkpoint. Sec. 2, Eqs. (1)--(2), defines the
exact discrete posterior bridge between corruption levels. Sec. 3.1, Eqs.
(3)--(6), defines path consistency: a prediction at `x_t` should agree in
expectation with a prediction after an intermediate bridge hop to `x_s`. The
paper stresses that the optimum is a per-token posterior marginal, not the
full joint distribution (Sec. 3.1), so path consistency does not remove the
factorization barrier.

The max-step diffusion anchor is required for an identifiable Bayes fixed point;
unanchored self-consistency admits degenerate fixed points. Sec. 4.4 and the
listed ablations compare step-size schedules, max-step anchor/regularizer, and
divergence choices. CDLM improves few-step generation, but this is a trained
model-family result, not evidence for a path-invariant static weight mask.

**Application to A+C.** Exact bridge states could be a future pair bank if the
research question becomes robustness across corruption levels. They should not
be silently substituted for the current natural nested pairs: bridge sampling
changes the state distribution and CDLM's trained objective is absent from
LLaDA-8B-Base. The useful caution is to test path/state generalization, not to
claim “conditional distribution consistency” as a new pruning objective.

### D. Conditional dependence can be invisible to forward or scalar endpoint loss

**Generation Order and Parallel Decoding in Masked Diffusion Models: An
Information-Theoretic Perspective**, Zhang et al., arXiv:2602.00286, 2026.
Primary source: <https://arxiv.org/html/2602.00286>.

This paper studies decoding distributions, not pruning. Sec. 3.4 defines the
factorized within-block approximation (Eq. 5) and shows why conditional
dependence creates parallelization bias. Sec. 4, Eqs. (7)--(12), decomposes
rollout error under model error and motivates easy-first ordering under an
explicit entropy-error assumption. Sec. 5.2, Eqs. (15)--(18), distinguishes
forward conditional total correlation from reverse conditional total
correlation. The reverse quantity is evaluated under samples produced by the
factorized decoder and is sensitive to support mismatch; the forward quantity
can remain modest when invalid configurations receive probability mass. Their
verification lower bound is Eq. (19), with expected proposal count at least
`exp(forward conditional TC)`.

The Block-HMM ablation (Sec. 6.1) makes this concrete: at low parity noise,
mean-field decoding has reverse KL above 70 nats and incoherence near 50% while
forward KL stays around 11--13 nats. The arithmetic experiments use LLaDA and
compare L2R/R2L/easy-first schedules. This establishes a readout warning:
matching a scalar gold-vs-rest score or average forward error can miss
wrong-vs-wrong rearrangements and support-invalid joint outcomes.

**Application to A+C.** Use a full-vocabulary output-fidelity control and a
small structured/incoherence diagnostic before promoting scalar A+C. Do not
import conditional-TC as a new scalar without a measured mapping to the static
mask. The paper does not show that its blockwise reverse KL can be computed
cheaply for the LLaDA pruning bank.

### E. Self-generated calibration can help pruning, but the evidence is AR

**Rethinking Pruning Large Language Models: Benefits and Pitfalls of
Reconstruction Error Minimization**, Shin et al., EMNLP 2024, pp. 1182--1191.
Primary sources: <https://aclanthology.org/2024.emnlp-main.68/> and
<https://aclanthology.org/2024.emnlp-main.68.pdf>.

This is the closest pruning precedent, but it evaluates LLaMA-7B and OPT-125M,
not DLMs. Eq. (1) formulates pruning as dense/sparse prediction reconstruction.
Sec. 2 Eqs. (2) and Fig. 2 compare block reconstruction, global propagation,
and cross-block reconstruction. On 50% unstructured pruning of LLaMA-7B with
Wanda/SparseGPT/Magnitude, Sec. 3.1/Fig. 3 reports over 90% lower final-block
reconstruction error from the engineering variants. Sec. 3.2/Fig. 4 then
shows the important counter-result: aggressively minimizing reconstruction on
small calibration data can overfit and worsen test error, perplexity, and
downstream tasks. Increasing self-generated calibration data improves both
test reconstruction error and perplexity. Appendix C generates 10,240 dense
model texts (four deterministic prefix tokens followed by stochastic
generation), but notes that some generations are code or otherwise irrelevant
to the target English evaluation.

**Application to A+C.** Self-generated data is evidence for measuring the
deployed state distribution and for charging calibration data as a design
choice. It is not evidence that generated contexts beat clean contexts for
LLaDA or that reconstruction proxy gains predict NELBO. The paper's exact
warning supports an independent NELBO split and a content/domain filter rather
than simply adding more generated trajectories.

### F. Other already-read primary sources rechecked

The following original papers were reopened and their claims remain bounded as
below.

| Paper and primary source | Verified observation | What it permits here | What it does not permit |
|---|---|---|---|
| **Induction in Both Directions**, Catruna & Radoi, arXiv:2607.15893v2, Sec. 4, Table 1, Fig. 2. <https://arxiv.org/html/2607.15893> | Small matched attention-only DLMs use previous and next context; induction is direction-symmetric; a residual direction carries mask-rate information. | A context-update pair is a plausible DLM functional diagnostic. | Transfer of a 1--3-layer toy circuit to LLaDA-8B or a pruning score. |
| **Parallelism and Generation Order in MDLMs**, Zhong et al., arXiv:2601.15593, Sec. 3 Eq. (2), Appendix/benchmark tables. <https://arxiv.org/html/2601.15593> | Within-step decoding uses a factorized approximation; order and parallelism vary by task and model; the paper separates the two. | Keep decode-state/order as a validation stratum. | A universal easy-first or order-weighted static mask. |
| **Masks Can Be Distracting**, Piskorz et al., arXiv:2511.21338v2, Sec. 5.3/Table 1, Appendix A. <https://arxiv.org/html/2511.21338> | LLaDA-Base is highly sensitive to added masks; added-mask gradients exceed non-mask gradients; context comprehension worsens with extra masks in long-context tests. | Treat mask count as a measured confound and preserve it in a controlled pair. | Interpreting a gold reveal gain as semantic context value, or copying mask sensitivity to weights. |
| **Measuring Temporal Linguistic Emergence**, Lu, arXiv:2604.23235, Sec. 3--4, Appendix Tables 1--2. <https://arxiv.org/html/2604.23235> | POS/semantic information is more recoverable than lexical identity; re-masking sensitivity peaks around steps 15--18, with 10.92 direct versus 0.03 collateral drop in the reported 32-step baseline. | Stratify held-out checks by state and report a middle-window diagnostic. | A universal clock weight, MI score, or temporal scalar without pruning evidence. |
| **Subliminal Clocks**, Rulli et al., arXiv:2607.01774v2, Eqs. (1)--(7), Sec. 4.3/Fig. 4. <https://arxiv.org/html/2607.01774> | LLaDA/Dream residuals encode denoising progress; steering Eq. (4) changes confidence/entropy; norm-matched random perturbations are weaker; deeper layers can compensate for early perturbations. | Use progress only to stratify or stress-test calibration states. | A clock-preserving weight objective or a claim that clock MI is importance. |
| **FAIR-Calib**, arXiv:2606.06547, Sec. 4 Eqs. (11)--(15), Table 3. <https://arxiv.org/html/2606.06547> | PTQ error can flip frontier commitments; teacher occupancy gives a trajectory KL decomposition; frontier-hit and masked-stage reliability are complementary (Table 3: 63.12, 62.89, 64.64 vs 61.76 baseline). | Compare against frontier/occupancy weighting and cite it as adjacent prior art. | Claiming teacher-trajectory weighting or irreversible frontier protection is new to pruning. |

## Synthesis: what the literature changes

The literature supports three observations, with different status:

1. **Observed in primary work:** DLM inference visits partially visible contexts;
   context changes can make an earlier token brittle (CoRe); standard MDLMs are
   weak at identifying visible corruptions (Corrective DLM); and generated
   calibration can reduce AR pruning overfit (Shin et al.).
2. **Plausible but unestablished for this project:** a static mask selected on
   clean/gold context can lose capacity needed to preserve responses after a
   generated or corrupted visible commitment; this may matter more for
   structured reasoning than ordinary endpoint reconstruction.
3. **Not supported as a new direction:** clocks, average timestep weighting,
   generic trajectory banks, unconditional reconstruction minimization,
   arbitrary shuffling, or a new constraint/solver by themselves. Existing
   Reveal-KL, FAIR-Calib, generic output preservation, and search literature
   already cover adjacent ingredients.

The strongest opposing evidence is also clear. CoRe and Corrective DLM alter
decoding/training behavior; neither evaluates a frozen static mask. Corrective
DLM explicitly finds that standard MDLM confidence is poorly aligned with
visible corruption, so a dense model's generated error states may be noisy.
Shin et al. find calibration-data generation helps, but on AR models and with
domain filtering. The information-theoretic paper warns that even a better
forward/scalar response proxy can miss support-invalid joint outcomes. The
current A+C mini result is underpowered and selected on the same bank.

## One concrete candidate: context-response calibration (clean-first, GCR extension)

This is a hypothesis, not an approved experiment. It changes the pair bank and
state distribution while keeping the natural-bank scalar endpoint A and the
exact-budget hard-mask allocator as reference factors. The generated-context
variant uses a separate full-vocabulary response readout; it is not a scalar
readout replacement. It should not be called CoRe, Corrective DLM, CDLM, or
on-policy pruning; those papers do different things.

### Hypothesis

At fixed sparsity and a fixed within-projection Wanda support, a mask that
preserves both endpoint fidelity and the dense model's *conditional response to
a controlled context perturbation* will retain more deployment-relevant
denoising behavior than endpoint-only A. The generated-context variant asks
whether this remains true on states visited by the dense decoder. The
hypothesis predicts an increment over A-only only if the response term improves
held-out quality, not merely its calibration score.

### Pair construction

1. Freeze the model revision, tokenizer, prompt format, seed, sequence length,
   decoding schedule, and native support family. Use a dense teacher to produce
   trajectories on calibration prompts under the same 256-step protocol used
   for the current work. Save each partially visible state, its mask set, the
   newly committed token, and the remaining query positions.
2. For a state `s`, choose a visible generated position `j`, a still-masked
   position `k`, and a query set `Q` that excludes `j` and `k`. Construct a
   paired state `s'` by holding sequence length and `|mask(s)|` fixed: mask
   `j`, reveal `k` with the teacher's generated token (or an explicitly labeled
   top-2/nucleus alternative in a separate corruption arm), and keep `Q`
   unchanged. This one-for-one swap changes which evidence is visible while
   controlling mask count. It does not prove a pure semantic effect; token
   position, generated-token quality, and progress remain recorded strata.
3. Add a second, separate perturbed-visible arm: replace one generated visible
   token at `j` with a type-preserving alternative sampled from the dense
   distribution, keeping the mask set and `Q` exactly unchanged. This
   alternative is not assumed to be incorrect or corrective; it is a controlled
   context perturbation. The primary generated bank should use the actual
   generated state plus this perturbation; clean gold-reveal pairs remain the
   deployment-like reference condition.
4. Exclude the calibration prompts used to build the bank from NELBO and task
   validation. Do not cross-sequence-shuffle variable-length logits. Any null
   must preserve exact query coordinates, endpoint marginals, and mask counts.

### Readout

There is a critical label distinction. A dense-generated continuation does not
have a ground-truth token for every still-masked query. Reusing the current
gold-vs-rest scalar with the teacher's top-1 as if it were gold would turn the
new bank into a pseudo-label experiment and could hide teacher errors. Therefore
the primary GCR response readout should use the dense and sparse *full logit
vectors* on the fixed query set, after subtracting the vocabulary mean. Let
`z_D(s,Q)` and `z_M(s,Q)` denote these centered vectors. Define

`C_GCR(M) = mean_(s,s') || [z_M(s',Q)-z_M(s,Q)] - [z_D(s',Q)-z_D(s,Q)] ||^2.`

Keep endpoint A on the natural gold-reveal bank with the existing scalar
gold-vs-rest readout, and use `C_GCR` as a separate response challenger. If a
known corpus continuation is available for a query, report a matched scalar
gold-vs-rest version as a diagnostic; otherwise do not call a teacher-generated
token gold. The full-vocabulary endpoint error is a required control for both
banks, because the information-theoretic paper shows that scalar/forward losses
can miss wrong-vs-wrong and support failures. Report a small joint-incoherence
or top-token-flip diagnostic on held-out states if available; do not call it a
new universal metric.

Use `A_natural + C_GCR` only as a predeclared challenger, with the same hard
mask solver and exact budget. This deliberately changes one information axis
(generated context response) while retaining the existing endpoint A
reference; it should not be presented as an apples-to-apples replacement of
the natural A+C scalar objective until a common full-vocabulary control is
measured.

### Algorithm 1 candidate (common hard-mask backend)

1. Build and freeze the existing A-only anchor `M_A` with its declared
   calibration cost and native support/rank family.
2. For each arm, start from the same `M_A`, exact global removal count, frozen
   Wanda within-row ordering, original surviving weights, and GCR/natural state
   bank. The arms are: (i) A continuation minimizing endpoint A, (ii) current
   natural A+C continuation, and (iii) GCR-A+C continuation minimizing
   `A_natural+C_GCR`. A constrained-C arm may be included only as the already
   specified fixed-anchor challenger `C` subject to `A <= A(M_A)+epsilon`; it
   is not required for the first literature screen.
3. Propose only count-preserving boundary trades in the common 32-layer tied
   lattice. Materialize each complete jointly sparse mask and evaluate its
   current-bank objective in the sparse model; do not add old marginal costs.
   Accept only a predeclared decrease beyond numerical tolerance. Cache
   per-pair values after an accepted incumbent and charge all candidate/state
   evaluations to every arm.
4. Freeze the masks before reading independent quality. Compare held-out
   predeclared primary held-out downstream metric (for example, full GSM8K
   after development freeze) at identical 50% sparsity. Report WikiText
   NELBO/MC as a complementary quality and tradeoff diagnostic; do not require
   NELBO and downstream accuracy to move in the same direction. Keep the
   current mini-100 as development evidence only. Report natural-bank,
   GCR-bank, mask-count, and generated-perturbation strata separately.

This algorithm is a common exact-budget exchange backend, not a novelty claim.
The only candidate-specific claim is that controlled context-response pairs add
independent downstream value beyond endpoint A and the existing natural A+C
bank.

### Minimum comparison and stop rule

The smallest informative factorial is:

| Mask/allocator | Natural gold-reveal bank | GCR generated/perturbed bank |
|---|---:|---:|
| native Uniform reference | yes | optional diagnostic |
| endpoint A | yes | yes |
| current scalar A+C | yes | same-bank control only |
| GCR-A+C | same-solver control | yes |
| full-vocabulary endpoint fidelity | yes | yes, if compute permits |

The candidate is supported only if the predeclared primary held-out downstream
metric improves over endpoint A at equal exact sparsity and comparable measured
compute. NELBO/MC and the held-out response diagnostic are complementary
evidence and attribution tools; neither must improve together with the primary
task metric. If the GCR gain disappears under same-count pairs, this indicates
sensitivity to count, evidence identity, or support construction, but does not
uniquely identify a progress effect. If it appears only on perturbed states but
hurts clean task quality, it is a robustness trade-off rather than the main
pruning method. If all three continuation arms improve similarly, credit the
shared solver or calibration rather than C.

### Priority: clean-bank alternative before generated-state GCR

The clean-bank alternative should precede the high-shift GCR variant. First
construct query-aligned, same-count context pairs from the existing corpus
continuations, preserving true query labels and the natural bank as a reference.
Compare endpoint A against `A_natural + C_clean` with the same hard-mask solver.
This is cheaper, avoids teacher pseudo-labels, and directly tests whether the
current gold-reveal/mask-count confound is responsible for the observed C
signal. It still does not isolate semantics perfectly because the visible
evidence locations change.

Only if the clean-bank response term has a predeclared downstream or useful
cost-quality signal should the generated-state GCR extension be considered.
GCR then tests deployment-state coverage and visible-token perturbation
robustness, at higher calibration cost and with a full-vector readout. If the
clean-bank alternative is neutral or harmful, the literature does not justify
escalating to GCR merely because generated contexts are intuitively realistic.

## Decision

Keep one candidate family: **context-response calibration**, with the
query-aligned clean-bank alternative as the first test and generated-context
GCR as a higher-risk extension. The current natural A+C bank and endpoint A are
mandatory controls. This is a narrow, testable extension motivated by context
brittleness and visible-state exposure. Do not add clocks, SAE features,
activation coverage, generic rollout stacks, Fisher/Jacobian terms, or another
optimizer to it. Do not launch an experiment automatically. The next evidence
is a predeclared held-out downstream comparison under the same hard-mask solver
and exact budget, with NELBO reported as a complementary diagnostic; the
current A+C mini 61 versus A 55 is not sufficient.

## Sources checked

1. CoRe, <https://arxiv.org/html/2602.04096>.
2. Corrective Diffusion Language Models, <https://arxiv.org/html/2512.15596>.
3. Consistent Diffusion Language Models, <https://arxiv.org/html/2605.00161>.
4. Generation Order and Parallel Decoding, <https://arxiv.org/html/2602.00286>.
5. Mean-Field Parallel Decoding, <https://arxiv.org/html/2606.15805> (decoding-only adjacent prior; Algorithm 1, Eqs. (4)--(9), Tables 1 and 3).
6. Rethinking Pruning Large Language Models, <https://aclanthology.org/2024.emnlp-main.68/>.
7. PRISM, <https://arxiv.org/html/2510.01384> (per-token quality/self-correction boundary and scalar-vs-likelihood ablation).
8. Induction in Both Directions, <https://arxiv.org/html/2607.15893>.
9. Parallelism and Generation Order in MDLMs, <https://arxiv.org/html/2601.15593>.
10. Masks Can Be Distracting, <https://arxiv.org/html/2511.21338>.
11. Measuring Temporal Linguistic Emergence, <https://arxiv.org/html/2604.23235>.
12. Subliminal Clocks, <https://arxiv.org/html/2607.01774>.
13. FAIR-Calib, <https://arxiv.org/html/2606.06547>.
