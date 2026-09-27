# Design

## Context

See proposal.md for motivation and the three delta specs for behavior contracts. This is planning only; no experiment is running because of this change.

Observed implementation:
- `dlm_context_response50/core.py` creates 80 paired contexts and uses scalar gold-vs-rest log-odds, A = mean endpoint squared error, C = mean squared error difference. `rank_rates` maps 32 signed marginal costs to a 45–55% schedule.
- `dlm_multiscale_ac50/analysis.py` reuses `dlm_owl65.core.exact_row_counts`; this DP rounds budgets, not the response objective. The runtime checks source hashes and rejects modified dependencies of started experiments.
- The multiscale runtime already offers atomic per-state/per-document checkpoints, exact-mask identities, native scoring, tmux and GPU occupancy checks. Its probe sharding is fixed to one or two shards and its pipeline has global phase barriers; the new scheduler must not inherit those limitations.
- Observed multiscale state remains interrupted: 64 probe conditions and five allocations exist; Short/Multi each have 56 completed documents. These counts are discovery evidence, not hard-coded completion logic.
- Obsidian Research-State was read successfully during planning; its latest append covers the theoretical multiscale design, while later implementation/run evidence is in local receipts. Neither overrides newer explicit user decisions.

## Goals / Non-Goals

**Goals:** one reproducible manifest, ten arms with isolated changes, bounded computation, shared within-family measurements, any explicitly selected number of GPUs, useful status/stop/resume, and a development report that can contain negative results.

**Non-Goals:** see proposal.md exclusions. In particular, this is not a full factorial study, a new optimizer claim, or a proof that A+C causes reasoning improvement. Layer-global Uniform is not the native row-wise reference. Prior DKD-style conditional-probability and A-floor designs remain separate candidates. Centered logits and a legacy-AC warm start are the bounded choices of this screen, not synonyms for those candidates, not proven superior, and not evidence that the user prescribed every numerical default.

## Decisions

### 0. Theory, provenance and mathematical contract

This section is normative for the mathematical meaning of the screen. Sections 1–9 specify its implementation. Freeze an `objective_version` and exact bank/edge/readout/reduction identities in every new receipt. Existing Multi artifacts retain their original objective identity, with a separate adapter mapping; do not rewrite historical receipts to add new fields. A mismatch with the frozen implementation fails import instead of silently “correcting” a started experiment.

#### 0.1 Source-to-design ledger and limits

| ID / source | Supported ingredient | Adaptation in this screen / limit |
|---|---|---|
| T1: [Heidari–Pradhan–Venkataramanan 2019, §II–III, Lemma 1, Eq.13–15](https://arxiv.org/html/1901.10576) | Biased product bases; cross-correlation for independently coupled coordinates with different marginals | Apply the real-valued expansion to dense–sparse residuals on visible/masked bits. It motivates the logit-visibility coordinate; it does not prove an optimal pruning allocation. |
| T2: [O'Donnell, Analysis of Boolean Functions, Ch.8](https://arxiv.org/abs/2105.10386) and [Noise Stability of Transformer Models, §6](https://arxiv.org/html/2602.08287) | Product-space noise analysis; an existing Transformer noise-stability application | Historical origin of the multiscale idea. The stationary resampling formula is not the formula for our cross-phase reveal bank. Transformer training gains are not DLM pruning evidence. |
| T3: [Shapley–Taylor, Eq.2–3](https://proceedings.mlr.press/v119/sundararajan20a/sundararajan20a.pdf) | Mixed finite differences over a set function | Four reveal states expose group-combination residuals. We neither compute the Shapley–Taylor attribution nor inherit its attribution axioms as loss guarantees. |
| T4: [DKD, §3 Eq.5–7](https://arxiv.org/html/2203.08679) | Exact target/non-target KL decomposition and decoupled weights | Distinct deferred probability-readout candidate, not the source of a mandatory centered-logit metric or proof that it is worse. |
| T5: [Sobolev Training, §3](https://arxiv.org/html/1706.04859), [2ndMatch, §4 and Table 4](https://arxiv.org/html/2506.05398) | Output-plus-sensitivity matching precedents | Finite masked-context differences are not Jacobians. 2ndMatch is image-diffusion finetuning; its first-order ablation did not improve KD, so response matching is not automatically beneficial. |
| T6: [CoRe, §3–4](https://arxiv.org/html/2602.04096) | Context perturbation supplies a DLM revision signal different from static confidence | Supports functional relevance of context response; it does not validate our static masks, gold-reveal calibration or loss. It is supporting evidence, not a claim about the sole historical origin of A+C. |
| T7: [SPDY, §3–4](https://proceedings.mlr.press/v162/frantar22a/frantar22a.pdf), [EvoPress, §3](https://arxiv.org/html/2410.14649) | Whole-model informed allocation and compression-preserving exchanges | Reuse optimization principles openly. No new optimizer or convergence theorem is claimed for our capped deterministic variant. |
| T8: [LLM KD via Interactions, §5.2–6](https://arxiv.org/html/2607.08776) | Useful interaction preservation can trade off with suppressing other interactions | Counterevidence to “preserve every complex interaction more strongly.” Square changes error weighting; its benefit remains a hypothesis. |

Primary-source relevant sections were checked in the source review; this is not an exhaustive novelty certification. Historical derivations that led to this plan:
- [Scalar theory synthesis](../../../research/theory_ac_upgrade_synthesis_2026-09-22.md), especially §3–4.
- [Monotone Multi derivation](../../../research/ac_multiscale_monotone_design_2026-09-22.md), especially §3–7.
- [Original centered-logit proposal](../../../research/four_axis_synthesis_2026-09-18.md).
- [DKD follow-up](../../../research/literature_method_synthesis_2026-09-22.md).
- [Allocation review](../../../research/allocation_design_review_2026-09-18.md).

These older notes contain alternatives, not permission to combine them implicitly. Equation-defined behavior below and frozen artifacts take precedence over illustrative earlier settings. The ledger separates published ingredients, our algebraic adaptations, untested hypotheses and numerical defaults.

#### 0.2 Common endpoint/response interpretation

For a fixed clean span, query token y and context x, let D be dense and M the frozen-weight sparse model. The legacy scalar is

    f_X(x) = log[p_X(y|x)/(1-p_X(y|x))]
           = z_X,y - logsumexp(z_X,v for v != y).
    e(x) = f_M(x) - f_D(x).

Compute this scalar with the legacy FP32 logit routine, then residuals and squared-error reductions in FP64. Do not compute a rounded probability then its odds. Before/after refer to context visibility, not different target tokens. The query must remain masked and have the same corpus gold in both states.

For one pair, A=(e0²+e1²)/2 and C=(e1-e0)². With u=(e0+e1)/2 and v=(e1-e0)/2, A+C=u²+5v². C reweights varying residuals; perfect endpoint matching already implies C=0. Neither C alone nor agreement with a possibly wrong teacher is a task-accuracy guarantee. Minimize the RESPONSE RESIDUAL `(f_M1-f_M0)-(f_D1-f_D0)`, not the sparse model's response itself: legitimate teacher sensitivity should not be suppressed.

More generally for node residuals e and edge weights w summing to one, C=sum_edges w_ij*(e_j-e_i)² is a graph quadratic form e^T L_w e (coordinate-wise also for vector readouts). The input construction, readout, and allocation solver are independent design axes. A,C and every marginal must keep their declared averaging denominators; adding more edges must not multiply lambda inadvertently.

#### 0.3 Multi: product coupling, scale and exact five-arm definitions

Import the verified bank from `experiments/dlm_multiscale_ac50/output/config.json` and its referenced files. The frozen setup is 8 spans × 2 chains × K=8 states =128 states per split, 8 fixed masked queries per span (shared across both chains), calibration seed 20260922 and diagnostic seed 20260923. Conceptually s_j=logit(.05)+j*(logit(.95)-logit(.05))/7 and p_j=sigmoid(s_j). Use the STORED floating-point p_j and inputs when importing, not newly rounded values.

For each eligible non-query position sample U_i independently once per chain, then V_i(p)=1[U_i<=p]. The same U_i is used at all phases. This is a nondecreasing gold-reveal chain; identical adjacent input states are allowed. p denotes eligible-context visibility probability, not actual full-input visibility or model-call count. Q is selected before the masks. Do not force any reveal, resample unchanged transitions, fix the total mask count, condition the sampling on teacher predictions, or insert generated tokens while claiming this product-coupling theory.

**Direct adaptation of T1:** expand e(V(p)) in its p-biased orthonormal basis with coefficients a_p,S. For p<r,

    rho(p,r) = sqrt[p*(1-r)/(r*(1-p))]
             = exp(-(logit(r)-logit(p))/2),
    E[(e(V(r))-e(V(p)))²]
      = sum_S [a_p,S² + a_r,S² - 2*rho(p,r)^|S|*a_p,S*a_r,S].

Different phases have different bases/coefficients. Do NOT replace this with `2*sum_k(1-rho^k)*E_k`; that expression assumes same-marginal stationary resampling, not this bank. These are population identities; a finite 16-chain sample is an estimator, not an exact empirical spectral decomposition. No Fourier coefficients are estimated or fitted by the implementation.

For one chain and query:

    A = (1/K)*sum_j e_j²
    E1 = [(0,1),(2,3),(4,5),(6,7)]
    E2 = [(0,2),(1,3),(4,6),(5,7)]
    E4 = [(0,4),(1,5),(2,6),(3,7)]
    Cd = (2/K)*sum_(i,j in Ed) (e_j-e_i)²
    C_path = (1/(K-1))*sum_(j=0..K-2) (e_(j+1)-e_j)²
    C_all = (2/(K*(K-1)))*sum_(i<j) (e_j-e_i)²

| Arm | Loss on the SAME bank | Question answered |
|---|---|---|
| MS-A (stored legacy arm name A) | A | Does response weighting help beyond these endpoints? |
| Short | A+C1 | Are disjoint short intervals sufficient? |
| Path | A+C_path | Is connecting every adjacent phase sufficient? |
| All | A+C_all | Is global residual variance sufficient? |
| Multi | A+(C1+C2+C4)/3 | Does the declared mixture of interval scales help? |

Average queries, then two chains within each span, then eight spans equally, exactly as the frozen reducer. Each Ed is a perfect matching and exposes every endpoint equally. All has equal endpoint degree; Path endpoints have degree 1 and interior nodes degree 2, so Path does NOT have the same per-node exposure as Multi. Every edge-mean quadratic form here has trace 2, but equal trace is not equality of directional weighting. Do not retrofit degree normalization into the frozen Path control.

Direct algebraic checks, for K=2^m and uniform dyadic scale mean:

    0 <= C_multi <= 4*A
    4*Var_phase(e)/m <= C_multi <= 4*Var_phase(e)
    C_all = 2*K/(K-1)*Var_phase(e)       # population variance, ddof=0

For K=8, residuals [1,1,1,1,1,1,1,1] and [1,1,-1,-1,1,1,-1,-1] both have A=1 and C1=0, but C_multi is 0 versus 4/3. This shows a different preference, not extra information beyond perfect endpoint matching. The lower bound is about PHASE-INDEX graph modes, not input-token interaction order. Short is disconnected; Path is connected too, so connectivity alone cannot establish Multi's superiority. No inverse-gap or teacher-response normalization is used. Large-response scales can dominate and must be reported separately.

If documenting the response-sign bound, use exactly the Multi edge distribution and gamma>0:

    P(Delta_M*Delta_D <= 0 AND abs(Delta_D)>=gamma) <= C_multi/gamma².

This is an unconditional event probability, not the conditional flip rate among eligible edges. The old `response_sign_flip_rate` instead uses ALL pairs, strict product<0, abs(Delta_D)>1e-6, and conditions on eligibility. Preserve that legacy metric and its metadata; never report it as verification of this bound. This screen does not add a new sign-bound model diagnostic.

#### 0.4 Square: group interaction without an extra penalty

T3 motivates the four-node input family, not a Shapley loss. Use node order [00,10,01,11] and edges [(00,10),(00,01),(10,11),(01,11)]. Define f as in §0.2 and e_ab=f_M-f_D at the SAME fixed queries:

    A_square = (e00²+e10²+e01²+e11²)/4
    C_square = [(e10-e00)²+(e01-e00)²+(e11-e10)²+(e11-e01)²]/4
    I_X = f_X,11-f_X,10-f_X,01+f_X,00.

Writing e_ab=u+a*s_a+b*s_b+c*s_a*s_b with s0=-1,s1=1 gives

    A_square = u²+a²+b²+c²
    C_square = 2*a²+2*b²+4*c²
    A_square+C_square = u²+3*a²+3*b²+5*c²
    I_M-I_D = 4*c.

Do not add `(I_M-I_D)²=16*c²` as another loss term. Log u²,a²,b²,c² and the squared interaction residual from already available readouts as diagnostics; average squared per-query modes, not modes before squaring. A constant residual gives C=0; residual [1,-1,-1,1] gives A=1,C=4. Paired edge increments telescope along either path, so a difference of total path increments is identically zero and is not a valid additional objective.

At K=4, Multi's abstract dyadic graph is a square with the SAME A/C normalization. Its actual contexts are totally nested, whereas Square has incomparable x10 and x01 when both groups are nonempty. Graph equivalence does not make the input banks equivalent. Equal-sized groups match the two middle states' mask counts but do not isolate semantics; a nonlinear count-only function can have nonzero I. Group order 2 is not token order 2 when groups have many tokens, and neither is cross-layer pruning interaction. T8 cautions against assuming that more interaction preservation always helps.

#### 0.5 Vector: centered logits and explicit reductions

This is the earlier centered-logit proposal, not an implementation of T4/DKD. The motivation is the scalar blind spot: probabilities (.4,.5,.1) -> (.4,.1,.5) with gold coordinate 0 have unchanged gold odds, although the best alternative changes. A sparse model that stays at (.4,.5,.1) has scalar A=C=0. Both full-distribution endpoint matching and centered vectors detect the mismatch; this example alone does not prove that a C term is necessary.

At endpoint b, let z_X,b be the emitted V-dimensional logits of the unchanged model/head. Define P=I-11^T/V only algebraically (never materialize it) and E_b=P*(z_M,b-z_D,b). For pair i with its original Q_i queries:

    A_i = sum_(q in Q_i,v) [E_0,qv²+E_1,qv²] / (2*|Q_i|*V)
    C_i = sum_(q in Q_i,v) [E_1,qv-E_0,qv]² / (|Q_i|*V)
    A_vec = (1/80)*sum_i A_i; C_vec = (1/80)*sum_i C_i.

Pairs receive equal weight even when |Q_i| differs. Do not concatenate all query rows and take a global mean. With ten pairs per span this equals an equal pair-within-span then equal span mean. Endpoint identity includes ordered query positions and gold token IDs; vector readout covers ALL vocabulary coordinates including gold. There is no temperature, probability weighting, top-k, row norm division or mean subtraction across queries/endpoints.

The relevant identity for any residual vector r is

    (1/V)*||P*r||² = (1/V²)*sum_(v<w) (r_v-r_w)².

Thus this loss preserves relative logit margins and is invariant to an independent common vocabulary shift at either endpoint of either model. Equal treatment of low-probability logit coordinates is a design trade-off, not a proven best geometry.

Numerical contract for NEW vector scores: cache raw emitted dense logits losslessly as FP32 (the model remains BF16); convert raw dense and sparse query logits to FP64 chunks for centering, residual arithmetic, squared sums and means. Compute each query's whole-vocabulary mean before centering its chunks; subtracting each chunk's own mean changes the objective and is forbidden. Cache raw logits, not pre-centered FP32 vectors. Compare chunked results against full FP64 reductions with atol=1e-10, rtol=1e-9 on small bounded CPU fixtures. Floating reductions need not be bit-identical across chunk sizes; numerical tolerances do not permit changed formulas.

The identity `Q=U^T P U/V` for a fixed affine head gives the same loss from post-finalnorm hidden residuals in real arithmetic (including any fixed logit scale squared). Actual BF16 head rounding is not represented by that algebra. Plain hidden MSE is different; Gram/hidden-only optimization remains deferred pending emitted-logit equivalence and cost checks. No new model head or surviving-weight update is allowed.

#### 0.6 DKD retained as a distinct deferred competitor

T4 exactly decomposes full KL using b_X=(p_X(y),1-p_X(y)) and q_X(v)=p_X(v)/(1-p_X(y)), v!=y:

    KL(p_D||p_M) = KL(b_D||b_M)+(1-p_D(y))*KL(q_D||q_M).

The prior DKD adaptation proposed endpoint matching `A_dec=mean_endpoints[KL(b_D||b_M)+beta*KL(q_D||q_M)]` and response matching `C_q=mean_pairs,queries[(1/(V-1))*sum_(v!=y)((q_M1-q_M0)-(q_D1-q_D0))²]`. q is computed stably by a softmax excluding gold. A non-target endpoint term is necessary to detect a common alternative-token bias across both endpoints. The response term is OUR adaptation, not a theorem or method from DKD. The non-target endpoint weight beta deliberately differs from vanilla KL's teacher-mass attenuation.

Older notes use both summed and vocabulary-mean response norms; these differ by V-1 and cannot share the same numerical coefficient. The formula here records the later vocabulary-mean convention, but beta/gamma/temperature and a coefficient-scale rule are NOT frozen for a DKD run. In particular lambda=1 of scalar/logit losses cannot be transplanted into a mixture of endpoint KL and probability MSE without an explicit normalization decision. Probability saturation and low-probability changes can matter; there is no theorem ranking this against centered logits.

No DKD job is part of this ten-arm revision. A future separate comparison needs full-vocabulary endpoint KL, A_dec, and A_dec+gamma*C_q on one bank/backend with fixed compute and coefficients. Merely beating scalar AC would not isolate the value of decoupling or C. This deferral is a scope choice, not a scientific rejection; do not silently implement DKD under Vector-A/Vector-AC labels.

#### 0.7 Allocation is a heuristic followed by measurement

For new Square/Vector objectives, compute g_l=(L_l,52-L_l,48)/(N_l,52-N_l,48) on the actual full sparse models, preserving signed costs. All other blocks remain native Uniform50. With ascending average ranks r_l in [1,32],

    rank01_l=(r_l-1)/31
    ideal_sparsity_l=.5-.10*(rank01_l-mean(rank01)).

Larger positive costs therefore receive LESS sparsity. Preserve original tie rules, per-row ordering and exact-quota DP. The DP minimizes integer budget-rounding deviation, not L; neither the rank mapping nor a sum of independent probes guarantees the best assembled mask. Final calibration and diagnostic scores must be measured on each assembled physical mask, not inferred from marginals. Do not declare the existing failed local extrapolation evidence against every local method.

Exchange tests whether further hard-mask optimization of legacy A+C helps, not whether C is necessary under that solver. With donor removing DeltaN more weights and receiver removing DeltaN fewer, the first-order predicted LOSS DECREASE is DeltaN*(g_receiver-g_donor). This sign matters, but only actual full-bank loss can authorize acceptance. T7 supports feasible exchanges, not the frozen old-cost shortlist, warm start, d=41, epsilon or 24-candidate budget. Those remain explicit bounded-screen defaults; failure is not convergence or rejection of the entire exchange method. Adaptive radii, refreshed/rotating proposals and common-A-anchor controls remain separate follow-ups, not hidden behavior changes.

#### 0.8 Hypotheses, defaults and implementation acceptance

| Family | Testable hypothesis | Primary discriminator / interpretation boundary |
|---|---|---|
| Multi | Multiple reveal intervals improve allocation beyond endpoint/short-interval weighting | Multi vs MS-A/Short/Path/All on the same bank; all-pair control tests a simple variance explanation. |
| Square | Responses along two reveal branches supply useful error weighting | Square-AC vs Square-A; random groups do not identify semantic circuits. |
| Vector | Preserving vocabulary response adds value once full endpoints are represented | Vector-AC vs Vector-A. Comparison with legacy scalar changes the readout too. |
| Exchange | Current rank-based AC allocation leaves attainable calibration/quality gains | Exchange-AC vs legacy AC is continuation with extra compute, not an equal-cost solver contest or a C ablation. |

Same-bank A controls, source-linked definitions and exact budget are scientific controls. Square p values/group cap/query count/seeds, lambda=1, 45–55% range, exchange warm start/radius/caps and memory limits are declared defaults or inherited settings, not values dictated by the cited papers. They must be recorded before model work and must not be tuned against mini100. The current revision preserves the existing ten-arm scope and exchange caps while making these distinctions explicit.

Implementation acceptance requires source-linked CPU checks: cross-bias identity by finite enumeration; frozen Multi reducers/edges and matching weights; Square modes; centered-vector invariance/margin identity and query-weighting; count-conserving exchanges and measured-only acceptance. Checks of algebra establish formula correctness only. Successful implementation, lower calibration loss, and downstream superiority are three different outcomes. A negative task result does not justify changing the sampling to obtain a preferred direction.

### 1. Separate implementation and immutable historical runs

Add `experiments/dlm_ac_screen50/` with modules for CPU preparation/manifests, banks/objectives, probes/allocation, exchange, GPU workers, orchestration, reporting and CPU tests. Import stable helpers from old modules without editing them. A new output root holds the manifest, imported-source receipts, fresh artifacts, jobs and reports. Imported legacy results keep their original hashes plus an adapter receipt; never relabel them with the new config fingerprint.

The scheduler takes a lease on the existing multiscale pipeline root before resuming it, then invokes the existing candidate worker interface with the original output root and environment under the new tmux session. It schedules each distinct ready candidate directly instead of relaunching the old all-phase pipeline. Original validators run before consuming old receipts. New reports map original arm `A` to display ID `MS-A` without changing stored legacy identity. Missing original phases can be scheduled with their original worker entry points; missing/corrupt sources are errors, not permission to rebuild under a different config.

Alternative rejected: adding arms into the existing started run changes source/config hashes and invalidates previously completed work.

### 2. Freeze the scientific experiment matrix

| Arms | Bank/readout | Allocation | Primary control |
|---|---|---|---|
| MS-A, Short, Path, All, Multi | Original nested 128-state banks, scalar | Existing 64 probes and five allocations | Multi vs four same-bank controls |
| Square-A, Square-AC | New 40 quartets / 160 states, scalar | Shared 64 block probes, old signed-rank/exact-budget backend | Square-AC vs Square-A |
| Vector-A, Vector-AC | Original 80 pairs / 160 states and original queries, centered logits | Shared 64 block probes, same backend | Vector-AC vs Vector-A |
| Exchange-AC | Original scalar 80 pairs | Bounded hard exchange from legacy AC | Exchange-AC vs legacy AC |

Legacy native Uniform/A/AC are three historical reference masks, not additional GSM8K generation jobs when identity checks permit reuse. New-bank calibration/diagnostic measurements of reference masks are separate measured jobs and must not be labeled cached scores. The first screen contains at most ten new-arm mini100 evaluations before physical-mask deduplication. Under the currently observed 56+56 imported documents, at most 888 new generations remain (44+44+3×100+2×100+2×100+100); actual missing counts are recomputed, never assumed. No extra confirmation request set is consumed.

All arms keep 32-block allocation, not free 224-projection allocation; seven projection ratios are tied subject to exact integer quotas. Surviving weights and within-row Wanda order are fixed. Seeds/decoding/model/budget are inherited explicitly from existing manifests. New Square bank seed is 20260926 and is an announced new-bank design choice, not equivalent to the legacy bank. For each active AC arm lambda is one; Multi uses alpha_d=1/3 within that C term. This does not specify coefficients for the deferred DKD candidate. There is no success-score target that stops or changes generation.

### 3. Square bank and objective

This implements §0.4 and T3 with random groups; group sizes and p values are declared sampling defaults, not values supplied by Shapley–Taylor.

- Reuse the same eight clean calibration spans, length 256; take the same diagnostic span source indices 8–15 used by multiscale, separately labeled as previously used development data.
- Freeze Square calibration seed 20260926 and diagnostic seed 20260927. Use the existing `seed_for` hash routine: `seed_for(split_seed,"square-query",sequence_index)` for query sampling, and independent `seed_for(split_seed,"square-base",sequence_index,quartet_index)` / `seed_for(split_seed,"square-groups",sequence_index,quartet_index)` for base and group draws. Iterate spans, eligible positions and p indices in ascending order. Draw one Uniform(0,1) per eligible position for each base and reveal iff U<=p; shuffle the sorted remaining positions with the group RNG. Use `random.Random` with each derived seed; query sampling is from sorted positions without replacement. Store realized inputs and all derived seeds.
- Per span select eight query indices uniformly without replacement before any masks. Use the same query set across five quartets. Keep all queries masked in all four states.
- For each base visibility probability p in [0.05, 0.25, 0.50, 0.75, 0.90], independently sample visible eligible positions. On the remaining masked eligible positions draw one deterministic random permutation; set group size g=min(13, floor(remaining/2)), and take disjoint first/second groups R1/R2 of size g. If g=0, keep the degenerate quartet rather than resampling based on outcomes.
- Construct x00/base, x10/base+R1, x01/base+R2, x11/base+both using gold tokens. Store all positions, group sizes, state IDs, actual mask counts and seeds. The groups are random token sets, not identified semantic evidence.
- For each query let e_ab=f_sparse(x_ab)-f_dense(x_ab). A_square=mean_ab(e_ab²); C_square=mean over edges (00,10),(00,01),(10,11),(01,11) of (e_v-e_u)². Average queries, then five quartets, then eight spans equally. Square-A uses A; Square-AC uses A+C. No extra mixed-interaction penalty.

Both Square arms use exactly the same 160 endpoints, equal nominal state-forward counts and identical input masks/queries. Their final weight masks may differ. Across bank families the state distributions differ, so comparisons with legacy AC or Multi are end-to-end screens, not isolated C ablations. The 160-state count matches legacy forward count, not necessarily query-count/statistical information.

### 4. Centered-logit vector readout with bounded memory

Use the existing legacy pair/query manifest byte-for-byte. For each query, F_X=z_X-mean_vocab(z_X), E=F_sparse-F_dense. Define A_vec=mean over endpoints, queries and vocabulary of E²; C_vec=mean over queries/vocabulary of (E_after-E_before)². Average pairs exactly as the legacy scalar objective. Vocab normalization is 1/V, and lambda=1. Neither hidden-state MSE nor top-k/probability-weighted approximation is an interchangeable fallback.

First implementation computes these metrics directly from raw emitted logits cached losslessly as FP32, with centering and score arithmetic in FP64 as required in §0.5. Cache only dense query logits in resumable per-state disk tensors; do not persist sparse vocabulary vectors for every probe. For a sparse pair, retain its before residual in CPU memory and stream after residual in chunks to compute A/C with FP64 accumulation. Commit one complete pair metric atomically, including teacher/state/query and candidate identities; if interrupted mid-pair, recompute that pair. Both A/AC metrics come from the same forward outputs. Teacher files are shared across probe workers and become readable only after completion.

CPU prepare estimates disk and RAM from actual query counts, vocab and dtype, including FP64 before-residual buffers and reduction chunks (upper bound for one bank is 160×256×126464×4 bytes, about 20.7 GB decimal). The runtime checks free disk, reserves storage before teacher writes, and records peak RAM/CUDA; CUDA bound remains 30 GiB. This is a real cost absent from scalar readouts. If the declared memory bound cannot be met, mark the arm failed; do not silently change readout/queries/precision to obtain a result. Exact head-Gram and hidden-only acceleration are deferred until numerical equivalence with the emitted-logit objective is demonstrated; they are not necessary to implement this first screen.

Fresh Vector diagnostic pairs use the eight clean diagnostic spans (indices 8–15), not the 40 existing noisy states (five mask rates each). Generate ten before states per span using the ten STORED calibration `p_mask` values in ascending timestep order; these are mask probabilities, not visibility probabilities. For state i=10*(sequence_index-8)+timestep_index, draw a length-256 FP32 CPU `torch.rand` with generator seed 20260928+i, mask where rand<p_mask, and fail rather than resample if fewer than two masks occur. Invoke the unchanged legacy `make_pairs` on the resulting ordered 80-state manifest with seed 20260929, retaining its original reveal count and remaining-masked query rule. Freeze inputs, mask/reveal seeds, ordered queries and gold IDs before any model work. Queries here are legacy post-reveal queries, not upfront Multi/Square queries; the product-coupling theorem is not claimed for this bank. They never guide ranking or search. Dense diagnostic readouts are shared by the two final vector models. This does not create an independent unseen-corpus claim.

### 5. Shared probe collection and final masks

For Square and Vector separately, reproduce the native Uniform physical mask and frozen activation ranking, then probe each block at 48% and 52%, holding other blocks at Uniform50. Record actual removed counts and all objective components. Marginal cost is (L52-L48)/(removed52-removed48), preserving negative values. Use existing average-rank mapping and exact quota DP without modifying implementation. Derive A/AC allocations on CPU from the same 64 conditions per family. Measure native Uniform on each new family calibration bank as an objective reference (one additional 160-state candidate per family), then measure each final mask on its calibration and diagnostic banks. Persist Square mode energies and Vector A/C components as defined in §0; never approximate final joint loss by summing probe costs. Objective measurements for a deduplicated mask can be reused only under the same bank/readout identity. Each family costs 64×160=10,240 sparse state forwards before teacher/final diagnostics; neither is a cheap consequence of mini100 alone.

Worker shards hold only the dense block weights they need to restore, using existing block-sharded runtime patterns. Full final candidate building uses original weights and selected row quotas, verifies all 224 matrices and exact count, and hashes the physical model before evaluation. Equal quota vectors imply equal masks only under the verified same rankings; cross-family deduplication requires actual physical identity.

### 6. A bounded block-tied exchange algorithm

Start from legacy AC row counts and physical mask. The unchanged legacy 80-pair scalar A+C is the sole acceptance objective. Preserve legacy row-quota rounding offsets rather than resetting each block to a rounded percentage.

All verified blocks have six 4096-input projections and one 12288-input ff_out. A block increment uses d weights per row on the six projections and 3d on ff_out. The removed-count change is exactly 53,248d weights per block. Freeze d=41, approximately 1.001%p in sparsity per block. A donor adds this increment and a receiver subtracts it, maintaining the exact global count. Validate this quantum from actual shapes; reject incompatible shapes rather than assuming them. Require affected projection sparsities within [0.45,0.55], allowing only the immutable integer-rounding excess already present at the anchor. Explicitly record anchor bounds per matrix.

Proposal selection: reuse original signed AC block costs to sort directed donor/receiver pairs by predicted improvement cost_receiver-cost_donor, descending, with deterministic block-ID tie breaks. At each round enumerate feasible proposals from the current incumbent, then take the first eight distinct physical masks. Do not use diagnostic/task scores. Evaluate each on all 80 pairs; objective-cache hits are keyed by physical mask and complete bank/readout identity. Use at most three rounds and at most 24 new candidate full-bank measurements. Accept the lowest measured loss only if below incumbent by epsilon=max(1e-6, 1e-5×abs(initial_loss)); set initial_loss=L0 from the first finite initial evaluation, freeze epsilon immediately, then compare the second initial evaluation against L0 before any proposal. Use L0 as the incumbent loss after that check; do not average selectively or choose a favorable repeat. Reject nonfinite measurements. On no improvement or no feasible proposals, terminate. Reaching the round cap is budget exhaustion, not convergence.

The initial loss is reevaluated twice as a reproducibility check; differences beyond epsilon fail preflight rather than widening tolerance after seeing candidates. Persist evaluated candidates and an atomic round decision (incumbent hash, selected candidate, measured delta, remaining budget). Resume after a crash must neither accept a move twice nor replenish the search budget. Distinguish cache hits from new evaluations in accounting. Maximum fresh search measurements: two initial checks plus 24 proposals, or 4,160 state forwards before diagnostics.

Measure scalar A/C on the frozen Vector diagnostic pairs from §4 for both the legacy AC anchor and the final Exchange mask. This is a separately labeled diagnostic bank; optimization remains on the unchanged original 80 calibration pairs. Use the legacy FP32 scalar routine on raw query logits. A validated dense raw-logit cache can supply that scalar without another transformer pass; otherwise schedule a scalar dense teacher on the same inputs, with exactly one cache producer. Reuse the anchor diagnostic if the final physical mask is unchanged. These diagnostic values never guide exchange selection.

Alternative postponed: common A-anchor A-continuation/AC-continuation/A-floor searches answer a different objective question. This screen changes only the solver from the existing AC endpoint; an Exchange win does not establish C's necessity under exchange. No beam, adaptive radius, gradients or projection-level moves are added.

### 7. Dependency scheduler and operational contract

Expose `run.sh prepare`, `validate`, `status [--json]`, `report`, `launch --gpus <ids> [--dry-run]`, and `stop`. A future launch command is shown in docs, not executed by preparation. Resume uses the same launch and unchanged scientific config. Device IDs are separate execution metadata; scientific seeds do not depend on assignment, worker count or job order.

Create a persisted DAG: verified imports; family teacher jobs; block-probe shards; CPU allocation; final mask/diagnostic jobs; document evaluation; final report. Original Multi/Short final jobs are immediately ready when imports validate. Schedule ready jobs fairly in round-robin family order, prioritizing resumed incomplete evaluations on ties. Partition pending block indices into configurable small shards (default four blocks), independent of number of GPUs. Keep GPU workers one per device; the tmux controller performs light CPU aggregation with thread caps. Shared artifact producers have a single owner and completion checks.

A top-level lease covers the screen and a compatible legacy lease prevents two owners of old outputs. Record worker process group IDs and descendant ownership; stop sends TERM, then KILL after a bounded grace period only for verified owned processes. Do not kill by broad filename patterns. The parent traps interrupts and stops children; any worker failure fails the current execution attempt and stops siblings, preserving completed independent work for resume. Research lifecycle is updated after children terminate. Detaching the app/SSH has no effect on tmux.

Status reads receipts without nvidia-smi. Show calibration/teacher/probe/search/generation counters independently, elapsed times, log locations and unknown ETA until same-stage measurements exist. Runtime launch can check only requested GPUs; no stale assumption that previous GPU0/2 availability still holds. CPU commands set CUDA_VISIBLE_DEVICES empty and cap BLAS/torch threads; no 8B CPU load.

### 8. Evaluation, reporting and cost

Reuse native evaluation helpers and exact frozen requests. Imported predictions must match document, prompt, target, token and protocol hashes as well as model identity; do not reproduce prompts with a changed few-shot RNG stream. New document writes are atomic. Cached exact same physical models reuse generation only, while bank-specific diagnostics remain distinct jobs.

Predeclare the seven contrasts in reporting spec, Holm family size seven, and raw paired gains/losses/p-values. For incomplete arms show common-ID subsets with descriptive statistics only. Even after completion, Holm controls the declared family of comparisons only; prior adaptive reuse of mini100 prevents a pristine confirmatory interpretation. Extra comparisons with legacy AC/Uniform are exploratory. Record scalar and vector objectives separately, as their raw magnitudes are not comparable. Include per-question predictions and correct counts; exact match is not a reasoning-quality annotation.

Nominal NEW family forward accounting before physical/input deduplication, failures or bounded smoke checks:

| Family | Teacher | Uniform calibration reference | Probes/search | Final calibration / diagnostics | State-forward total |
|---|---:|---:|---:|---:|---:|
| Square | 160 calibration +160 diagnostic | 160 | 64×160=10,240 | 2×(160+160)=640 | 11,360 |
| Vector | 160 calibration +160 diagnostic | 160 | 64×160=10,240 | 2×(160+160)=640 | 11,360 |
| Exchange | Existing calibration teacher reused; diagnostic teacher 160 if not shared | No new Uniform reference | <=(2+24)×160=4,160 | Anchor + final diagnostic <=320 | <=4,640 standalone; <=4,480 when diagnostic teacher shared |

The two initial Exchange calibration measurements are already included in 4,160. The final selected calibration score is an accepted/cached full-bank measurement, not an extra sweep. New-family Uniform and final-mask scores can deduplicate only with matching physical mask AND bank/readout identity. Vector teacher storage includes BOTH calibration and diagnostic banks; use actual per-state query counts (the 20.7 GB figure in §4 is per-bank upper bound, not a two-bank reservation). Legacy Multi work is imported/scheduled from actual missing receipts, never charged as wholly new. Bounded GPU smoke is at most 32 additional state forwards across the screen, with separate accounting; setup/model loading and failed/repeated work are also reported. These are state-evaluation counts, not batched kernel calls or wall-clock predictions.

Count teacher setup, probes/search, diagnostics and generation independently, with both incurred-this-run and reused computations. Account vector disk/RAM and failed attempts. Do not estimate runtime solely as remaining questions × prior generation time. Use actual same-stage worker throughput and critical-path readiness for ETA; unknown is preferable to unsupported speedup claims.

### 9. Research memory and planning/execution boundary

Before actual launch, the executing assistant checks both Obsidian status and Research-State, creates a running Experiments note with the AGENTS.md sections, verifies the write and supplies its path/receipt to the launch record. A shell-only launcher writes the same local sections and marks remote sync pending; it never claims an MCP write it did not perform. The assistant reconciles pending records through available MCP, verifying writes. Stop/failure/completion preserve distinct statuses; planning never sets running. No automatic follow-up or confirmation launch follows a report.

## Risks / Trade-offs

- New objective/calibration bank can improve its own loss and worsen mini100 → matched A controls, fixed task protocol, no success guarantee.
- Adaptive exchange overfits eight spans → bounded search, separate diagnostic output, reused mini100 explicitly development.
- Square groups confound content and mask count → label as reveal-response, not pure semantic interaction; keep A-only on identical states.
- Old code fingerprints make convenient refactors destructive to resume → separate wrapper and import receipts; fail on stale dependencies.
- Vector teacher storage and model-load duplication limit parallel speedup → preflight sizes, chunked readout, measured costs, bounded shards; no runtime promise.
- BF16 numerical variation across devices could move exchange decisions → record hardware, fixed seeds/precision, repeat initial objective before search, frozen tolerance.
- No exchange improvement can reflect the fixed shortlist/radius → report bounded negative result, not global optimality or objective failure.
- Multiple exploratory arms on reused mini100 can produce chance winners → fixed comparisons, paired reporting, defer confirmation to a separate request.

## Migration Plan

1. Add the new package and CPU tests without changing historical source/config/output bytes.
2. Run CPU prepare, validation and dry-run, inspect job/reuse/space estimates, and save hashes of protected files.
3. When the user requests execution (including a combined implementation-and-run request), choose then-available GPU IDs, record the running note, perform bounded GPU smoke checks, then run the same frozen manifest in tmux.
4. Stop/resume operates on owned jobs and committed receipts. Rollback removes or ignores only the new unstarted output root; historical artifacts remain intact. A started invalid configuration is retained as failed evidence and replaced by a new root after an explicit recorded correction.

## Open Questions

- Which GPUs will be available at launch, and what are actual per-stage throughput/peak memory? These are execution metadata to measure, not scientific settings to tune.
- Do existing artifacts all still pass hash validation at apply/launch time? If not, report exact missing/mismatched sources; do not infer validity from this planning snapshot.
