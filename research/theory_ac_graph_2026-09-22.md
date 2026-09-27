# Theory-led restructuring of scalar A+C: graph, Dirichlet, and design evidence

Date: 2026-09-22

Scope: literature review and objective redesign only. No GPU forward, model run, mask search, benchmark, or experiment was executed. The current project state was read from Obsidian immediately before this review; both sync status and Research/DLM-Pruning/Research-State.md returned valid responses. The current mini evidence is A55 and A+C61 on the 80-pair development setting; full held-out quality remains unavailable.

## State the existing objective as a graph quadratic form

For pair \(i\), let \(v=(i,0)\) and \(u=(i,1)\) be the two natural gold-reveal endpoints. Let \(f_D(v)\) and \(f_M(v)\) be dense and sparse gold-versus-rest log-odds, and define the scalar residual

\[
r_v=f_M(v)-f_D(v).
\]

The current objective is

\[
A(r)=\frac{1}{P}\sum_{i=1}^{P}\frac{r_{(i,0)}^2+r_{(i,1)}^2}{2},
\qquad
C(r)=\frac{1}{P}\sum_{i=1}^{P}
\left(r_{(i,1)}-r_{(i,0)}\right)^2.
\]

Let \(G_0=(V,E_0)\) contain exactly the 80 disjoint natural pair edges, with unit weight and incidence matrix \(B_0\). Then

\[
C(r)=\frac1P r^\top L_0r,\qquad L_0=B_0^\top B_0.
\]

Thus the current scalar A+C is already an endpoint \(\ell_2\) penalty plus a graph Dirichlet energy on a graph with one edge per pair. This identity is exact; calling \(C\) a graph penalty does not create a new method.

The question for this review is narrower: can a defensible graph or sampling theorem justify adding specific cross-pair edges or edge weights that express a context relation missing from the disjoint-edge graph? Because A already anchors every endpoint, connectivity cannot add missing endpoint information. It can only reweight or regularize relationships among endpoint residuals. Any quality claim still requires held-out DLM evaluation.

## Read-depth ledger

| Work | Primary source | Read depth and locators | What the paper actually establishes | Boundary for our pruning inference |
|---|---|---|---|---|
| Belkin, Niyogi, Sindhwani, **Manifold Regularization** (JMLR 2006) | [JMLR PDF](https://jmlr.org/papers/volume7/belkin06a/belkin06a.pdf) | Deep: Sec. 2 Eqs. (1)–(5), Theorem 2; Sec. 3 Theorem 7; Sec. 4 empirical graph objective and Eq. (13) | A loss plus ambient RKHS norm and intrinsic graph/manifold smoothness term; representer theorems under RKHS/operator assumptions | Supports an \(r^\top Lr\) construction when endpoint residuals are assumed smooth on a justified context graph. It does not prove that pruning residuals are smooth or that downstream quality follows. |
| Spielman and Srivastava, **Graph Sparsification by Effective Resistances** (STOC 2008) | [Author PDF](https://www.cs.cornell.edu/~abrahao/tdg/papers/p563.pdf) | Deep: Sec. 1 algorithm/Theorems 1–2; Sec. 2 incidence/Laplacian/effective resistance; Sec. 3 proof setup | Sampling edges proportional to \(w_eR_e\) gives a weighted sparsifier preserving every Laplacian quadratic form within \(1\pm\epsilon\), with \(O(n\log n/\epsilon^2)\) sampled edges | Can compress an expanded context graph while preserving the graph C term for every residual vector. It preserves the proxy, not the model or task quality. |
| Batson, Spielman, Srivastava, **Twice-Ramanujan Sparsifiers** (SIAM J. Comput. 2012) | [SIAM page](https://epubs.siam.org/doi/abs/10.1137/090772873), [PDF](https://www.cs.cmu.edu/~odonnell/hits09/batson-spielman-srivastava-twice-ramanujan-sparsifiers.pdf) | Targeted deep: Theorem 1.1 and rank-one/barrier construction statement | Every weighted graph has a deterministic weighted subgraph with \(O(n)\) edges and a spectral quadratic-form approximation; retained edges may receive new weights | A deterministic alternative for an expanded pair graph. The reweighted-edge theorem does not preserve equal pair counts or establish a better pruning mask. |
| Fuhr and Pesenson, **Poincaré and Plancherel–Polya Inequalities on Weighted Combinatorial Graphs** (SIAM J. Discrete Math. 2013) | [arXiv primary manuscript](https://arxiv.org/abs/1108.5637), [SIAM page](https://epubs.siam.org/doi/10.1137/120873674) | Targeted: Intro main results, Theorem 1.1/1.3, Sec. 3 Theorem 3.2, Sec. 4 Theorem 4.1 | Under weighted-graph geometry and Paley–Wiener/bandlimited assumptions, graph gradient energy controls norm/reconstruction and frame convergence | Gives a clean warning: a spectral/sampling guarantee needs connectedness and a low-frequency assumption. Neither is established for DLM residuals. |
| Anis, Gadde, Ortega, **Efficient Sampling Set Selection for Bandlimited Graph Signals Using Graph Spectral Proxies** (IEEE TSP 2016) | [Author-uploaded manuscript](https://www.researchgate.net/publication/282402877_Efficient_Sampling_Set_Selection_for_Bandlimited_Graph_Signals_Using_Graph_Spectral_Proxies), [DOI](https://doi.org/10.1109/TSP.2016.2546233) | Deep: Theorem 1/Corollary 1 uniqueness, Eqs. (14), (19)–(22), Theorem 2 spectral proxies, Sec. IV method | A sample set identifies a bandlimited graph signal when the sampled low-frequency eigenvector matrix has full column rank; spectral proxies avoid a full eigendecomposition and bound reconstruction error | Suggests a pair-bank design diagnostic, not a new loss. It applies only if DLM residuals are approximately low-frequency on a known graph. |
| Kiefer and Wolfowitz, **The Equivalence of Two Extremum Problems** (Canadian J. Math. 1960) | [Primary PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/B8B0626C11F52B0FD8C67C5D54BDDD43/S0008414X00010002a.pdf/equivalence_of_two_extremum_problems.pdf) | Targeted: full 4-page note, theorem pp. 364–365 | For a finite-dimensional regression design with nonsingular information matrix, D-optimality is equivalent to minimizing worst-case variance (G-optimality) | Could select calibration pairs by a predeclared finite feature map, but it requires a valid linearized feature model; it does not optimize a nonlinear sparse model directly. |
| Dereziński, Warmuth, Hsu, **Minimax Experimental Design** (COLT 2019) | [Primary PDF](https://www.stat.berkeley.edu/~mmahoney/pubs/derezinski19b_colt19.pdf) | Targeted: Sec. 2 leverage/inverse-score distributions, Theorem 10, Sec. 4 proof discussion | Combines volume sampling and leverage/inverse-score i.i.d. sampling to control worst-case regression MSE; requires a finite design matrix and matrix conditioning assumptions | A possible robust pair-bank sampler after defining response features. It changes calibration coverage, not the A+C semantic objective, and is not justified without a stable feature map. |
| Lu et al., **AlphaPruning** (NeurIPS 2024) | [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2024/file/10fc83943b4540a9524af6fc67a23fef-Paper-Conference.pdf) | Targeted: Sec. 2.2 ESD/HT metrics, Sec. 3.1–3.4, Eq. (4), Table 2, App. F | Uses PL Alpha Hill shape metrics of weight-matrix ESDs to map layer quality to layer sparsity; validates allocation empirically with Wanda/SparseGPT and LLaMA/OPT families | This is a practical allocation comparator and a useful separation of allocation from readout. It supplies no graph/Dirichlet theorem and no evidence that HT shape should replace DLM endpoint/response fidelity. |

The first four graph papers and Anis were read at theorem/method level. Kiefer–Wolfowitz was read in full because its equivalence theorem is short. Dereziński and AlphaPruning were read at the method/theorem and main-ablation level for applicability screening.

## Deep reading and exact transfer boundaries

### 1. Manifold regularization: a source for the Dirichlet term, not a pruning guarantee

Belkin et al. formulate the ambient regularized learner in Eq. (1) as empirical loss plus \(\gamma_A\|f\|_K^2\). Their empirical manifold version adds a graph term proportional to

\[
\frac{\gamma_I}{(l+u)^2} f^\top L f
\]

alongside the ambient RKHS term; the paper's Sec. 4 derives the corresponding labeled/unlabeled kernel objective. In the unsupervised/clustering construction, Eq. (13) is explicitly a kernel norm plus \(\sum_{i\sim j}(f(x_i)-f(x_j))^2\). Theorem 2 gives an expansion over labeled and unlabeled points, and Theorem 7 gives an analogous operator representer result under a bounded intrinsic operator.

The paper assumes that the graph or manifold geometry is relevant to the target function. For DLM pruning, the candidate signal is the dense–sparse residual \(r\), and the graph nodes are partially revealed endpoint states. The exact borrowed object is therefore the quadratic form \(r^\top Lr\), not the representer theorem. The existing bank cannot define a new cross-edge by taking the same query and the same visible-token set: deterministic inputs are identical or duplicate. A valid extension must explicitly build a mask-lattice. For scored target position \(t_q\), define distinct states \(x_{q,S}\) with \(t_q\notin S\), and add only \(S\leftrightarrow S\cup\{j\}\) for a non-target context position \(j\notin S\). This is an artificial smoothness prior over one-token context reveals, not a consequence of the theorem. A hidden-state similarity graph remains only a separate diagnostic.

The strongest contrary case is a legitimate high-frequency residual: two semantically different contexts may be close in hidden representation or share a reveal count while the pruning error changes sharply between them. The graph penalty then oversmooths the very interaction A+C should expose. If no valid distinct mask-lattice edges exist, \(L\) is block diagonal with the current two-node components and the construction collapses exactly to the existing C.

### 2. Poincaré and spectral gap: what connectivity would and would not buy

For a connected weighted graph with Laplacian \(L\) and vertex-weighted mean \(\bar r\), the standard finite-graph Poincaré inequality has the form

\[
\|r-\bar r\mathbf 1\|_2^2
\leq \lambda_2(L)^{-1} r^\top Lr,
\]

up to the chosen degree/measure normalization. The spectral gap \(\lambda_2>0\) is the condition that turns edge differences into a global control of nonconstant residual variation. Fuhr and Pesenson's Theorem 3.2 makes the corresponding graph-sampling restriction explicit for Paley–Wiener functions: the graph is finite and connected, the signal is bandlimited, and the sampling partition must satisfy their admissibility conditions. Their Theorem 4.1 then gives geometric convergence of a frame reconstruction algorithm under the stated sampling bound.

The pruning inference is limited. A's endpoint term already controls the constant component and every endpoint magnitude. A connected graph would only add a structured prior over nonconstant cross-pair residuals. The current graph of disjoint edges has multiple components and global \(\lambda_2=0\); this is why no global Poincaré bound can be claimed for the current C. Connecting components could make \(\lambda_2>0\), but that bound would be a statement about the residual's smoothness on the chosen graph, not a statement about NELBO, capability, or mask quality. There is also no established bandlimited assumption for DLM pruning residuals. A low graph-frequency residual is a hypothesis to test, not a fact to put into the method name.

### 3. Effective resistance and spectral sparsification: an exact way to thin a graph proxy

Spielman and Srivastava write \(L=B^\top W B\), so every graph energy is a sum of weighted edge differences. Their Theorem 1 states that sampling \(q=O(n\log n/\epsilon^2)\) edges with probability proportional to \(w_eR_e\), and reweighting sampled edges, gives with constant probability

\[
(1-\epsilon)x^\top Lx
\leq x^\top \widetilde Lx
\leq (1+\epsilon)x^\top Lx
\quad\text{for every }x.
\]

Their Sec. 1 also gives the effective-resistance algorithm and Theorem 2's approximate resistance queries. The proof uses the incidence representation and the fact that \(\sum_e w_eR_e=n-1\) for a connected graph. This is a true all-vectors quadratic-form guarantee, stronger than preserving only cuts.

BSS strengthens the edge-count story. Its Theorem 1.1 gives a deterministic weighted subgraph with at most \(\lceil d(n-1)\rceil\) edges and a stated spectral approximation factor; the theorem permits new edge weights. The paper's barrier/rank-one construction is an existence and algorithm result for graph quadratic forms, not a task-preservation result.

A feasible static adaptation is: first define a larger candidate context graph \(G\) from eligible shared masked-transition edges, then form a sparsified calibration graph \(\widetilde G\) using resistance sampling or BSS. Evaluate

\[
J_{\widetilde G}(M)=A(M)+\lambda\,r(M)^\top L_{\widetilde G}r(M)
\]

with inverse-probability or sparsifier weights. For every fixed sparse mask residual \(r\), the theorem bounds the C-proxy distortion. It does not say which mask is better, and it does not justify sampling the current 80 disjoint edges: that graph has no redundant connectivity to exploit. BSS's reweighted edges also conflict with a strict equal-count interpretation unless weighted calibration edges are explicitly allowed.

### 4. Graph spectral sampling and optimal design: bank coverage diagnostics

Anis, Gadde, and Ortega's Theorem 1 states that a sampling set \(S\) uniquely identifies a graph-bandlimited signal when the graph-bandlimited subspace has no nonzero signal supported entirely outside \(S\). Their Corollary 1 expresses this as full column rank of the sampled low-frequency eigenvector matrix; Eq. (14) gives the least-squares reconstruction. Eq. (19) bounds reconstruction error by a subspace-angle factor times the high-frequency/model-mismatch residual. Their Definition 2 and Theorem 2 replace an expensive bandwidth calculation with spectral proxies and give a bound for variational reconstruction when the signal is approximately bandlimited.

This suggests a diagnostic for the 80-pair bank: choose a provisional context graph and a low-frequency basis \(U_K\), then measure the smallest singular value of the sampled endpoint matrix or a spectral-proxy score. A bank with poor rank may leave entire smooth residual directions unobserved. This is a coverage test, not evidence that the missing directions matter to pruning. If \(r\) is not approximately bandlimited, the theorem's reconstruction bound is inapplicable.

Kiefer–Wolfowitz gives a complementary finite-feature design result. For feature vector \(\phi(x)\) and design measure \(\xi\), the information matrix is \(M(\xi)=\int \phi(x)\phi(x)^\top d\xi(x)\), and the variance function is \(d(x;\xi)=\phi(x)^\top M(\xi)^{-1}\phi(x)\). Under compactness, linear independence, and nonsingularity, their theorem makes determinant maximization equivalent to minimizing \(\max_x d(x;\xi)\). A DLM adaptation would need a predeclared finite feature map, such as low graph eigenvectors or a fixed linearized response feature, and would select calibration pairs to reduce worst-case feature uncertainty. The theorem does not apply to an arbitrary nonlinear mask-to-logit map.

Dereziński et al. make leverage and inverse-score sampling practical for minimax regression, but their finite design matrix and MSE model remain assumptions. They are useful for a later calibration-bank study, not a reason to change A+C now.

### 5. AlphaPruning as an allocation comparator

AlphaPruning defines the weight-matrix ESD of \(X_i=W_i^\top W_i\), fits a power-law tail \(p(\lambda)\propto\lambda^{-\alpha}\) in Eq. (2), and uses the PL Alpha Hill estimator in Eq. (3). Section 3.3 maps layer quality values \(q_i\) to sparsities with a normalized linear mapping in Eq. (4), preserving a target global sparsity. Table 2 evaluates LLaMA/LLaMA-2 at 70% sparsity with Magnitude, Wanda, and SparseGPT; Appendix F compares per-matrix, per-block, and mixed allocation.

The only justified borrowing here is experimental separation: a graph-structured A+C readout and a layer/budget allocator should be compared as separate factors. HT-SR shape is not a substitute theorem for DLM endpoint fidelity, and the AlphaPruning correlations are empirical model-family results. Root's separate HT-SR review should handle any deeper use of this prior.

## Ranked practical candidates

### Candidate 1 — conditional mask-lattice Dirichlet extension

This candidate is not available from the current 80 endpoint bank alone. First construct distinct mask-lattice states. For each scored target position \(t_q\), let \(S\) be a set of visible context positions with \(t_q\notin S\), and define \(x_{q,S}\) by revealing exactly \(S\) while keeping \(t_q\) masked. Add a context edge only when

\[
(q,S)\longleftrightarrow(q,S\cup\{j\}),\qquad j\neq t_q,\ j\notin S.
\]

The two inputs then differ by exactly one non-target context reveal; same-query/same-visible-set duplicates are excluded. This is an explicitly constructed mask-lattice prior, not a relation already present in the natural gold-reveal bank. Keep natural pair edges separately and use fixed, predeclared edge weights.

With incidence \(B\), fixed edge-weight matrix \(W\), and \(L=B^\top WB\), use

\[
J_{\mathrm{Dir}}(M)
=A(M)+\lambda\,
\frac{2r(M)^\top Lr(M)}{\operatorname{tr}(L)}.
\]

The factor 2 is necessary. For \(P\) unit natural edges, \(\operatorname{tr}(L)=2P\), so \(2r^\top Lr/\operatorname{tr}(L)=r^\top Lr/P=C(r)\). Thus the natural-edge-only case with \(\lambda=1\) is exactly old A+C. More generally, the normalization is graph energy divided by total edge weight.

The added term penalizes inconsistent pruning residuals across one-token context transitions while A anchors every endpoint. It becomes a genuine restructuring only when \(E_{\mathrm{lat}}\) contains valid distinct lattice neighbors. If the bank cannot supply or explicitly construct them, this candidate must not be ranked as strongest. The smoothness assumption is artificial and requires an independent test. If the lattice graph is semantically wrong, it can oversmooth a valid high-frequency residual; if it is empty, the method collapses exactly to A+C.

### Candidate 2 — resistance-sparsified context graph

Use the same \(G=(V,E_{\mathrm{nat}}\cup E_{\mathrm{lat}})\), but replace the full mask-lattice edge set by a spectral sparsifier. Sample eligible edges with probability proportional to \(w_eR_e\) and reweight them, or use deterministic BSS weights, giving

\[
J_{\mathrm{spec}}(M)
=A(M)+\lambda\,r(M)^\top L_{\widetilde G}r(M).
\]

Spielman–Srivastava then provides an all-residual-vector \((1\pm\epsilon)\) bound between the sparsified and full graph C terms, subject to the connected weighted-graph and sampling assumptions. This is useful if the exact-transition graph is expanded far beyond 80 pairs and its objective evaluation is expensive. It is not a quality-improving objective by itself: a mask that minimizes the sparsified proxy need not have better held-out NELBO or downstream behavior. With the current disjoint 80-edge bank, it collapses to unnecessary bookkeeping and should not be the main method.

### Candidate 3 — spectral/G-optimal pair-bank design

Treat the graph low-frequency basis or another fixed feature map \(\phi(v)\) as the design feature. Select endpoint/pair samples so the sampled feature matrix has full rank and low worst-case leverage, using Anis-style spectral proxies or a Kiefer–Wolfowitz/Dereziński design criterion. Keep A+C unchanged after the bank is selected.

This candidate addresses calibration coverage rather than the objective. It is useful only if the residuals are empirically close to the chosen low-frequency/linearized subspace. Otherwise it can select a mathematically well-conditioned bank that is irrelevant to the task. It should be a later bank factor, not the main A+C method.

## Strongest contrary cases and collapse tests

1. **Disjoint-edge and normalization collapse.** With only \(P\) unit natural edges, \(\operatorname{tr}L=2P\) and \(2r^\top Lr/\operatorname{tr}L=C(r)\) exactly. Any report must show the number and construction of distinct \(E_{\mathrm{lat}}\) neighbors; graph regularization alone is not a contribution.
2. **A already anchors all endpoints.** Poincaré or spectral gap does not supply endpoint information absent from the current objective. It only imposes a prior over cross-edge differences.
3. **Wrong smoothness prior.** A real pruning error may change sharply across a semantically meaningful reveal. A graph penalty can hide this error and improve calibration C while harming held-out quality.
4. **Disconnected graph.** If \(G\) remains disconnected, \(\lambda_2(L)=0\) globally. Component-wise Poincaré bounds require a component anchor or a separate mean for each component.
5. **Bandlimited assumption failure.** Graph sampling theorems require a known low-frequency subspace or bounded high-frequency residual. The DLM residual is not known to satisfy this.
6. **Spectral guarantee scope.** Effective-resistance/BSS guarantees preserve \(r^\top Lr\) for every fixed vector \(r\). They do not preserve dense-versus-sparse logits, NELBO, capability, or the optimizer's selected mask.
7. **Fixed-budget incompatibility.** Sparsifier theorems allow weighted edges and usually random sample counts or \(O(n)\) edge counts. The pruning problem has an exact Wanda rank/support budget; these are separate budgets.
8. **Leakage.** Building \(E_{\mathrm{lat}}\) or weights from sparse residuals makes the objective self-referential. Edge eligibility and weights must be fixed from dense context metadata before mask selection.
9. **Small-bank economics.** With 80 natural pairs and 160 endpoint evaluations, graph construction or resistance computation is unlikely to repay itself. The candidate is only plausible if the bank is expanded or calibration cost becomes limiting.

## Minimal future comparison, without running it here

Use the same frozen Wanda ranks, 32-layer allocation, exact global budget, natural pair bank, and mask-search backend for:

- scalar A+C baseline (\(E_{\mathrm{nat}}\) only);
- \(J_{\mathrm{Dir}}\) with \(E_{\mathrm{lat}}\) fixed before scoring;
- \(J_{\mathrm{Dir}}\) with \(\lambda=0\) and a shuffled-edge/null construction that preserves endpoint degrees;
- optional resistance-sparsified \(J_{\mathrm{spec}}\) only when the expanded graph is large enough to justify it.

Freeze masks before evaluating disjoint pair documents, full-vocabulary dense/sparse error, held-out NELBO, and downstream capability. Report the graph edge count, component count, \(\lambda_2\) per component, normalized Dirichlet energy, and the correlation between old C and new cross-context energy. A win in graph energy without independent quality improvement is a proxy win only. No GPU/model run is authorized or scheduled by this note.

## Decision

Do not currently rank Candidate 1 as strongest. First construct and audit the explicit mask-lattice neighbors; only then is a bounded test justified. Candidate 2 is a computational proxy-preservation tool; Candidate 3 is a calibration-bank design factor. Do not claim that graph connectivity, Poincaré inequalities, spectral sparsification, or optimal-design theorems guarantee pruning quality. If valid distinct lattice neighbors cannot be constructed, keep the current scalar A+C and record that the graph proposal collapsed to baseline.

## Primary URLs

- [Belkin, Niyogi, Sindhwani (2006), Manifold Regularization](https://jmlr.org/papers/volume7/belkin06a/belkin06a.pdf)
- [Spielman and Srivastava (2008), Graph Sparsification by Effective Resistances](https://www.cs.cornell.edu/~abrahao/tdg/papers/p563.pdf)
- [Batson, Spielman, Srivastava (2012), Twice-Ramanujan Sparsifiers](https://epubs.siam.org/doi/abs/10.1137/090772873)
- [Fuhr and Pesenson (2013), Poincaré and Plancherel–Polya Inequalities](https://arxiv.org/abs/1108.5637)
- [Anis, Gadde, Ortega (2016), Efficient Sampling Set Selection](https://doi.org/10.1109/TSP.2016.2546233)
- [Kiefer and Wolfowitz (1960), The Equivalence of Two Extremum Problems](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/B8B0626C11F52B0FD8C67C5D54BDDD43/S0008414X00010002a.pdf/equivalence_of_two_extremum_problems.pdf)
- [Dereziński, Warmuth, Hsu (2019), Minimax Experimental Design](https://www.stat.berkeley.edu/~mmahoney/pubs/derezinski19b_colt19.pdf)
- [Lu et al. (2024), AlphaPruning](https://papers.nips.cc/paper_files/paper/2024/file/10fc83943b4540a9524af6fc67a23fef-Paper-Conference.pdf)
