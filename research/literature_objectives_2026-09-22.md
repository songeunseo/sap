# Literature-derived objectives for DLM pruning A+C

Date: 2026-09-22  
Scope: literature review and objective design only. No GPU forward, new mask, benchmark, NELBO run, or experiment was started.

## Verdict first

The strongest objective candidate from the requested literature is a **DKD-style decoupled target/non-target context-response objective**. Keep the target-versus-rest channel (A), add a conditional non-target endpoint anchor, and add one context-response term over the same alternatives. This is one coherent decomposition; it does not stack the prior centered-logit response, Fisher weighting, tail vectors, or Jacobian terms.

For natural nested pair (i), let \(e\in\{-,+\}\) denote the less/more revealed state, \(g_i\) the gold token, \(W_i=\mathcal V\setminus\{g_i\}\), and \(p^e_{X,i,w}\) the temperature-scaled probability for dense \(X=D\) or sparse \(X=M\). Define the DKD conditional non-target distribution

\[
q^e_{X,i}(w)=\frac{p^e_{X,i,w}}{1-p^e_{X,i,g_i}},\qquad w\in W_i.
\]

Use the endpoint non-target anchor

\[
L_{\mathrm{NCKD,end}}(M)
=\frac{1}{2P}\sum_{i=1}^P\sum_{e\in\{-,+\}}
\mathrm{KL}\!\left(q^e_{D,i}\,\middle\|\,q^e_{M,i}\right),
\]

and the DLM-specific context-response term

\[
C_q(M)=\frac1P\sum_{i=1}^P\frac1{|W_i|}
\left\|
\left(q^+_{M,i}-q^-_{M,i}\right)
-\left(q^+_{D,i}-q^-_{D,i}\right)
\right\|_2^2. \tag{Cq}
\]

The single candidate is

\[
\mathcal J_{\mathrm{pairDKD}}(M)
=A_{\mathrm{bin}}(M)+\beta L_{\mathrm{NCKD,end}}(M)+\gamma C_q(M), \tag{pairDKD}
\]

where \(A_{\mathrm{bin}}\) is the target-versus-rest binary term (use the existing A implementation if it is its binary log-odds equivalent), and \(\beta,\gamma\) are fixed before mask selection. The endpoint NCKD term is deliberately not multiplied by the teacher-confidence factor \((1-p^e_{D,i,g_i})\); this fixed weighting is the DKD decoupling that prevents the target channel from suppressing the conditional alternative channel. The endpoint \(L_{\mathrm{NCKD,end}}\) is essential: a constant wrong-token error at both endpoints gives \(C_q=0\), but the endpoint anchor detects it. A response-only term would therefore be underconstrained.

This is more specific than full-vocabulary endpoint KL. Full KL mixes target confidence and alternatives and is retained as a same-bank control/evaluation metric; \(A_{\mathrm{bin}}\) and \(L_{\mathrm{NCKD,end}}\) expose those channels separately, while \(C_q\) asks whether the alternatives react to added context in the same direction and magnitude. It is also different from a centered-logit response: \(q\) removes the gold-token mass before comparing alternatives and imposes their probability competition. It is not Pearson-only, so response magnitude is retained.

Use a minimal controlled comparison before claiming value:

1. A-only: the current endpoint target channel, with the same support family, pair bank, exact sparsity, and mask search budget.
2. DKD endpoint control: \(A_{\mathrm{bin}}+\beta L_{\mathrm{NCKD,end}}\).
3. The candidate: \(A_{\mathrm{bin}}+\beta L_{\mathrm{NCKD,end}}+\gamma C_q\).

Keep \(\beta,\gamma\), temperature, \(\varepsilon\) if used for numerical flooring, and all pair-bank rules fixed before evaluation. The decisive comparison is endpoint control versus the candidate; A-only diagnoses whether the DKD endpoint anchor itself accounts for the change. Full-vocabulary KL, held-out NELBO, and downstream/capability evaluation must be reported on disjoint documents. The primary quality check is held-out downstream/capability behavior; NELBO is complementary evidence, not a required win.

The response term has a concrete blind-spot counterexample. Let a dense pair have \(p_D^-=(.4,.5,.1)\), \(p_D^+=(.4,.1,.5)\), and let the sparse pair be \(p_M^-=p_M^+=(.4,.5,.1)\), with the first coordinate gold. Existing scalar gold-vs-rest A and scalar C can both be zero. A response-only conditional term also vanishes if the sparse and dense wrong distributions are each constant across endpoints but differ by a fixed offset; \(L_{\mathrm{NCKD,end}}\) catches that endpoint error. In the shown switching example, \(C_q\) detects the missed wrong-token response. This is an algebraic counterexample, not a measured frequency.

The proposal is an empirical objective candidate, not a new distillation theorem or a claim that DKD/RKD relations transfer automatically to a frozen LLaDA mask. Its possible contribution is the directed, same-query masked-context response result under fixed support and budget. No experiment is scheduled by this note.

## Why this follows from the literature

### Decoupled Knowledge Distillation (CVPR 2022)

Zhao et al. split the teacher/student classification output into a target-class binary channel and a conditional non-target channel. In Sec. 3.1, their Eq. (5) decomposes the classical KD loss; Eq. (6) writes it as

\[
\mathrm{KD}=\mathrm{TCKD}+(1-p_t^{\mathcal T})\mathrm{NCKD},
\]

and Eq. (7) proposes

\[
\mathrm{DKD}=\alpha\,\mathrm{TCKD}+\beta\,\mathrm{NCKD}.
\]

TCKD is the KL between ([p_t,1-p_t]) for target versus all non-target classes. NCKD is the KL between the conditional distributions over non-target classes. The paper’s design motivation is precise: classical KD couples NCKD to teacher confidence, suppressing it on well-predicted examples, and couples the two contributions so their importance cannot be controlled independently. The pseudo-code in Algorithm 1 computes `tckd` from target/non-target binary probabilities and `nckd` from the softmax over non-target logits.

The main ablations are in Sec. 3.2. Table 1 evaluates vanilla training, KD, single TCKD, and single NCKD on CIFAR-100 with ResNet, WideResNet, and ShuffleNet teacher/student pairs, using loss weight 1.0. NCKD alone is often comparable to or better than KD; the authors report a +1.76% versus +1.13% example for ResNet8×4. Table 2 uses stronger augmentation, Table 3 symmetric label noise, and Table 4 ImageNet ResNet-34 teacher → ResNet-18 student: TCKD changes top-1 from 70.71 to 71.03. Table 5 splits CIFAR-100 samples by teacher target confidence for ResNet32×4 → ResNet8×4; using NCKD on the top-50% confidence samples gives 74.23 versus 73.96 on the lower-confidence half. Table 6 reports five-trial CIFAR-100 results; DKD exceeds classical KD by +1.03 to +2.99 points across the listed teacher/student pairs.

**Our separation.** Current (A) is close to TCKD: it preserves the gold-vs-rest channel. Current scalar (C) is not NCKD: it neither anchors the conditional non-target endpoint distributions nor measures their context response. The direct lesson is to make both channels explicit. The candidate is therefore not a rename of old scalar A+C and is not full-vocabulary KL; it is a DKD-inspired decomposition plus a DLM pair response. DKD itself is a trained image classifier distillation result; it does not establish that this objective improves a frozen LLaDA mask.

Primary sources: [CVPR open-access page](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html), [arXiv full text](https://arxiv.org/abs/2203.08679).

### Relational Knowledge Distillation (CVPR 2019)

Park et al. argue in Sec. 1 that knowledge can be represented by relations among examples rather than by each output independently. Sec. 3.2 gives the general relational objective (Eq. (4)):

\[
\mathcal L_{\mathrm{RKD}}
=\sum_{(x_1,\ldots,x_n)}
l\big(\psi(t_1,\ldots,t_n),\psi(s_1,\ldots,s_n)\big).
\]

Their distance-wise potential (Eqs. (5)--(6)) is

\[
\psi_D(t_i,t_j)=\frac{\|t_i-t_j\|_2}{\mu},
\qquad
\mu=\operatorname{mean}_{(i,j)}\|t_i-t_j\|_2,
\]

and their angle-wise potential (Eqs. (9)--(10)) is

\[
\psi_A(t_i,t_j,t_k)=
\left\langle
\frac{t_i-t_j}{\|t_i-t_j\|_2},
\frac{t_k-t_j}{\|t_k-t_j\|_2}
\right\rangle,
\]

with Huber loss between teacher and student potentials. The normalization and Huber loss are intended to avoid sensitivity to representation scale and outliers. Their Sec. 3.2.4 warns that relational loss alone is unsuitable where individual output values are essential, and should be combined with task or individual-output losses in that case. That warning maps directly to our need to retain (A) alongside the endpoint anchor and \(C_q\).

The primary metric-learning experiment uses CUB-200-2011 and Cars 196, a ResNet50-512 teacher, and ResNet18 students with 16/32/64/128-dimensional embeddings. Table 1 shows the RKD variants outperforming triplet baselines and often DarkRank; on Cars 196, ResNet18-128 without (\ell_2) normalization reaches 82.50 Recall@1 versus 77.17 for the teacher. Table 2 self-distills identical ResNet50-512 models: CUB 61.24 → 65.68 at generation 1 and Cars 77.17 → 85.65, with no further gain after generation 1. Tables 3--5 cover state-of-the-art metric learning, classification, and few-shot transfer. The paper also reports that RKD can be combined with task loss, with the generic form (\mathcal L_{task}+\lambda_{KD}\mathcal L_{KD}).

**Our separation.** \(C_q\) is a relation potential over two DLM context states, but its relation is defined over DKD-style conditional alternatives, not image embeddings. It borrows the relational unit and the explicit retention of individual target information; it does not claim that RKD's angle loss or all-triplet computation transfers to DLM pruning. Since it is a direct vector difference, it remains O(|V|) and does not require a (V\times V) matrix.

Primary sources: [CVPR open-access page](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html), [paper PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf), [arXiv](https://arxiv.org/abs/1904.05068).

### DIST: Knowledge Distillation from A Stronger Teacher (NeurIPS 2022)

Huang et al. study a failure mode of exact KL matching when a stronger teacher and weaker student have a large prediction discrepancy. Sec. 3.1 replaces exact output equality with a relation-preserving match. Their Eqs. (6)--(8) use Pearson distance (d_p(u,v)=1-\rho_p(u,v)) and row-wise correlation of the prediction vector (inter-class relation). Sec. 3.2 adds column-wise correlation across instances (intra-class relation), Eq. (9), and Eq. (10) gives

\[
\mathcal L_{tr}=\alpha\mathcal L_{cls}+\beta\mathcal L_{inter}+\gamma\mathcal L_{intra}.
\]

The motivation is that inference often depends more on relative preferences than exact probability values. This is useful evidence for relational response preservation, but Pearson correlation is scale/shift invariant; using correlation alone would miss a confidence or margin change important to a DLM decoder. \(C_q\) therefore uses an absolute conditional-probability response difference and retains (A) for the target confidence channel.

The paper’s Table 1 compares training strategies A1/B1/B2/B3. Table 2 evaluates ImageNet ResNet-18 students from ResNet-34 and ResNet-50 teachers; DIST reaches 72.07 versus 70.66 for KD on the ResNet-34 → ResNet-18 pair. Table 3 varies teacher size from ResNet-34 to ResNet-152; for ResNet-18, DIST improves over KD by 0.77--1.12 points. Table 4 tests stronger training strategies and dissimilar architectures (ResNet-50 teacher to ResNet-18, ResNet-34, MobileNetV2, EfficientNet-B0; Swin-L to ResNet-50/Swin-T). Table 8 is the key ablation: ImageNet ResNet-34 → ResNet-18 gives KD 71.21, inter-only 71.63, intra-only 71.55, and both 72.07; using KL rather than Pearson for the intra relation gives 71.62. Table 9 shows the effect persists without classification loss (DIST 70.65 versus KD 68.12).

**Our separation.** DIST supports measuring a relation over multiple states, but its task is learned student training and its rank/correlation invariance is too permissive for static pruning. \(C_q\) keeps the DLM finite context transition, uses a scale-sensitive conditional-probability difference, and leaves the endpoint target channel explicit.

Primary source: [NeurIPS 2022 paper PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/da669dfd3c36c93905a17ddba01eef06-Paper-Conference.pdf).

### Sobolev Training for Neural Networks (NeurIPS 2017)

Czarnecki et al. add target derivatives to output matching. Sec. 2 Eq. (1) is

\[
\sum_i\left[
\ell(m(x_i),f(x_i))+
\sum_{j=1}^K\ell_j(D_x^jm(x_i),D_x^jf(x_i))
\right].
\]

When the full Jacobian/Hessian is expensive, Eq. (2) uses random unit-vector projections of the derivative tensors. Sec. 4.2 distills Atari policies from A3C teachers: a three-convolution/two-fully-connected teacher is distilled into a two-convolution/smaller-fully-connected student. The paper reports lower held-out action error and KL on Pong, Breakout, and Space Invaders, especially in low-data settings. Sec. 4.3 compares synthetic-gradient variants on CIFAR-10 and ImageNet; CIFAR-10 reaches 93.5% with Sobolev synthetic gradients versus 93.2% for a regular critic, and the four-split ResNet-50 ImageNet result is 87.4% top-5 versus 86.9% for a regular critic. The artificial regression section reports improvement on six of seven functions, while Ackley is an explicit low-data/high-frequency failure case.

**Our separation.** A natural pair finite difference is a DLM context-response analogue of a derivative, but it is not an input Jacobian and has no Sobolev approximation guarantee. \(C_q\) must therefore be tested as a finite relation objective; it should not be sold as derivative matching or combined with a Jacobian term in the first comparison.

Primary source: [NeurIPS 2017 paper](https://proceedings.neurips.cc/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf).

### 2ndMatch (CVPR 2026)

Zheng and Shlizerman fine-tune pruned image diffusion models with output/noise prediction, KD, and a second-order Jacobian matching term inspired by finite-time Lyapunov exponents. The objective is reported as

\[
L_{total}=\lambda_{NP}L_{NP}+\lambda_{KD}L_{KD}+\lambda_{Jac}L_{2nd-Jac},
\]

where the added term matches dense/pruned (J^\top J) sensitivity using random projections. Their Table 4 is a decisive ablation on CIFAR-10: NP gives FID 5.29, NP+KD 5.05, adding first-order Jacobian matching worsens it to 5.14, while the second-order term improves to 4.58; FTLE moves from 0.413/0.418 to 0.408, near the dense 0.405. Table 6 at 44% pruning shows 2ndMatch improves every listed pruning method, including Diff-Pruning 5.05 → 4.58. Table 5 varies pruning ratios and U-Net/U-ViT architectures; e.g. U-Net at 70% pruning is 9.33 → 7.10 and U-ViT at 80% is 6.68 → 5.01.

**Our separation.** This establishes that a sensitivity target can add to output matching in image diffusion after weight updates, but it does not validate a fixed-weight, discrete DLM context-pair allocator. The negative first-order ablation also argues against adding generic (J^\top J) or Sobolev terms to \(C_q\) without a separate controlled study.

Primary source: [CVPR 2026 open-access entry](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html), [paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.pdf).

## Additional new pruning precedents

These were not in the prior local list and constrain the interpretation of any objective improvement.

### Shin et al., EMNLP 2024

Shin et al., “Rethinking Pruning Large Language Models: Benefits and Pitfalls of Reconstruction Error Minimization,” Sec. 2 Eq. (1), formulate post-training pruning as

\[
\min_{w,m}\|f(\bar w;D)-f(m\odot w;D)\|_2^2
\quad\text{s.t. }\|m\|_0\le k.
\]

They then enlarge the reconstruction unit with block reconstruction (Eq. (2)), global propagation, and cross-block reconstruction. On LLaMA-7B and OPT-125M at unstructured 50% sparsity, using 256 C4 calibration sequences of 1024 tokens and 10 Adam epochs, BR/GP/CR reduce final reconstruction error by 87--94% relative to layer reconstruction. However, Table 1 shows that CR can increase perplexity and reduce zero-shot accuracy despite its lower reconstruction error; for Wanda LLaMA-7B, BR+GP has mean accuracy 56.22 while BR+GP+CR falls to 55.94. Table 2 shows calibration/test error divergence for LLaMA-7B (Wanda CR 0.51 calibration versus 2.23 test for no CR and 0.38 versus 2.48 with CR). Figure 4 shows self-generated calibration data improves test error and perplexity. The authors explicitly limit the study to LLaMA-7B/OPT-125M and identify overfitting as more severe for the larger model.

**Implication.** An A or \(C_q\) improvement on the same pair bank is an in-calibration result. The candidate needs disjoint documents and held-out downstream/capability evaluation; WikiText NELBO is complementary. A hard A floor is not a guarantee of downstream safety.

Primary source: [ACL Anthology page](https://aclanthology.org/2024.emnlp-main.68/), [authoritative PDF](https://aclanthology.org/2024.emnlp-main.68.pdf).

### LiPRA, Neurocomputing 2026

Fang et al., “LiPRA: Lightweight pruning rate allocation for LLMs via global sensitivity measurement,” use budget-preserving perturbations of module pruning rates, a continuous local surrogate, and a KKT-derived sensitivity-to-rate mapping. The publisher abstract reports evaluation on LLaMA-1/2/3, OPT, and DeepSeek-16B-MoE and a 2 percentage-point perturbation scale. The method allocates at block level and then refines sublayers; it reports gains over OWL and DSA on LLaMA-1/2. The full paper was not available in the accessible primary result during this review, so no unverified section/equation/table claim is made here.

**Implication.** A new exact-budget search or sensitivity-to-rate mapping is already crowded. Any contribution must come from the DLM masked-context non-target relation and its independent quality value, not from KKT or rate allocation alone.

Primary source: [publisher article and abstract](https://www.sciencedirect.com/science/article/pii/S092523122602151X), [DOI](https://doi.org/10.1016/j.neucom.2026.134753).

## Counterexample and control

The current scalar readout can be unchanged while the wrong-token structure changes. Let a dense pair’s probabilities be

\[
p_D^-=(.4,.5,.1),\qquad p_D^+=(.4,.1,.5),
\]

and let the sparse pair be (p_M^-=p_M^+=(.4,.5,.1)), with the first coordinate as gold. Gold-vs-rest log-odds is unchanged at both endpoints, so the scalar endpoint errors are zero; the scalar response C also records zero, even though the dense model switches the two wrong-token preferences and the sparse model does not. The \(C_q\) term detects the change in the wrong-token response relation, while \(L_{\mathrm{NCKD,end}}\) catches endpoint-only non-target errors. This is a constructed algebraic counterexample, not a measured frequency.

The A/B control is mandatory:

- **A:** current endpoint-only (A(M)), same support family, pair bank, hard-mask search budget, and exact global sparsity.
- **B:** endpoint-control and candidate arms above, using exactly the same backend and calibration compute; only \(L_{\mathrm{NCKD,end}}\) and then \(C_q\) are added in predeclared ablations.

Freeze both masks before evaluating disjoint pair documents, full-vocabulary dense-versus-sparse KL/centered-logit error, WikiText NELBO, and the existing GSM8K capability view. Add a matched-count pair bank only as a separate factorial diagnostic, and apply it to both A and B. A pair-shuffled null is valid only if query positions and endpoint marginals remain aligned; arbitrary cross-sequence shuffling is not a semantic null.

## What is borrowed and what could be new

Borrowed: target/non-target decomposition (DKD), relation potentials (RKD), correlation/relative preference matching (DIST), value-plus-derivative matching (Sobolev), sensitivity matching (2ndMatch), reconstruction objectives and calibration overfit warning (Shin), and budget-preserving sensitivity allocation (LiPRA).

Potentially new only after evidence: a fixed-weight DLM pruning allocation whose DKD-style conditional non-target context response is measured by \(C_q\) over natural masked-context pairs, with a demonstrated held-out downstream/capability benefit over the endpoint control under the same support and exact budget. NELBO is complementary evidence. This is a narrow empirical framework claim. It does not establish AR-versus-DLM specificity, global optimality, causal commitment preservation, or universal superiority.

## Decision

Retain the DKD-style conditional-response objective as one objective challenger. Do not merge it with the prior centered-logit response, Fisher weighting, simple tail vectors, Sobolev/2ndMatch Jacobians, clock/coverage features, rollout value, or a new optimizer in the first comparison. No experiment is scheduled by this note.
