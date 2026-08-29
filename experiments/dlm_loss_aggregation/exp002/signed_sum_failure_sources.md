# Signed SUM failure: primary-source check

This note checks the cited literature against the original papers and authors'
code. It separates reported evidence from implications for EXP-001/EXP-002.
Obsidian was unavailable during this check, so no prior research state is
inferred here.

## Bottom line

There were two distinct August 28 experiments, and the cancellation hypothesis
applies to only one of them:

- **CGQ-SparseGPT (morning):** no loss, backward pass, parameter gradient, or
  signed-gradient aggregation was used. CGQ applied a positive token weight to
  SparseGPT activations before accumulating an activation Gram matrix. Signed
  gradient cancellation therefore cannot explain its WinoGrande result.
- **DLM loss aggregation (afternoon):** the SUM score averaged the signed
  parameter-pruning effect \(-w\,\partial L/\partial w\) over states. Cancellation
  is a well-motivated hypothesis for this experiment, but is not yet a causal
  explanation of its LLaDA result.

Prior work provides two especially relevant observations for the latter:

1. Molchanov et al. argue that a trained network's signed first-order
   feature-map effect can have expectation near zero while its absolute effect
   remains informative, and their CNN experiment favored absolute over signed
   oracle changes.
2. GBLM-Pruner directly reports a severe failure of gradient accumulation
   relative to sample-axis \(L_1/L_2\) aggregation on LLaMA-2-7B.

Neither paper proves that cancellation caused DLM-SUM's 0/1319 GSM8K result.
The controlled evidence for that claim must come from DLM-ABS and DLM-SQUARE,
which use the same 80 LLaDA gradients and differ only in aggregation.

## CGQ-SparseGPT applicability audit (separate experiment)

Primary experiment sources: the committed
[`cgq_token_weights`](https://github.com/songeunseo/sap/blob/3e32ed6b5aa51737a8882f2dd23ae125e53772fb/cgq_sparsegpt.py),
[`SparseGPT.add_batch`](https://github.com/songeunseo/sap/blob/3e32ed6b5aa51737a8882f2dd23ae125e53772fb/lib/sparsegpt.py),
[two-hour report](https://github.com/songeunseo/sap/blob/db4c22de9730431e0e4496f40d09ca7a29fb6fc1/codex/cgq_sparsegpt_2h/results/report.md),
and [WinoGrande report](https://github.com/songeunseo/sap/blob/6790e3efe4e921649b7d8361f8ad69ad112978b7/codex/cgq_sparsegpt_downstream_50/results/report.md).

### What CGQ actually computed

For token \(t\), the implementation computes

\[
c_t=\max \operatorname{softmax}(\mathrm{logits}_t),\qquad
m_t=\begin{cases}1.0&\text{masked}\\0.7&\text{visible}\end{cases},\qquad
r_t=m_t+\sqrt{c_t}.
\]

It detaches \(r_t\), multiplies the input activation \(x_t\) by it, and then
uses the existing SparseGPT accumulator. Ignoring the accumulator's common
normalization constant, this changes the activation statistic to

\[
H_{\mathrm{CGQ}}=\sum_t(r_tx_t)(r_tx_t)^\top
                 =\sum_t r_t^2x_tx_t^\top.
\]

Consequently:

- \(r_t\) is always positive (`0.7` to `2.0` by construction), and its
  effective contribution to \(H\) is **squared**.
- Every per-token outer-product contribution is positive semidefinite, and
  each Hessian diagonal contribution is \(r_t^2x_{t,i}^2\geq0\).
- Off-diagonal activation products can be negative, but they are not signed
  loss gradients averaged across samples. Taking
  `abs(gradient * activation)`, squaring that quantity, or applying GBLM's
  sample-axis norm is therefore not a modification of the statistic CGQ used.

The signed-cancellation explanation is thus **inapplicable to CGQ-SparseGPT**.
It addresses the later DLM-SUM estimator, whose implementation explicitly
forms \(-w\,\partial L/\partial w\), not the earlier CGQ estimator.

### What the masked-diffusion loss does and does not imply here

The primary LLaDA paper, [Nie et al., *Large Language Diffusion
Models*](https://arxiv.org/abs/2502.09992), defines its Eq. (3) cross-entropy
with an indicator that includes only masked positions. This supports the
general statement that visible tokens are not direct prediction targets in the
training loss. Visible tokens can still influence masked predictions through
the Transformer context.

It does not support a gradient-based explanation of CGQ because CGQ never
computes that loss or calls backward: it performs forward passes to obtain
confidence and then accumulates both masked- and visible-token activations.
Nor does the LLaDA paper establish that visible-token parameter-gradient paths
must be smaller or noisier. That claim remains a hypothesis even for the later
DLM-gradient experiment.

### What the observed CGQ result supports

At 50% sparsity, CGQ changed 8.0565% of mask decisions and improved the
held-out masked-token proxies (KL `-8.65%`, top-1 agreement `+2.6549` points),
but WinoGrande fell from `69.06%` to `68.11%` (`-0.95` point, 12 examples).
The observed result is therefore a **single-task transfer failure**, not
evidence that signed gradients cancelled and not evidence that CGQ degraded
every target.

The implementation and evidence leave three plausible, unproven explanations:

1. **Proxy mismatch:** confidence/mask-weighted layer reconstruction and
   masked-token distribution preservation need not optimize WinoGrande
   decisions.
2. **Combined heuristic:** mask state and confidence enter one formula, and
   pre-multiplication makes the effective coefficient \(r^2\). The run cannot
   attribute the downstream change to the mask prior, confidence, or their
   interaction. The relevant ablation is Plain, mask-only, confidence-only,
   and combined CGQ—not Wanda versus a gradient score.
3. **Calibration sensitivity:** the run used 4 WikiText-2 sequences across 4
   timesteps (16 states, 256 tokens). The ICLR 2025 calibration paper below
   makes source mismatch a plausible concern, but it neither identifies the
   cause here nor recommends fixing it by merely increasing sample count.

One further caution is visible in the recorded calibration statistics: at
\(t=0.8\), visible tokens had higher mean \(r\) than masked tokens (`1.5440`
versus `1.4973`) because their confidence was higher. Thus the combined score
does not consistently rank all masked tokens above all visible tokens, despite
the `1.0` versus `0.7` mask prior.

## 1. Molchanov et al. (ICLR 2017)

Primary sources: [ICLR record](https://openreview.net/forum?id=SJGCiw5gl),
[arXiv](https://arxiv.org/abs/1611.06440), and
[author-hosted PDF](https://users.aalto.fi/~ailat1/publications/molchanov2017iclr_paper.pdf).

### What the paper shows

- Section 2.2 derives a first-order Taylor approximation for removing a
  feature map. Its structured feature-map criterion takes an absolute value of
  the spatially aggregated first-order term; for a minibatch, it computes the
  criterion per example and averages across examples.
- In "Relation to Optimal Brain Damage" (PDF p. 5), the paper argues that after
  sufficient training the signed term \(y=(\partial C/\partial h)h\) tends
  toward zero in expectation, while \(E|y|\) reflects its nonzero variability.
- Section 3.1 reports that `Oracle-abs` retained higher accuracy than
  `Oracle-loss` in its CNN experiment. The authors interpret this as large
  individually loss-decreasing changes becoming unstable when accumulated.

### Correction and scope

This is not a theorem that every individual weight gradient in any trained
model has zero expectation. The object is a hidden feature-map effect in
trained CNNs, the pruning unit is a feature map, and pruning is interleaved
with fine-tuning elsewhere in the study. It is direct precedent for a
cancellation risk, not proof for LLaDA or the DLM calibration distribution.
The 2017 paper supports absolute first-order effects; it does not introduce the
EXP-001 mean-square statistic.

## 2. Taylor-FO-abs and Taylor-FO-sq (arXiv:2310.20203)

Primary source: [Sapkota and Bhattarai, *Importance Estimation with Random
Gradient for Neural Network Pruning*](https://arxiv.org/html/2310.20203)
([arXiv record](https://arxiv.org/abs/2310.20203)).

Section 2 explicitly defines

\[
I_i^{\mathrm{abs}}=\frac1M\sum_{n=1}^M|x_{i,n}\,\delta x_{i,n}|,
\qquad
I_i^{\mathrm{sq}}=\frac1M\sum_{n=1}^M(x_{i,n}\,\delta x_{i,n})^2.
\]

Thus the cited `Taylor-FO-abs` and `Taylor-FO-sq` terminology and aggregation
order are correct. Under the analogy \(x\,\delta x \leftrightarrow -w\,g\),
these have the same aggregation form as EXP-001 ABS and SQUARE.

However, this is neuron/channel pruning on ResNet/VGG with CIFAR-100/STL-10,
not weight-level unstructured LLM pruning. The equivalence to
weight-times-parameter-gradient is an application-level analogy, not a result
proved by the paper. Its separate normalization proposal normalizes a random
gradient injected at the network output to magnitude one; it is not evidence
that every sample's full parameter-gradient tensor should be normalized in a
DLM experiment. This is an arXiv/workshop paper, not an ICLR main-conference
paper.

## 3. GBLM-Pruner (arXiv:2311.04902)

Primary sources: [paper HTML](https://arxiv.org/html/2311.04902v2),
[arXiv record](https://arxiv.org/abs/2311.04902), and the authors'
[`gradient_computation.py`](https://github.com/VILA-Lab/GBLM-Pruner/blob/main/gradient_computation.py#L139-L156).

### Exact aggregation

Equation (4) is

\[
S_{ij}=|W_{ij}|\,\|G[:,i,j]\|_p,
\]

where the norm is across calibration samples for each weight. The official
code accumulates `abs(grad * scale)` for \(L_1\), and
`(grad * scale)**2` followed by a square root for \(L_2\). It clears gradients
between samples. Division by sample count is omitted, but with a fixed sample
count this does not change the ranking.

Therefore the claim that GBLM-Pruner "normalizes each calibration sample's
gradient first and then averages" is incorrect. Its phrase "normalizing ...
across samples" means taking a per-weight \(L_1\) or \(L_2\) norm along the
sample axis. It does not rescale each sample's entire gradient tensor to equal
norm. In EXP-001 terms, its gradient-only variants are closest to ABS and RMS
(RMS is ranking-equivalent to SQUARE), not a new sample-normalized estimator.

### What its ablation establishes

Section 3.4, Table 5 reports LLaMA-2-7B perplexity at 50% sparsity:

| Gradient aggregation | Perplexity |
|---|---:|
| \(|W|\,|G_{acc}|\) | 119.72 |
| \(|W|\,\|G\|_1\) | 7.17 |
| \(|W|\,\|G\|_2\) | 7.09 |

This is strong empirical evidence within that setup that signed accumulation
followed by an outer absolute value lost useful information relative to
per-sample magnitude aggregation. It is not identical to EXP-001 SUM, which
retains the final sign and prunes the smallest signed effects, including
negative values.

The full GBLM metric also adds a Wanda-like activation term and a validation-
selected scale \(\alpha\). Its reported superiority over Wanda/SparseGPT is
therefore evidence for that complete method on autoregressive LLaMA models,
not proof that sample normalization is the decisive trick or that a pure DLM
gradient metric will win on LLaDA. The paper itself notes that superiority is
not consistent on every individual zero-shot task.

## 4. Calibration-data sensitivity (ICLR 2025)

Primary source: [Ji et al., *Beware of Calibration Data for Pruning Large
Language Models*](https://proceedings.iclr.cc/paper_files/paper/2025/file/2ede933e10afa991a10b6f36b6522129-Paper-Conference.pdf)
([conference record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ede933e10afa991a10b6f36b6522129-Abstract-Conference.html)).

The paper supports the claim that calibration **source** matters more as
sparsity rises in its tested settings. On DCLM-7B with Wanda, the reported
range across four calibration sources grows from less than 0.1 percentage
point below 50% sparsity, to about 0.5 at 50%, and 2.3 at 60%; more structured
sparsity also increases the range.

It does **not** support the recommendation to simply use more calibration
samples. Section 3.3 varies 64--2048 sequences and reports only 0.1--0.2 point
fluctuations in mean performance; increasing the amount does not close the
gap between data sources, though it reduces sampling standard deviation. The
paper's conclusion is that a small amount can be adequate and source/training-
data similarity matters.

Those experiments use autoregressive DCLM/LLaMA models and methods such as
Wanda, DSnoT, and OWL, not official DLM-loss gradients or LLaDA. Calibration
sensitivity is therefore a plausible additional variable, not an explanation
of SUM versus ABS when the calibration manifest is held exactly fixed.

## Implications for DLM aggregation EXP-001/EXP-002 (not CGQ)

- The present comparison already isolates aggregation more cleanly than the
  proposed 2x2 description: SUM, ABS, and SQUARE use the same official DLM
  loss, weights, 80 states, corruptions, and backward passes.
- EXP-001 computes a **parameter-pruning effect**
  \(d_{i,s}=-w_i\,\partial L_s/\partial w_i\), not
  `gradient × activation`, and it does not assign different parameter sets to
  masked versus visible tokens. Transformer weights are shared across token
  positions.
- A masked-position-only loss does allow visible tokens to affect the loss
  indirectly through the network, but the cited papers do not establish that
  those paths are necessarily smaller or noisier in LLaDA.
- DLM-SUM's observed 0/1319 result is consistent with cancellation being
  harmful, especially given EXP-001's low sign consistency, but the decisive
  controlled comparison is the still-required full DLM-ABS/SQUARE result.
- A new per-state gradient normalization experiment is not justified as part
  of EXP-002 by these citations. It would be a separate follow-up with a newly
  specified normalization axis and norm.
