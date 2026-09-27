# Allocation criterion mismatch audit — 2026-09-14

## Objective and setup
Find one incorrect allocation judgment that could motivate a simple DLM criterion. This is a post-hoc CPU audit of frozen LLaDA-8B-Base revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2, 80 corrupted states, 224 projections and the existing GSM8K mini100. No new model evaluation, mask, or sparsity setting was run.

## Observed downstream and allocation

| Allocation transfer | Correct /100 | B00–07 sparsity | B24–31 sparsity |
|---|---:|---:|---:|
| Uniform |12|64.99%|64.99%|
| OWL |5|59.90%|66.13%|
| DLP public mean path |4|56.50%|75.77%|
| DSA fixed graph |7|56.05%|68.50%|
| AlphaPruning |1|53.04%|78.60%|
| LSA |6|60.01%|73.16%|
| Aggregate |19|69.47%|64.03%|
| Role |24|69.56%|63.93%|

Every manifest removes exactly 4,536,008,704 / 6,979,321,856 weights. These are allocation transfers with frozen DLM-Wanda ranking, not complete reproductions of the original AR pipelines. Mini100 is a repeatedly used development set; the table alone does not establish population differences or general AR allocation failure.

## One concrete judgment to investigate
The pinned DLP public `get_dlp_ratios` path uses the mean of pooled Wanda scores. Its normalization assigns more pruning to blocks with larger means. The [DLP paper](https://arxiv.org/html/2505.23807v1) describes central score magnitude as redundancy using a median; distinguish this paper/code discrepancy when naming the tested baseline.

In the frozen LLaDA data:

- Late-eight / early-eight mean score ratio: **7.40×**.
- Mean score versus depth Spearman: **0.9989**.
- Mean score versus block additive marginal KL per additional pruned weight: **+0.9106**.
- Consequently assigned density versus that marginal cost: **−0.9106**; excluding B31: **−0.9016**. All eight sequence-specific correlations are negative (−0.936 to −0.662).
- Block average channel RMS versus DLP mean: **0.9993**. Thus a large part of this score ordering tracks activation scale.
- Linear-depth partial rank correlation of score and cost drops to **+0.2900**. This is descriptive and does not eliminate nonlinear depth effects.

Block marginal here is sum of seven individually measured projection KL increments divided by their summed exact extra removed counts at 65→70%. It is **not** a jointly pruned block measurement. Never extrapolate these curves to DLP/Alpha rates above75%.

**Interpretation:** the audited allocation direction opposes the historical marginal-damage ordering. That identifies a specific suspect judgment, but does not establish that denoising caused the mismatch or that simply reversing the score improves generation.

## DLM observation checked, and its limit
[LLaDA](https://arxiv.org/html/2502.09992v1) predicts masked tokens from corrupted inputs with a noncausal Transformer. Existing tensors allow a mask-probability diagnostic without new collection.

Increasing nominal corruption probability reduces the average channel RMS and energy effective support in early q/k/v inputs (median within-block Spearman −1.0 over ten mask probabilities). q/k/v share their input and are not three independent confirmations. Energy effective support is a diagonal concentration statistic, not matrix rank or proof of token redundancy.

However, block RMS rankings at the lowest and highest corruption probabilities have **Spearman 0.9949**. Depth correlations stay **0.9941–0.9989** at all ten probabilities. The broad early/late scale ordering already persists at low corruption.

**Decision:** distribution changes alone do not justify timestep weighting or a mask-count correction. Do not promote this as a newly established DLM-specific allocation feature. The data do not isolate repetition of MASK from context changes or generic model depth effects.

## Candidate narrowed, method not yet established
The next bounded hypothesis is: **DLP-style central activation magnitude may mistake denoising-related computation for removable redundancy.** Two causal links remain untested: whether corruption changes the relevant allocation judgment relative to fully visible input, and whether the targeted judgment correction improves pruning outcomes. A matched clean/corrupted diagnostic is specified in `next_diagnostic.md`; it is not running.

Do not use a naked score inversion, a hand-coded late-layer protection schedule, role-max, channel3848 removal, or timestep coefficients as a DLM-specific method on this evidence. The prior timestep and single-super-outlier negatives remain in force. Interaction modeling and a complete downstream predictor are not prerequisites.

## Verification and limitations
Original audit data hashes were checked against their saved references. Baseline prediction/result/config/manifest links, ordered example/prompt/target/protocol identities, projection shapes, integer row counts and global budgets all passed. Activation-source hash, ten-timestep pooling identity, capacity state ordering and finite values passed. This audit uses the previously audited correctness flags; it did not regenerate text, recompute strict EM, or reload every physical mask payload.

The historical capacity collection uses seven-variant BF16 batched suffix forwards. Its fidelity is not established by later corrected batch-one exchange work. Treat the damage correlations as exploratory; a targeted physical batch-one measurement should precede a causal method claim. There is no matched AR model, no fully clean activation collection in this audit, and no token-vector mean/covariance supporting a common-component removal rule.

Reproducible outputs: `analysis.json`, `projection_features.csv`, `block_diagnostics.csv`, `timestep_features.json`, `run.log`; input and script SHA256 hashes are stored in `analysis.json`.
