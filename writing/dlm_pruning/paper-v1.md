# Conditional Response Preservation for Pruning Diffusion Language Models

**Internal manuscript, v1.** This revision reports the completed four-arm evaluation from the frozen source artifacts. It is an evidence-grounded internal paper, not a publication submission.

## Abstract

A static pruning mask for a diffusion language model must operate while visible context changes. We study whether a pruning objective should preserve those changes in addition to matching predictions at individual masked states. For each unresolved query, we form nested gold-context states, measure dense–sparse error in gold-token log-odds, and add finite response errors across three phase distances. A fixed Wanda ranking, local block probes, and exact 50% sparsity produce four allocations from the same calibration bank: endpoint-only A, natural-chain Multi, Cross, and scale-matched CrossMatched. On the frozen primary set of 1,119 GSM8K questions, Multi loses 10 net answers to A (64 gains, 74 losses; −0.894 percentage points; exact p=0.44372; Holm p=1; unadjusted paired-bootstrap 95% interval [−2.949, 1.162]). The two connection controls are similarly inconclusive: Multi−Cross is −0.089 points (74/75; Holm p=1) and Multi−CrossMatched is +0.179 points (67/65; Holm p=1). Thus this experiment measures a distinct response-preservation preference, but it does not establish an added downstream benefit for this bank, allocator, checkpoint, sparsity, or benchmark. The result also does not establish equivalence or a universal failure of response-preserving pruning.

![Measured primary paired effects for the three prespecified contrasts.](/home/tmluser1/sap/writing/dlm_pruning/review_2026-09-27/experiment/primary-effects.png)

**Figure 1. Measured primary paired effects.** Each point is the candidate-minus-reference exact-match difference on the frozen 1,119-question primary set; bars are unadjusted paired-bootstrap 95% intervals. The three contrasts are Multi−A, Multi−Cross, and Multi−CrossMatched. Exact McNemar and Holm-adjusted values are reported in Table 2. This figure is generated from the verified per-question outputs by the experiment review handoff.

## 1. Introduction

Pruning removes weights from a language model while seeking to retain its ability to use context. In a masked diffusion language model (DLM), context changes as the reverse process reveals tokens. LLaDA predicts masked tokens from partially observed text through a Transformer-based reverse process [Nie et al., 2025](https://arxiv.org/abs/2502.09992v3). A single static mask therefore serves many context states, while calibration must decide which parts of the dense model's behavior to preserve.

Matching predictions at sampled states is a direct calibration target. It does not distinguish every pattern of error across related states, however. If the same unresolved query is evaluated before and after a context reveal, a response error is the difference between the endpoint errors. The distinction is useful mathematically: endpoint loss can assign the same cost to residual patterns whose context responses differ. Whether that distinction improves answers is an empirical question.

**Can a static DLM pruning allocation benefit from preserving how confidence changes when context is revealed, under the same state bank, weight ranking, and sparsity budget?**

We answer this question with a controlled four-arm study. A matches dense predictions at individual states. Multi adds response errors along nested reveal chains at three phase distances. Cross and CrossMatched keep the same state bank but alter the connections, testing whether the selected natural-chain pairing explains any effect. All four arms use the same native sparse-prefix Wanda ordering, the same 224 projection units, and exactly 50% removal of the 6,979,321,856 prunable weights.

The completed primary comparison gives a narrow answer. On the 1,119 questions held out from the two earlier 100-question screens, Multi obtains 636 correct answers, versus 646 for A. It has 64 gains and 74 losses, for −0.894 percentage points. The interval spans both directions and the exact test is not conventionally significant after Holm correction. Multi also differs from Cross by −0.089 points and from CrossMatched by +0.179 points; all three fixed Holm-adjusted p-values equal 1. These observations constrain the tested criterion and allocator. They do not identify a causal mechanism, prove no effect, or reject response-preserving objectives generally.

The contribution is therefore a concrete calibration construction and an inconclusive result about its incremental capability value at this setting. The mathematical objective separates endpoint and response errors; the completed comparison did not establish that this separation produces a GSM8K advantage under the frozen protocol. Fresh calibration diagnostics are reported as diagnostics, without treating their lower or higher losses as explanations for task outcomes.

## 2. Related work

### 2.1 Weight selection and sparsity allocation

Wanda ranks weights using magnitude and input activations, independently within each output row [Sun et al., 2024](https://arxiv.org/abs/2306.11695v3). EvoPress searches nonuniform compression profiles and emphasizes that compression errors across model components need not combine independently [Sieberling et al., 2025](https://proceedings.mlr.press/v267/sieberling25a.html). These works separate local weight selection from the allocation of a global budget. We freeze the Wanda ordering and surviving values and vary only the block allocation induced by the calibration objective.

### 2.2 Calibration for diffusion language models

DLM compression uses masked states and generation-related signals. Sink-Aware Pruning averages soft sink scores over noised calibration inputs and feeds the resulting activations to Wanda or SparseGPT [Sink-Aware Pruning](https://arxiv.org/html/2602.17664). Quant-dLLM uses partially visible calibration inputs for mixed-precision allocation under an average bit budget [Quant-dLLM](https://proceedings.iclr.cc/paper_files/paper/2026/file/805da7ef883245cb35e012cc179a5f6f-Paper-Conference.pdf). FAIR-Calib probes a teacher to estimate frontier and reliability weights for hidden-state reconstruction in quantization [FAIR-Calib](https://arxiv.org/html/2606.06547). Our study asks a narrower controlled question: when the calibration states and local ranking are held fixed, does coupling errors for one unresolved query across selected contexts change downstream outcomes?

### 2.3 Preserving changes in behavior

Sobolev Training incorporates target derivatives alongside function values in approximation and distillation [Czarnecki et al., 2017](https://arxiv.org/abs/1706.04859v3). 2ndMatch uses projected Jacobian measurements to preserve sensitivity during diffusion-model finetuning [2ndMatch](https://arxiv.org/html/2506.05398). Our term measures finite changes under discrete gold-context reveals and uses them to allocate a static mask. The present evidence supports that operational distinction; it does not establish a broader priority or novelty claim.

## 3. Method

### 3.1 Paired masked states and readout

Let D be the dense model and M a static sparse model. For each clean sequence, we sample a fixed query set Q before constructing states. Every query remains masked. Each non-query position receives an independent uniform variable U. At visibility probability p_j, its gold token is revealed when U≤p_j; sharing U across phases creates nested visible sets. The visibility probability describes eligible context positions, not the realized whole-input reveal fraction.

The calibration bank contains eight length-256 sequences, two chains per sequence, eight queries per sequence, and eight phases per chain. The eight visibility probabilities are equally spaced in logit space from 0.05 to 0.95. Unchanged states are retained. These are gold-context calibration chains, not generated denoising trajectories.

For query q with gold token y, the scalar readout is the gold-versus-rest log-odds

\[
f_X(x,q)=\log\frac{p_X(y\mid x,q)}{1-p_X(y\mid x,q)},\qquad
e_{c,j,q}=f_M(x_{c,j},q)-f_D(x_{c,j},q).
\]

For any paired phases j and k, the response error is exactly

\[
\bigl[f_M(x_{c,k},q)-f_M(x_{c,j},q)\bigr]
-\bigl[f_D(x_{c,k},q)-f_D(x_{c,j},q)\bigr]
=e_{c,k,q}-e_{c,j,q}.
\]

Thus the response term reweights finite endpoint errors according to how they change across the selected context transition. The scalar readout does not measure redistribution among incorrect tokens.

### 3.2 Endpoint and response objectives

The endpoint objective averages squared residuals over phases, chains, sequences, and queries:

\[
A=\mathbb E_{\mathrm{sequence},c,q}\left[\frac18\sum_{j=0}^{7}e_{c,j,q}^{2}\right].
\]

The natural phase pairs are

| Scale | Phase pairs |
|---|---|
| 1 | (0,1), (2,3), (4,5), (6,7) |
| 2 | (0,2), (1,3), (4,6), (5,7) |
| 4 | (0,4), (1,5), (2,6), (3,7) |

For each scale d, C_d averages squared differences of e over its four pairs. Multi uses

\[
C_{\mathrm{natural}}=(C_1+C_2+C_4)/3,\qquad L_{\mathrm{Multi}}=A+C_{\mathrm{natural}}.
\]

Every phase appears once at each scale. Therefore (a−b)²≤2a²+2b² gives C_d≤4A and 0≤C_natural≤4A. This bound is an algebraic property of the finite objective; it gives no downstream accuracy guarantee.

Cross connects states from different chains in both directions while retaining the same query and phase indices. CrossMatched uses the same cross connections with β=0.932319907797, fixed from the uniform-model calibration ratio C_natural/C_cross. The four objectives are A, A+C_natural, A+C_cross, and A+βC_cross. These controls assess the specified pairings; they do not isolate a universal semantic property of natural chains.

### 3.3 Fixed-budget allocation

All four arms use the native sparse-prefix Wanda rankings and the same surviving weight values. Each of 32 blocks is probed at 48% and 52% sparsity across seven tied projections. The finite objective change per extra removed weight is

\[
g_\ell=\frac{L(M_{\ell,52})-L(M_{\ell,48})}
{N_{\mathrm{pruned}}(M_{\ell,52})-N_{\mathrm{pruned}}(M_{\ell,48})}.
\]

Average ranks map costs to rates from 45% to 55%; grouped row-count rounding enforces exactly 3,489,660,928 removed weights out of 6,979,321,856. The allocation is a local-probe heuristic: it extrapolates finite block perturbations while changing multiple blocks together. Cached probe outputs are shared across objectives, but constructing the bank and probes still requires model inference.

## 4. Experimental setup

The checkpoint is GSAI-ML/LLaDA-8B-Base, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, evaluated in BF16. GSM8K uses fixed five-shot requests, 256 denoising steps, a 256-token generation limit, block length 256, temperature zero, and strict-match grading. Seeds, prompts, model identities, masks, and source hashes are frozen in the experiment configuration.

Each arm generates all 1,319 questions. The primary analysis uses 1,119 IDs after removing the 200 questions examined in the earlier development and separate100 screens. This partition is fixed, but it does not certify that the primary questions were unseen elsewhere in project history. The development and separate100 results are retained as historical screens rather than independent confirmatory evidence.

The three primary contrasts are Multi−A, Multi−Cross, and Multi−CrossMatched. We use two-sided exact McNemar tests with Holm correction across these three contrasts. Paired bootstrap intervals use 10,000 question-level draws from `np.random.default_rng(20260927)` and are unadjusted. Fresh diagnostics use a separate eight-span bank of 128 states; uncertainty is measured over spans. The response-sign metric uses all 28 within-chain phase pairs whose dense response magnitude exceeds 10⁻⁶, rather than only the 12 objective edges.

## 5. Results

### 5.1 Measured primary effects

The four-arm result is shown in Figure 1 and summarized in Tables 1–4. On the primary 1,119 questions, A obtains 646 correct answers, Multi 636, Cross 637, and CrossMatched 634.

**Table 1. Primary exact-match scores.** Correct answers on the frozen primary set of 1,119 GSM8K questions.

| Arm | Correct / 1,119 |
|---|---:|
| A | 646 |
| Multi | 636 |
| Cross | 637 |
| CrossMatched | 634 |

**Table 2. Fixed primary paired contrasts.** Gains and losses are counted question by question; effects are candidate-minus-reference percentage points. Bootstrap intervals are unadjusted paired-bootstrap 95% intervals, and p-values are exact McNemar values with Holm adjustment over the three fixed contrasts.

| Contrast | Gains / losses | Net | Effect (pp) | Exact p | Holm p | Bootstrap 95% interval (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Multi−A | 64 / 74 | −10 | −0.894 | 0.44372 | 1 | [−2.949, 1.162] |
| Multi−Cross | 74 / 75 | −1 | −0.089 | 1 | 1 | [−2.234, 1.966] |
| Multi−CrossMatched | 67 / 65 | +2 | +0.179 | 0.93068 | 1 | [−1.787, 2.145] |

The point estimate for Multi−A is negative, with 64 questions changing from incorrect under A to correct under Multi and 74 changing in the opposite direction. The interval includes both directions. The two connection contrasts are also small relative to their intervals. These tests do not establish equivalence, and the result does not show that response preservation fails in other banks, allocators, models, or benchmarks.

For context, the descriptive full-1319 scores are A 764, Multi 758, Cross 757, and CrossMatched 755. On the previously examined 200 questions, the scores are 118, 122, 120, and 121, respectively. The primary partition is the inferential sample; these additional scores are reported to make the complete evaluation visible.

### 5.2 Historical screens

**Table 3. Historical 100-question screens.** These descriptive screens precede the primary analysis and are not certified unseen across the complete project history.

| Arm | Development 100 | Separate 100 |
|---|---:|---:|
| Multi | 63 | 59 |
| A | 60 | 58 |
| Uniform | 54 | 52 |

On the separate100 screen, Multi−A has 6 gains and 5 losses (+1.0 point; exact p=1; Holm p=1). Multi−Uniform has 13/6 (+7.0 points; exact p=0.16707; Holm p=0.50121), and A−Uniform has 13/7 (+6.0 points; exact p=0.26318; Holm p=0.52635). These screens were fixed before the full comparison and are not certified unseen across the complete project history. They do not override the primary result.

### 5.3 Fresh calibration diagnostics

**Table 4. Fresh calibration diagnostics.** Values are measured on the separate eight-span bank; response-sign flips use the stated 28-pair threshold. These diagnostics are not task-accuracy outcomes.

| Arm | A | C_natural | C_cross | Query CE | Response sign flips |
|---|---:|---:|---:|---:|---:|
| A | 0.732798 | 0.868318 | 1.017252 | 3.008185 | 0.075893 |
| Multi | 0.750689 | 0.867225 | 1.009932 | 3.021240 | 0.074498 |
| Cross | 0.724621 | 0.835103 | 0.975545 | 3.019861 | 0.074219 |
| CrossMatched | 0.724543 | 0.832330 | 0.975925 | 3.020518 | 0.073940 |

These values are measured on the fresh eight-span bank. Cross and CrossMatched have lower response losses than the other arms on this diagnostic, while Multi has the highest endpoint A and query cross-entropy among the four. The differences are diagnostic properties of the frozen masks. They do not supply a causal mechanism for the GSM8K outcomes, and lower calibration loss does not imply higher task accuracy here.

The current run took 8.236 wall-clock hours. The current run plus the preserved first scheduler-failure attempt records 1,351,936 forward calls. This accounting excludes earlier construction of reused probes, banks, and masks, so it is not an end-to-end calibration-cost estimate.

## 6. Discussion and limitations

The mathematical objective distinguishes endpoint prediction error from finite response error for a fixed unresolved query. The completed comparison did not establish that adding the response term improves a static pruning mask's task performance. Under one bank, one LLaDA checkpoint, one 50% sparsity level, one local allocation heuristic, and one GSM8K protocol, Multi is 0.894 points below A on the primary sample, with uncertainty spanning zero. The cross controls likewise do not separate a natural-chain advantage from the tested alternatives.

The diagnostic and task results also point in different directions. Cross and CrossMatched have lower fresh response losses than Multi, while the source-direction contrasts are Multi−Cross=−0.089 points and Multi−CrossMatched=+0.179 points. These observations are descriptive; they do not identify why the masks differ. The gold-context bank may not represent contexts generated by a sparse model, and the scalar gold-versus-rest readout does not measure alternative-token relations.

The local block probes approximate a joint allocation, so the result constrains the objective, bank, ranking, rounding rule, and backend together. A single calibration realization and checkpoint cannot establish replication or generalization. Comparisons with competitive DLM pruning baselines and complete calibration accounting are also outside this four-arm study. Future work should predeclare any new bank or allocator and retain a fixed downstream primary contrast.

## 7. Conclusion

Preserving how a DLM's confidence changes across paired context states is a well-defined extension of endpoint prediction matching. We implement it with nested gold-context chains and three response scales, then compare it with two cross-chain controls under a shared 50% pruning budget. On the primary 1,119 GSM8K questions, the added response term does not show a downstream advantage in this frozen setup: Multi−A is −0.894 points with an unadjusted 95% interval of [−2.949, 1.162] and Holm p=1. The result is a constraint on this criterion and allocator, not an equivalence claim or a rejection of response-preserving pruning in general.

## Appendix A. Constructed algebraic illustration

This appendix contains a constructed example only. It is not a model measurement, a calibration result, or task evidence. Its purpose is to make the response identity concrete.

![Constructed residual sequences with equal endpoint error and different response error.](/home/tmluser1/sap/writing/dlm_pruning/figures/response-example.png)

**Figure 2. Constructed residual example, not observed evidence.** For eight synthetic phases, a constant residual (+1 at every phase) and a changing residual (+1,+1,−1,−1,+1,+1,−1,−1) both have endpoint loss A=1. Under the phase pairs in Section 3.2, the constant residual has zero response loss; the changing residual has C_2=4 and mean natural response loss 4/3. The resulting combined losses are 1 and 7/3. The figure is generated from the exact inputs saved in `figures/response-example.json`; no model output enters the construction.

## Appendix B. Evidence and reproducibility

The source of truth for empirical values is the verified per-question output and report at `/home/tmluser1/sap/experiments/dlm_crosschain_control50/output/`. `closeout.json` records source hashes, exact sample coverage, official grading, paired statistics, fresh diagnostics, historical reproduction, and attempt-level costs. The writing review directory contains an independent CPU-only recomputation of the primary scores and all three paired effects, plus the direct JSON-key inventory used for this revision. The experiment review handoff supplies the measured primary-effects figure and its plotted data. No model is loaded by the writing audit.
