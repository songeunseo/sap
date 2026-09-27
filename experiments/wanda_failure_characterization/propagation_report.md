# Where Wanda's local pruning error becomes functional damage

## Artifact-supported facts

### Exact reuse and reproduction gate

This analysis reused the pinned model, the frozen held-out manifest with SHA-256 `bb2cdaa6985a6ae6f75b7623a58653dcc80c73b7240cddbea551d62e7bab2b5e`, the clean Wanda `A_j`, the same 224-module ordering, and masks reconstructed from the same weights and `A_j`. All 448 reconstructed 50%/75% mask hashes matched the prior manifest.

Before enabling internal observers, the original collector was rerun for all 40 states and 224 modules with the exact `[sham, 50%, 75%]` suffix batch. Against the frozen failure map, both 50% masked-token KL and loss delta had max and mean absolute error `0.0`. The new trace retained the identical batch shape/order. Model-weight SHA-256 was unchanged before and after: `2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc`.

### Common functional perturbation points

For `q_proj`, `k_proj`, `v_proj`, and `attn_out`, the common branch point is the output of `attn_out`, immediately before the attention residual addition. Because residual dropout is disabled in eval mode, this is the exact attention branch contribution.

For `ff_proj`, `up_proj`, and `ff_out`, the common branch point is the output of `ff_out`, immediately before the MLP residual addition. `C1` is the input to `ff_norm` for attention-side targets (post-attention residual), and the target block output for MLP-side targets. Attention traces additionally record the same block's output after its MLP. Every later block output, final pre-normalization hidden state, final post-normalization hidden state, and masked logits was observed.

No intervention beyond the existing Wanda Linear mask occurred. Full tensors were reduced immediately to FP32 sufficient statistics: absolute mean-square difference, relative energy, relative L2, masked/unmasked energies, masked enrichment, and direction cosines.

### Full 224-module propagation map

Across modules, relative-energy distributions (min / median / max) were:

| Checkpoint | min | median | max |
|---|---:|---:|---:|
| common branch | 0.0000256 | 0.02355 | 0.06103 |
| C1 residual | 0.00000169 | 0.000814 | 0.005153 |
| final post-norm hidden | 0.000252 | 0.000920 | 0.004606 |
| masked logits | 0.000223 | 0.000600 | 0.004327 |

Finite propagation gain `R_L2(final post-norm)/R_L2(C1)` had p10/median/p90 `0.668/0.911/2.652`, with range `0.566--13.255`. Therefore both attenuation and amplification occur, but large gains are not synonymous with large KL.

For attention-side modules, the exact same-block MLP repair factor had p10/median/p90 `0.920/0.975/1.096` and mean `1.000`. The same-block MLP usually changes perturbation magnitude modestly, with a small number of larger exceptions.

### Masked versus unmasked localization

Common-branch masked enrichment ranged from `0.184` to `1.494`, with p10/median/p90 `0.731/0.890/1.109`. Its pooled Spearman with `Y50_KL` was only `0.080`; after within-type standardization it was `0.191`. Masked concentration is therefore heterogeneous but does not explain the global reconstruction/KL mismatch.

The named terminal MLP cases had moderate enrichment: block31 ff_out `1.133`, ff_proj `1.193`, up_proj `1.215`; block30 ff_out was higher at `1.402` despite lower KL. Block0 v_proj began near neutral (`1.013`) and its enrichment fell below one through later blocks. This is the opposite of what a masked-concentration explanation for these anomalies would require.

### Stage-by-stage association chain

The full-map pooled Spearman chain was:

| Stage comparison | Spearman |
|---|---:|
| Linear `E_rec` → common branch relative energy | 0.872 |
| common branch → C1 relative energy | 0.839 |
| C1 → final post-norm relative energy | 0.373 |
| final post-norm relative energy → KL | 0.775 |
| masked-logit relative energy → KL | 0.883 |
| Linear `E_rec` → KL | 0.667 |

Thus the individual Linear metric is generally a good proxy for the perturbation emerging from its parent sublayer. The largest loss of stage-to-stage rank information occurs during propagation from C1 to the final hidden state. Mapping the final hidden perturbation through final normalization/readout to masked logits then materially improves association with KL.

### Direction and final readout

Final-hidden perturbations were almost orthogonal to the corresponding sham representation: mean-cosine values across modules ranged from `-0.123` to `0.096`, and aligned energy fractions had median `0.00146`. Neither final-hidden cosine nor aligned fraction had a strong pooled association with KL (`-0.143` and `0.084`). Sham-logit/perturbation cosine was also uninformative (`rho=-0.001`).

Within masked positions, per-token logit-delta magnitude versus token KL Spearman averaged between `0.037` and `0.421` across modules; its module mean associated with module KL at `rho=0.447`. Direction/readout effects remain after norm, but the observed coarse cosine descriptors do not isolate the relevant direction.

No C1 or final-hidden relative-energy denominator fell below `1e-12`. Minimum module-wise state denominators were `9.56e-7` at C1 and `5.37e-5` at final hidden. Reported KL/energy ratios therefore have no denominator-floor cases, but are retained as descriptive medians/quantiles rather than standalone rankings.

### Known anomaly: block31 ff_out versus block30 ff_out

| Quantity | block31 ff_out | block30 ff_out |
|---|---:|---:|
| Linear `E_rec` | 0.006054 | 0.042921 |
| branch relative energy | 0.006054 | 0.042921 |
| branch masked enrichment | 1.133 | 1.402 |
| C1 relative energy | 0.003108 | 0.002987 |
| final post-norm relative energy | 0.002075 | 0.002042 |
| masked-logit relative energy | 0.002924 | 0.002398 |
| logit-delta RMS | 0.14348 | 0.13176 |
| median KL/final-hidden energy | 11.664 | 2.306 |
| final KL | 0.022711 | 0.008965 |

The discrepancy is not caused by Linear-to-MLP transformation: for `ff_out`, Linear error is already branch error. It is not caused by masked concentration, which is larger for block30. Residual scaling collapses the seven-fold branch-error difference: the two C1 and final-hidden relative energies become nearly equal. The remaining 2.5-fold KL difference is only partly reflected in logit norm and is chiefly expressed as much larger KL per matched final-hidden energy for block31. Consequently, “the last layer has no opportunity to recover” is not by itself supported: block30's block31 propagation slightly increases relative L2 (`gain≈1.027`), yet final magnitudes converge. The unresolved part is final prediction-direction/readout sensitivity.

Block31 ff_proj follows the same terminal readout-sensitive family: final relative energy `0.002046`, logit relative energy `0.004073`, KL `0.014430`, and median KL/final energy `5.426`. Block31 up_proj has a larger final perturbation (`0.004606`) but lower KL (`0.008823`) and ratio (`1.805`). Even within one terminal MLP, magnitude alone does not order functional damage.

### Known anomaly: block0 v_proj

Block0 v_proj follows a distinct trajectory. Its Linear error `0.022011` becomes a small common-branch perturbation (`0.001425`) through attention, unlike the direct ff_out mapping. C1 relative energy is `0.001361`; the same-block MLP initially attenuates relative L2 to gain `0.880`. The trace continues attenuating to about `0.70` through the early blocks, then grows after the middle layers, reaching final pre-norm gain `1.568`. Final post-norm relative energy is `0.003340`, masked-logit energy `0.001594`, and KL `0.003349`.

This early outlier is therefore propagation/amplification dominated and is not the same mechanism as terminal ff_out's high KL per matched final perturbation.

## Statistical associations with layer/module controls

Pooled associations were strongest for masked-logit energy (`0.883`), C1 energy (`0.801`), and final-hidden energy (`0.775`). After within-module-type standardization they fell to `0.432`, `0.271`, and `0.280`. Branch energy fell from `0.651` pooled to `0.151` controlled; masked enrichment remained weak (`0.191`).

Within individual types, masked-logit energy retained positive Spearman from `0.501` to `0.865` across all seven types. C1 energy was strong for attn_out (`0.783`), ff_out (`0.615`), ff_proj (`0.543`), and v_proj (`0.824`), but nearly absent for q/k projections. Final-hidden energy was stronger for ff_out (`0.787`), q_proj (`0.617`), and up_proj (`0.521`). This heterogeneity argues against one universal propagation stage.

A fixed six-variable descriptive ridge model, cross-validated by leaving out each layer, achieved `R2=0.385` and rank correlation `0.702`. This is materially better than the prior pre-pruning-statistic model, but it uses post-intervention propagation observations and is explanatory only—not a pruning score or evidence of pre-pruning predictability.

## Mechanistic interpretation

**Outcome E — mixed / unresolved.** The full chain rules out pure local-metric mismatch and masked-position concentration. It shows real location-dependent propagation: C1-to-final ordering changes substantially, and masked-logit magnitude is much closer to functional KL than Linear reconstruction. However, no single propagation mechanism explains all outliers after type/layer control.

Two mechanisms coexist in the observed cases:

1. block0 v_proj is attenuated early and amplified later, a downstream-propagation pattern;
2. terminal MLP modules reach similar final-hidden magnitudes but differ markedly in logit response and KL per unit final perturbation, a readout/directional pattern not captured by the measured cosines.

Because these mechanisms differ, converting the pooled propagation result directly into layer weighting or a modified Wanda criterion would overstate the evidence.

## Speculation

The terminal discrepancy may depend on perturbation direction relative to token-specific LM-head decision boundaries rather than global hidden/logit cosine. The early v_proj trace may reflect attention-mediated mixing followed by late amplification. These remain hypotheses: no Jacobian, JVP, random-direction control, or synthetic perturbation was performed.

## Exactly one recommended next diagnostic

Run a small matched-norm perturbation-transplant diagnostic using only the already captured exact block30/block31 ff_out residual perturbations: inject each frozen perturbation at both residual locations on a preregistered subset of the same states, preserving its direction while matching norm, to separate location-dependent propagation from perturbation-direction/readout sensitivity. Do not modify Wanda or run downstream evaluation.
