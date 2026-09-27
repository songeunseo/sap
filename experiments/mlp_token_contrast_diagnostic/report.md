# Does Token Importance Survive at MLP-Neuron Functional Saliency?

## Scope

This was an analysis-only diagnostic. It used no CGQ, Wanda, SparseGPT, masks, pruning, weight updates, downstream evaluation, or automatic follow-up.

Verification note: the spike-specific and reused weighting tests pass. One unrelated legacy EXP-004 evaluation test depends on the historical `/dev/shm` pruning-mask file set and currently fails because that external file set no longer matches its metadata; this diagnostic neither reads nor enters that mask/evaluation path.

## Artifact-supported facts

### Frozen inputs and implementation

- Model: `GSAI-ML/LLaDA-8B-Base`, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, bfloat16.
- States: EXP-001 run `20260828T175010-2484545`, 80 frozen states (8 sequences × 10 timesteps), sequence length 256. State SHA-256: `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`.
- EXP-005 reveal/remain artifact: all 80 state identities and partitions validated; SHA-256 `e847478e57e2f6d3e8ac2adacd7622f96d9ab7c0dddfc24588365f7383d10eb8`.
- Gate location: the actual input to each block's `ff_out`, after `silu(ff_proj(ff_norm(x))) * up_proj(ff_norm(x))`. No trainable parameter was added.
- Gate derivative: `d[l,k,s] = -Σ_p h[l,p,k] ∂L_s/∂h[l,p,k]`. Products and position sums were accumulated in FP32.
- Loss: `Σ_masked alpha*CE / (p_mask*256)`, with `rho=0.5` and unchanged EXP-005 symmetric weights.

The collector evaluates the exact linear loss basis `L_U` and `L_C=(L_R-L_M)/2` with autograd on the same dense forward, then reconstructs `d_R=d_U+d_C` and `d_M=d_U-d_C`. This avoids condition-dependent BF16 VJP rounding while remaining algebraically identical to evaluating the mirrored losses. A preliminary three-independent-VJP check reproduced the scalar loss identity but had nondeterministic BF16 gradient residuals up to 4.7%; it was rejected rather than weakening the sanity tolerance.

Mirror checks over all stored state/layer/neuron elements:

- masked mean alpha: 1 for both symmetric conditions;
- `alpha_R+alpha_M=2` at every masked token;
- maximum scalar-loss absolute/relative residual: `1.91e-6` / `1.64e-7`;
- maximum derivative absolute/relative residual: `2.98e-8` / `1.15e-7`;
- maximum derivative global-relative residual: `3.27e-8`, below the predeclared `0.025` stop limit.

The mirror result is a construction sanity check, not evidence of specialization.

### Score definitions

- Baseline: `S_U[l,k] = mean_s |d_U[l,k,s]|`.
- Contrast: `d_C=(d_R-d_M)/2=d_R-d_U`; `S_C[l,k]=mean_s |d_C[l,k,s]|`.
- Relative contrast: `R_C=S_C/(S_U+safe handling)`.
- Secondary only: `S_R=mean_s|d_R|` and `S_M=mean_s|d_M|` rankings.

The primary floor was `S_U >= 1e-6 * max(S_U)` within a layer. No neuron was excluded, even at the stricter `1e-4 * max(S_U)` check, and excluded neurons contributed zero contrast mass. Thus the reported `R_C` tails are not a near-zero-denominator artifact.

### Aggregate result

Across 32 layers:

- U/R Spearman: mean `0.9999691`, min `0.9999502`.
- U/M Spearman: mean `0.9999706`, min `0.9999528`.
- R/M Spearman: mean `0.9998818`, min `0.9998093`.
- `CV(q_R)`: mean `0.003059`, range `0.001922–0.005170`.
- `CV(q_M)`: mean `0.002989`, range `0.001893–0.005015`.
- U/R normalized mean rank displacement: mean `0.001531`, max `0.001999`.
- U/M normalized mean rank displacement: mean `0.001496`, max `0.001952`.
- `R_C` median is about 1.0–1.6% by layer. `CV(R_C)` is nonzero: mean `0.2120`, range `0.1538–0.2993`.
- U versus `S_C` Spearman: mean `0.8878`, range `0.8416–0.9724`. Contrast magnitude is related to baseline sensitivity but not identical to it.
- Absolute `S_C` rank stability against the global aggregate: mean Spearman `0.556` across sequences and `0.613` across timesteps.
- Relative `R_C` rank stability is substantially weaker: mean Spearman `0.297` across sequences and `0.106` across timesteps.

### Full 32-layer localization

`nMAD` is U/R mean absolute rank displacement divided by 12,288. `B10-R/M` are bottom-10% overlaps against Uniform.

| L | U/R ρ | U/M ρ | R/M ρ | CV(qR) | CV(qM) | median RC | CV(RC) | U/SC ρ | nMAD | B10-R | B10-M |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|.999992|.999992|.999970|.00398|.00390|.01213|.29064|.97243|.00080|.996|.993|
|1|.999991|.999992|.999968|.00400|.00392|.01201|.29378|.96984|.00084|.994|.998|
|2|.999985|.999986|.999943|.00378|.00371|.01188|.27948|.94260|.00110|.994|.994|
|3|.999975|.999976|.999904|.00373|.00366|.01171|.27731|.90900|.00140|.996|.995|
|4|.999962|.999964|.999853|.00357|.00350|.01139|.27209|.86585|.00171|.992|.993|
|5|.999976|.999977|.999906|.00278|.00274|.01084|.22584|.88281|.00138|.993|.996|
|6|.999979|.999979|.999919|.00232|.00228|.01050|.18783|.88637|.00129|.995|.994|
|7|.999980|.999980|.999922|.00201|.00198|.01019|.16498|.88607|.00126|.995|.995|
|8|.999981|.999982|.999928|.00202|.00199|.01024|.16446|.89676|.00124|.996|.995|
|9|.999980|.999980|.999922|.00198|.00195|.01035|.15777|.89512|.00125|.993|.992|
|10|.999982|.999982|.999930|.00192|.00189|.01044|.15495|.89977|.00122|.998|.997|
|11|.999980|.999981|.999924|.00195|.00192|.01067|.15380|.90062|.00126|.994|.995|
|12|.999979|.999980|.999919|.00198|.00194|.01081|.15704|.89229|.00131|.993|.995|
|13|.999977|.999978|.999914|.00201|.00197|.01099|.15644|.88673|.00137|.996|.995|
|14|.999974|.999975|.999901|.00206|.00202|.01123|.15911|.87643|.00144|.995|.990|
|15|.999973|.999974|.999897|.00218|.00214|.01139|.16380|.88016|.00147|.995|.992|
|16|.999971|.999972|.999889|.00220|.00215|.01160|.16558|.87274|.00151|.994|.993|
|17|.999969|.999969|.999878|.00222|.00218|.01189|.16477|.86498|.00160|.997|.995|
|18|.999967|.999969|.999874|.00230|.00225|.01213|.16932|.86052|.00162|.989|.993|
|19|.999961|.999963|.999850|.00244|.00239|.01249|.17369|.85858|.00174|.993|.994|
|20|.999957|.999958|.999833|.00269|.00263|.01281|.18768|.85094|.00183|.994|.989|
|21|.999962|.999964|.999854|.00264|.00258|.01311|.18246|.86051|.00176|.993|.994|
|22|.999956|.999958|.999831|.00291|.00285|.01342|.19654|.85021|.00190|.993|.995|
|23|.999950|.999953|.999809|.00324|.00316|.01405|.20891|.84160|.00200|.992|.991|
|24|.999954|.999956|.999823|.00348|.00338|.01481|.21873|.84976|.00195|.994|.989|
|25|.999959|.999962|.999846|.00378|.00367|.01515|.23023|.88211|.00179|.991|.993|
|26|.999955|.999958|.999828|.00412|.00401|.01540|.24713|.87979|.00187|.990|.991|
|27|.999951|.999955|.999814|.00448|.00435|.01584|.25991|.88818|.00191|.994|.995|
|28|.999951|.999954|.999812|.00471|.00457|.01579|.27121|.88771|.00196|.992|.994|
|29|.999956|.999959|.999832|.00457|.00443|.01541|.27103|.89333|.00185|.993|.995|
|30|.999961|.999964|.999851|.00466|.00452|.01542|.27718|.90370|.00173|.995|.995|
|31|.999966|.999968|.999871|.00517|.00502|.01591|.29932|.92053|.00160|.994|.993|

The contrast becomes larger and more heterogeneous toward late layers, but the induced U/R and U/M population ranking change remains tiny everywhere.

### Top/bottom rank-set stability

Means below are across 32 layers. Crossings are summed across layers and count both departures and arrivals. Exact per-layer values for every boundary are in `artifact_manifest.json`.

| Pair | Set | Mean overlap | Mean Jaccard | Total crossings |
|---|---|---:|---:|---:|
| U/R | top 5% / 10% / 20% | .9958 / .9963 / .9969 | .9917 / .9927 / .9938 | 164 / 290 / 492 |
| U/R | bottom 5% / 10% / 20% | .9916 / .9939 / .9955 | .9834 / .9879 / .9910 | 330 / 478 / 708 |
| U/M | top 5% / 10% / 20% | .9964 / .9964 / .9970 | .9929 / .9929 / .9940 | 140 / 282 / 476 |
| U/M | bottom 5% / 10% / 20% | .9932 / .9938 / .9955 | .9866 / .9876 / .9911 | 266 / 490 / 706 |
| R/M | top 5% / 10% / 20% | .9925 / .9929 / .9940 | .9851 / .9858 / .9880 | 296 / 562 / 950 |
| R/M | bottom 5% / 10% / 20% | .9851 / .9879 / .9911 | .9707 / .9761 / .9823 | 586 / 952 / 1406 |

Bottom sets are slightly less stable than top sets, but Uniform-versus-weighted overlaps remain above 99.1% at all aggregate boundaries.

### Timestep localization

Values are means across all 32 layers.

| t | U/R ρ | U/M ρ | R/M ρ | CV(qR) | CV(qM) | CV(RC) |
|---:|---:|---:|---:|---:|---:|---:|
|.05|.999934|.999949|.999770|.00598|.00525|.0854|
|.15|.999995|.999995|.999982|.00157|.00151|.0808|
|.25|.999998|.999998|.999992|.00098|.00096|.0860|
|.35|.999999|.999999|.999995|.00080|.00078|.0951|
|.45|.999999|.999999|.999998|.00049|.00048|.0667|
|.55|.999999|.999999|.999997|.00056|.00055|.0997|
|.65|1.000000|1.000000|.999999|.00039|.00039|.1037|
|.75|1.000000|1.000000|.999998|.00044|.00044|.1225|
|.85|.999873|.999863|.999490|.00811|.00812|.6986|
|.95|.999825|.999817|.999317|.00906|.00900|.6756|

The relative contrast is sharply more heterogeneous at `t=.85/.95`; it is not uniform over the denoising trajectory. Yet even there the weighted ABS rankings remain above `0.9998` correlated with Uniform. Low timestep-to-global `R_C` rank stability (`0.106` mean Spearman) also shows that the identity of contrast-sensitive neurons changes substantially with timestep.

### Comparison with EXP-005 per-weight sensitivity

EXP-005 reported mask XORs of about 0.1598% for Reveal versus Remain and about 0.1009%/0.0993% for Reveal/Remain versus Uniform. This spike does not construct neuron masks, so there is no exactly matched XOR comparison. As a rank-boundary diagnostic, average bottom-10% crossing fractions over the full neuron population are approximately 0.122% for U/R, 0.125% for U/M, and 0.242% for R/M. These are the same small order of magnitude, not a clear amplification at functional-neuron granularity.

## Interpretation

### Classification: LOCALIZED / UNSTABLE SIGNAL (Outcome C)

`d_C` is not a common scalar across neurons: aggregate `CV(R_C)` is 15–30%, and it increases in late layers. However, contrast is only roughly 1–1.6% of baseline sensitivity at the median, the actual symmetric weighted ABS rankings barely move, and the identities of high-`R_C` neurons are poorly stable across sequences and especially timesteps. The strongest heterogeneity is concentrated at the last two timesteps.

This supports a narrow statement: token-subset weighting creates a measurable neuron-specific contrast, but the present evidence does not show a stable functional-neuron importance ordering suitable for static structured pruning. It is neither complete scalar washout nor an emerged stable functional signal.

## Speculation

The late-timestep rise may reflect state-dependent specialization near heavily masked states, or simply sensitivity to the changing loss/activation geometry at those states. This experiment cannot distinguish those mechanisms and provides no pruning-performance evidence.

## Exactly one recommended next step

Run one held-out-state, analysis-only replication focused on whether late-timestep (`t=.85/.95`) `R_C` neuron rankings reproduce; do not proceed to ablation or structured pruning unless that stability gate passes.
