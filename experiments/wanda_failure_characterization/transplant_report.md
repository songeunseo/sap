# Perturbation transplant: location versus direction

## Artifact-supported facts

### Frozen perturbation sources

For every one of the existing 40 states, `delta30` was reconstructed as the FP32 difference between the block30.ff_out 50%-Wanda and same-path sham residual streams at the end of block 30. `delta31` was reconstructed analogously at the end of block 31 before final normalization. Both came from the existing masks and the original `[sham, 50%, 75%]` suffix path; no new mask or state was generated.

The tensors preserve sign, all 256 token positions, and all 4096 hidden features independently for each state. They are persisted in `transplant_deltas.pt` with hashes:

- delta30: `4772c278301c60c8bd92d8f48f89d738650f8ca1c9cee51d3c3e61c3c0d0239d`
- delta31: `4cca7947c00f67460b7c8ab41e9de3a6434b9a4769ad0e977601720b08c746e8`

### Native-injection reproduction gate

Each unscaled delta was added in FP32 to its dense same-path native residual and passed through the ordinary remaining dense network. Across both modules and all states:

| Reproduction quantity | max error | mean error |
|---|---:|---:|
| logits, max absolute | 0 | 0 |
| masked-token KL | 0 | 0 |
| official loss delta | 0 | 0 |

Residual injection is therefore exactly equivalent to the captured downstream effect of the native module-only Wanda perturbation in this execution path.

### Norm matching and numerical controls

For each state, natural relative sizes were `r30=||delta30||/||H30||` and `r31=||delta31||/||H31||`, and the fixed target was `tau=sqrt(r30*r31)`. A source direction `u_d` injected at location `l` was scaled to `u_d * tau * ||H_l||`.

Natural mean `r30/r31` was `0.05408/0.05520`. State-specific tau ranged from `0.04075` to `0.07148`, with median `0.05518`. All four cells achieved the same mean injected relative L2, about `0.0545`. Absolute deviation from tau had mean `1.79e-9`, p99 `7.45e-9`, and maximum `7.45e-9`.

Each location used a fixed dense three-row suffix: `[sham, D30, D31]`. Repeating both suffixes on the first state produced max hidden/logit differences of zero. Relative to an external batch-one dense forward, same-path sham KL had median/p99/max `0.000267/0.000487/0.000518`; loss drift had `0.001223/0.008590/0.008882`. Primary contrasts always use the same-path sham.

Model SHA-256 was unchanged before/after: `2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc`.

## Causal intervention results

### Matched-norm four-cell table

| Direction @ location | Mean KL | Median KL | loss delta | top-1 agreement | confidence MAE |
|---|---:|---:|---:|---:|---:|
| D30 @ L30 | 0.008528 | 0.0041 | 0.007168 | 0.9610 | 0.01212 |
| D30 @ L31 | 0.011243 | 0.0058 | 0.008004 | 0.9464 | 0.01515 |
| D31 @ L30 | 0.005992 | 0.0055 | 0.004474 | 0.9584 | 0.01441 |
| D31 @ L31 | 0.022614 | 0.0198 | 0.015385 | 0.9274 | 0.02546 |

The natural, unscaled anchors remain separate: block30.ff_out KL `0.008965`, block31.ff_out KL `0.022711`. Norm matching changed these native cells only slightly; causal conclusions below use the matched table.

### Factorial contrasts

| Contrast | Mean | Median | p10 | p90 | fraction positive |
|---|---:|---:|---:|---:|---:|
| location effect for D30 | 0.002715 | 0.001884 | 0.000014 | 0.004881 | 0.900 |
| location effect for D31 | 0.016622 | 0.014742 | 0.006818 | 0.026979 | 1.000 |
| direction effect at L30 | -0.002536 | 0.001336 | -0.012116 | 0.004353 | 0.575 |
| direction effect at L31 | 0.011371 | 0.011661 | -0.000395 | 0.028284 | 0.900 |
| location main effect | 0.009668 | 0.008185 | 0.002360 | 0.016028 | 0.975 |
| direction main effect | 0.004418 | 0.006871 | -0.005875 | 0.016639 | 0.775 |
| interaction | 0.013907 | 0.011987 | 0.002965 | 0.024864 | 0.975 |

Sequence-cluster bootstrap used seed `20260905`, 10,000 resamples of the eight sequences while retaining all five timesteps:

| Effect | Estimate | 95% cluster-bootstrap CI |
|---|---:|---:|
| location main | 0.009668 | [0.006637, 0.013055] |
| direction main | 0.004418 | [-0.004051, 0.010492] |
| interaction | 0.013907 | [0.008673, 0.020205] |

The direction main effect does not survive the sequence-cluster interval, whereas location and especially interaction do.

### Timestep and sequence stability

Interaction was positive at every timestep: `0.01096, 0.01605, 0.01483, 0.01634, 0.01136` for t=`0.1,0.3,0.5,0.7,0.9`. Location main effect was also positive at all five timesteps (`0.00613--0.01126`).

All eight sequence-level interaction means were positive (`0.00371--0.03179`). Seven sequences had a positive location main effect; the remaining one was still positive after averaging directions because its D31 location effect dominated. Sequence 14 produced a negative direction main effect (`-0.02209`), explaining why the direction-only bootstrap interval crosses zero, yet its interaction remained positive (`0.00371`). The interaction is not driven by one timestep or one sequence.

### Downstream magnitude

| Cell | final-hidden rel. energy | masked-logit rel. energy | logit RMS | median KL/final energy | median KL/logit energy |
|---|---:|---:|---:|---:|---:|
| D30 @ L30 | 0.002117 | 0.002473 | 0.1333 | 2.360 | 2.126 |
| D30 @ L31 | 0.001770 | 0.002342 | 0.1295 | 3.866 | 3.164 |
| D31 @ L30 | 0.004152 | 0.002976 | 0.1472 | 1.544 | 1.832 |
| D31 @ L31 | 0.002004 | 0.002803 | 0.1415 | 11.617 | 8.264 |

Across all 160 state/cell observations, KL versus final-hidden energy Spearman was `-0.024`; KL versus masked-logit energy was `0.255`. Identical injected norm therefore does not imply identical downstream norm, and downstream norm alone does not explain KL.

The most decisive matched comparison is D31 across locations. D31@L30 creates the largest final-hidden and logit perturbations, yet has the smallest mean KL. Moving the identical state-specific direction to L31 reduces final-hidden energy by roughly half but raises KL almost fourfold. Conversely, at L30 D31 is not consistently more damaging than D30, while at L31 it is substantially more damaging. This is a difference-in-differences effect, not an injected-magnitude artifact.

## Interpretation

**Outcome C — strong location × direction interaction.** Location has a positive main effect, but it is not sufficient: D30 becomes only modestly more harmful at L31. The D31 pattern becomes exceptionally harmful specifically at L31, while its direction advantage disappears or reverses at L30. The positive interaction is present in 39/40 states, all five timesteps, all eight sequence means, and its sequence-cluster bootstrap interval excludes zero.

Thus the natural terminal Wanda anomaly is not primarily a universal “last layer is fragile” effect and not a generally harmful D31 direction. It arises from compatibility between the D31 perturbation pattern and the representation/readout geometry at its native L31 location.

## Speculation

The compatibility may be token- and state-specific: the D31 pattern could align with prediction-relevant directions only in the block31 representation basis. This experiment does not identify those directions and makes no Jacobian, decision-boundary, or causal-feature claim.

## Exactly one recommended next diagnostic

Run a deterministic cross-state D31-direction swap at L31, using a preregistered cyclic mapping of the existing 40 frozen deltas and the same state-wise norm matching, to test whether the interaction requires each perturbation to remain paired with the state that generated it or reflects a state-general D31 directional family. Do not use random directions or modify Wanda.
