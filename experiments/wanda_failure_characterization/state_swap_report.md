# Is the D31 harmful perturbation state-specific or state-general?

## Artifact-supported facts

### Frozen states, deltas, and donor mappings

This diagnostic reused the existing 40-state ordering, held-out manifest, state-specific transplant magnitudes, and stored `delta31[40,256,4096]`. The delta artifact and its manifest were verified before model execution. The exact delta31 SHA-256 is:

`4cca7947c00f67460b7c8ab41e9de3a6434b9a4769ad0e977601720b08c746e8`.

No Wanda mask was loaded, generated, or applied. The dense L31 sham residual was reconstructed with the previous batch-three row-wise `ff_out` numerical path using three no-mask rows.

Both donor mappings were persisted before foreign outcomes were evaluated. Mapping SHA-256 is `8f1f1ea1271c83569ca5c85f2c194672841d21d77b913034fa57d70349fe259b`.

- Primary cyclic: `donor(i)=(i+1) mod 40`.
- Same-timestep foreign: the next sequence in sorted cyclic sequence order at the identical timestep.

The frozen state ordering is timestep-major. Consequently, cyclic and same-timestep mappings select the same donor for 35/40 receivers; they differ at the five timestep boundaries. This overlap is a preregistered mapping consequence, not result-dependent selection, but limits the secondary control's independence.

The mapping artifact records receiver/donor state index, sequence, timestep, p-mask, and masked-position count. Donor token positions and hidden features were transplanted unchanged—no token alignment or semantic remapping was performed.

### Native reproduction and numerical gates

The native matched D31@L31 condition was reproduced before foreign evaluation. Against the previous transplant artifact:

| Quantity | maximum error | mean error |
|---|---:|---:|
| KL | 3.73e-9 | 2.33e-10 |
| loss delta | 4.77e-7 | 6.56e-8 |
| final-hidden relative energy | 0 | 0 |
| masked-logit relative energy | 0 | 0 |
| logit RMS | 0 | 0 |

Native and foreign directions were independently normalized and scaled to the receiver's stored tau. Achieved relative-norm error had mean `1.86e-9`, p99 `7.45e-9`, and maximum `7.45e-9`.

Each evaluation used `[sham, native, foreign]` through the same L31 final-normalization/LM-head path. Repeated hidden states and logits had maximum difference zero for both mapping runs. External batch-one sham drift was retained only as control: KL median/p99/max `0.000232/0.000464/0.000518`; primary comparisons use the same-path sham.

Model SHA-256 was unchanged before and after: `2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc`.

## Causal intervention results

### Native versus foreign KL

| Condition | Mean KL | Retained damage versus native |
|---|---:|---:|
| native D31 | 0.022614 | 1.000 |
| cyclic foreign | 0.008730 | 0.386 |
| same-timestep foreign | 0.008664 | 0.383 |

The per-state foreign/native KL ratio had median `0.331`, p10 `0.097`, and p90 `0.678` for both mappings. The mean per-state ratio was `0.418` cyclic and `0.415` same-timestep; this differs from the ratio of aggregate means because states have different native KL scales.

### Pairing advantage

`PairingAdvantage = KL_native - KL_foreign`:

| Foreign condition | Mean | Median | p10 | p90 | fraction positive |
|---|---:|---:|---:|---:|---:|
| cyclic | 0.013883 | 0.011706 | 0.004027 | 0.025091 | 0.975 |
| same-timestep | 0.013950 | 0.011758 | 0.004447 | 0.025091 | 0.975 |

The fixed-seed (`20260905`) 10,000-replicate receiver-sequence cluster bootstrap gave:

- cyclic mean advantage 95% CI: `[0.008301, 0.020845]`
- same-timestep mean advantage 95% CI: `[0.008337, 0.020898]`

Both exclude zero while retaining all five timesteps of every resampled receiver sequence.

### Timestep stability

| t | native KL | cyclic KL | same-timestep KL | cyclic advantage | same-timestep advantage | native > foreign |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.016225 | 0.001910 | 0.001897 | 0.014315 | 0.014328 | 8/8, 8/8 |
| 0.3 | 0.023540 | 0.003731 | 0.003711 | 0.019809 | 0.019829 | 8/8, 8/8 |
| 0.5 | 0.024073 | 0.013334 | 0.013300 | 0.010739 | 0.010774 | 7/8, 7/8 |
| 0.7 | 0.027153 | 0.013069 | 0.013040 | 0.014083 | 0.014112 | 8/8, 8/8 |
| 0.9 | 0.022077 | 0.011606 | 0.011371 | 0.010471 | 0.010706 | 8/8, 8/8 |

The effect exists across the full denoising trajectory. Foreign damage increases at later timesteps, but does not approach native damage.

### Sequence stability

All eight receiver-sequence mean pairing advantages were positive. Cyclic advantages ranged from `0.002028` to `0.034182`; same-timestep advantages ranged from `0.002028` to `0.034182`, with only the mapping-boundary sequence showing a meaningful difference between controls. Sequence 14 has the smallest effect because its foreign KL remains high (`0.020449` versus native `0.022477`), but it does not reverse the aggregate result.

### Downstream perturbation magnitude

| Quantity | native | cyclic foreign | same-timestep foreign |
|---|---:|---:|---:|
| final-hidden relative energy | 0.002004 | 0.002108 | 0.002108 |
| masked-logit relative energy | 0.002803 | 0.002815 | 0.002802 |
| logit RMS | 0.14151 | 0.14351 | 0.14296 |
| loss delta | 0.01539 | 0.00609 | 0.00609 |
| top-1 agreement | 0.92744 | 0.95365 | 0.95503 |
| confidence MAE | 0.02546 | 0.01444 | 0.01444 |

Foreign perturbations do not decay more. Their final-hidden energy and logit RMS are slightly larger, while their KL is approximately 61% lower. Pairing advantage versus native-minus-foreign final-energy difference has Spearman `-0.342/-0.318` for cyclic/same-timestep; versus logit-energy difference it is only `0.119/0.116`. Propagation magnitude therefore does not explain the pairing advantage.

### Existing-direction cosine description

Flattened native/donor delta cosine averaged `0.161` for cyclic donors and `0.163` for same-timestep donors. Across all existing direction pairs, same-timestep cross-sequence cosine averaged `0.163`, while different-timestep cosine averaged `0.183`. Thus same timestep does not create a visibly tighter family under raw cosine.

Donor/native cosine is moderately associated with foreign KL (`rho=0.514` cyclic, `0.566` same-timestep) but weakly negatively associated with pairing advantage (`-0.233/-0.274`). This is descriptive only and was not used for donor selection.

## Interpretation

**Outcome A — strong state-specific alignment.** A foreign D31 perturbation with identical receiver-relative norm retains only about 38% of native damage. Pairing advantage is positive in 39/40 states, at every timestep, and for every receiver-sequence mean; both sequence-cluster bootstrap intervals exclude zero. Same-timestep foreign directions lose essentially the same amount of damage, so denoising stage alone does not explain the effect. Final-hidden and logit perturbation magnitudes are matched or slightly larger for foreign directions, ruling out downstream attenuation as the primary explanation.

The previous D31×L31 interaction therefore depends strongly on compatibility between each state's own pruning perturbation and its own L31 representation/prediction state. The evidence supports state-conditioned directional/readout sensitivity, not a state-general D31 harmful family or a timestep-conditioned family.

The 35/40 overlap between primary and secondary donor mappings reduces the amount of independent evidence supplied by the secondary control. It does not weaken the direct native-versus-same-timestep result, but it prevents treating the two foreign estimates as independent replications.

## Speculation

State specificity could reside in the perturbation's token-position pattern, its hidden-feature pattern, or their joint structure. Raw flattened cosine does not identify which component matters, and no harmful-subspace or decision-boundary claim is made.

## Exactly one recommended next diagnostic

Run one deterministic token/feature factorization transplant on the existing native–same-timestep donor pairs: preserve the receiver-native tokenwise perturbation norms while substituting the donor's normalized hidden-feature pattern, and compare it with the complementary donor-token/native-feature construction at the same total norm. This would isolate whether state specificity primarily lives in token-position allocation, hidden-feature direction, or their interaction. Do not modify Wanda or run a downstream benchmark.
