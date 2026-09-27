# Exact Token-Row Pairing in State-Specific D31 Failure

## Artifact-supported facts

The experiment reused the frozen 40 states, native `delta31[40,256,4096]`, receiver-specific `tau`, and L31 dense injection path. The D31 tensor SHA-256 was `4cca7947c00f67460b7c8ab41e9de3a6434b9a4769ad0e977601720b08c746e8`; the frozen permutation-manifest hash was `a2af5d051c66637b4a03356d902f8881b1dd7bf6fac6a550b08145214b5f665f`.

All 255 nonzero cyclic shifts were evaluated for all 40 states: 10,200 unrestricted state-shift conditions. The masked/unmasked-preserving control contained 10,060 unique state-permutations after preregistered deduplication. No result-dependent permutation was selected.

The 10,240 native token rows passed the direction gate: minimum norm 16.2387, p01 18.3432, median 35.4005, and zero rows below `1e-12`, `1e-10`, `1e-8`, or `1e-6`. The fixed epsilon was `1e-12`.

Batch-5 and the established batch-3 path matched exactly for NN logits, final hidden, KL, and loss. NN reproduced the frozen matched D31@L31 metrics within maximum errors of `3.73e-9` KL, `4.77e-7` loss, `4.66e-10` final-hidden energy, and `4.66e-10` masked-logit energy. Model weight SHA-256 was unchanged: `2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc`.

Pre-normalization cell-norm spread was at most `1.19e-7`. Achieved relative-norm mismatch was at most `1.12e-8` in both families.

## Causal intervention results

Results were first averaged over shifts within each receiver; the inferential sample is 40 receivers clustered into eight sequences.

| Family | NN | NS | SN | SS | R_NS | R_SN | R_SS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unrestricted cyclic | 0.022614 | 0.009818 | 0.017976 | 0.010575 | 43.4% | 79.5% | 46.8% |
| Mask-class preserving | 0.022614 | 0.008688 | 0.019924 | 0.008345 | 38.4% | 88.1% | 36.9% |

`SN` keeps feature rows at their native receiver-token positions and reassigns only allocation. It retained far more native damage than `NS`, which retains native allocation but moves feature rows. Moving intact native `(A,U)` pairs together (`SS`) destroyed most damage, even while preserving their internal pairing.

| Family/effect | Mean | Median | p10 / p90 | Positive receivers | Sequence-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Unrestricted PairingEffect | 0.002697 | 0.002266 | -0.000431 / 0.005928 | 82.5% | [0.001538, 0.003994] |
| Unrestricted PositionEffect | 0.012039 | 0.009103 | 0.001579 / 0.023166 | 97.5% | [0.007787, 0.017096] |
| Unrestricted FeaturePlacementEffect | 0.008157 | 0.006006 | 0.001425 / 0.014293 | 95.0% | [0.005042, 0.012288] |
| Class-preserving PairingEffect | 0.001173 | 0.001371 | -0.000629 / 0.002791 | 72.5% | [0.000733, 0.001781] |
| Class-preserving PositionEffect | 0.014268 | 0.011320 | 0.003634 / 0.023768 | 100% | [0.009956, 0.019799] |
| Class-preserving FeaturePlacementEffect | 0.011236 | 0.008666 | 0.003304 / 0.019219 | 97.5% | [0.007217, 0.016721] |

Across shifts, NN exceeded NS in 98.2%, SN in 86.1%, and SS in 93.6% of unrestricted comparisons. Under class preservation the corresponding rates were 98.0%, 78.5%, and 98.3%.

## Timestep and sequence stability

Unrestricted PositionEffect was positive at every timestep: 0.01264, 0.01789, 0.01264, 0.01058, and 0.00644 from `t=0.1` through `0.9`. Class-preserving values were 0.01255, 0.01908, 0.01704, 0.01520, and 0.00746. FeaturePlacementEffect was also positive at every timestep in both families.

PairingEffect remained positive in every timestep average but weakened at `t=0.9` (0.000674 unrestricted; 0.000630 class-preserving). All eight sequence means had positive PositionEffect, FeaturePlacementEffect, and PairingEffect in both families. No post-hoc sequence removal was used.

## Masked/unmasked-preserving control

Class preservation did not restore SS damage. Instead, SS retention fell from 46.8% to 36.9%, while PositionEffect increased from 0.01204 to 0.01427. Therefore the exact-position effect is not explained by feature rows merely crossing the masked/unmasked boundary.

SN retention increased from 79.5% to 88.1% under class preservation. Coarse masked status explains part of allocation reassignment damage, but native feature-row placement remains the dominant factor. PairingEffect survived class preservation with a positive cluster CI, although its magnitude was small relative to PositionEffect.

## Downstream-magnitude controls

For unrestricted permutations, mean NN final-hidden relative energy was 0.002004. NS, SN, and SS had larger energy by `5.34e-5`, `1.67e-4`, and `1.12e-4`, respectively, despite their lower KL. NN and SS masked-logit energy were essentially matched: NN exceeded SS by only `1.08e-6`, yet SS retained 46.8% of KL. NS had substantially larger masked-logit energy than NN while retaining only 43.4% of KL.

The class-preserving pattern was the same: every permuted cell had slightly larger final-hidden energy, and SS had larger masked-logit energy than NN while retaining only 36.9% of KL. KL reductions therefore cannot be reduced to attenuation. Rank associations between energy differences and KL differences were weak or contrary to a magnitude explanation, except that SN logit-energy variation explained some unrestricted pairwise variation (Spearman 0.409).

## Shift-distance association

Damage dropped immediately at circular distance one: NN 0.02261 versus NS 0.01314, SN 0.01765, and SS 0.01452. PositionEffect was 0.00809 at distance one and remained broadly large across the full range (for example 0.01147 at distance 128). There was no gradual recovery toward native damage at long distances; exact position, rather than a tunable distance scale, is the salient descriptive result.

## Mechanistic interpretation

**Outcome B — Receiver-token / feature-direction alignment is dominant.**

The decisive comparison is `SN >> NS`: preserving each feature direction at the exact receiver token that generated it retains 79.5% of native damage, and 88.1% when masked status is also preserved. Native allocation alone retains only 38–43%. Conversely, shifting intact `(allocation,direction)` pairs together loses more than half the damage. PositionEffect and FeaturePlacementEffect are large, stable, survive class preservation, and have cluster-bootstrap intervals excluding zero.

There is also a smaller real allocation–direction pairing effect, but it is secondary: its class-preserving mean is 0.00117 versus a 0.01427 PositionEffect, and SN retains most native damage. The evidence therefore does not support Outcome D's requirement that neither broken cell retain native damage.

## Speculation boundary

The intervention establishes exact receiver-token/feature-row compatibility, not its semantic or differential mechanism. It does not identify a Jacobian, decision boundary, harmful subspace, token-importance score, or correct Wanda modification. The results also do not establish that this compatibility can be estimated before pruning.

## Exactly one recommended next diagnostic

Perform a frozen **within-token feature-direction rotation control** that preserves every token's native allocation and receiver position while replacing only its feature direction with a deterministic direction drawn from the same state's native direction set under norm and mask-class controls. This would isolate whether native per-token feature direction—not merely keeping any native direction at the token—is required. Do not launch it automatically.
