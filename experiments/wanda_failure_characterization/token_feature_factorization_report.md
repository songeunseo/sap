# Exhaustive D31 Token × Feature Factorization

## Scope and frozen inputs

This analysis used the existing 40 held-out states, state ordering, L31 injection path, `tau`, and stored `delta31[40,256,4096]`. No state, corruption, Wanda mask, weight, or perturbation direction was generated. The frozen D31 tensor hash was `4cca7947c00f67460b7c8ab41e9de3a6434b9a4769ad0e977601720b08c746e8`; the preregistered exhaustive 280-pair mapping hash was `34e18530a468a7bd94138f69845d7a932ac43a598b65047cbe1d6a030c464e85`.

For each receiver, all seven other sequences at the same timestep were donors. Each perturbation was factorized tokenwise as `delta[t,:] = a[t] u[t,:]`, with normalized allocation `A=a/||a||`. The four exact finite interventions were `NN=A_N U_N`, `NF=A_N U_F`, `FN=A_F U_N`, and `FF=A_F U_F`, each renormalized to `tau_receiver * ||H31_receiver||`.

## Artifact-supported gates

- Zero-row gate: all 10,240 rows were valid. Minimum row norm was 16.2387, median 35.4005, p01 18.3432; counts below `1e-12`, `1e-10`, `1e-8`, and `1e-6` were all zero. Fixed epsilon was `1e-12`.
- Same-path gate: batch-5 and established batch-3 NN logits and KL were bit-identical; batch-3 sham repeat logits were bit-identical. Batch-5 `[sham, NN, NF, FN, FF]` was therefore used.
- NN reproduction: versus the frozen matched D31@L31 artifact, maximum absolute errors were `3.73e-9` KL, `4.77e-7` loss delta, `4.66e-10` final-hidden relative energy, and `4.66e-10` masked-logit relative energy. NN had zero spread across its seven repetitions per receiver.
- Norm matching: mean absolute deviation from tau was `1.95e-9`, p99 `7.45e-9`, maximum `1.12e-8`.
- Repeatability: maximum hidden and logit errors were both zero.
- Model SHA-256 was unchanged: `2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc`.

The earlier artifact did not persist full native logits, so the logits gate used an exact rerun of its established batch-3 path; frozen metric reproduction was checked independently as listed above.

## Causal intervention results

Primary inference first averaged the seven donors within each of the 40 receiver states.

| Cell | Mean KL | Median | p10 | p90 | Retained vs NN (ratio of means) |
|---|---:|---:|---:|---:|---:|
| NN | 0.022614 | 0.019770 | 0.010846 | 0.033892 | 100.0% |
| NF | 0.008726 | 0.007435 | 0.002937 | 0.015238 | 38.6% |
| FN | 0.017420 | 0.016192 | 0.006956 | 0.029408 | 77.0% |
| FF | 0.008503 | 0.007926 | 0.001922 | 0.014743 | 37.6% |

Native feature directions retained most, but not all, native damage when paired with foreign token allocation (`FN`). Native token allocation did not rescue foreign feature directions (`NF` was close to `FF`). Crucially, neither hybrid reproduced NN.

| Receiver-level effect | Mean | Median | p10 / p90 | Positive receivers | Sequence-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Token main effect | 0.002708 | 0.002245 | -0.000710 / 0.006650 | 87.5% | [0.001968, 0.003797] |
| Feature main effect | 0.011403 | 0.008364 | 0.002228 / 0.019052 | 97.5% | [0.007244, 0.016905] |
| Token × feature interaction | 0.004970 | 0.003451 | -0.000347 / 0.011372 | 85.0% | [0.003279, 0.006760] |
| NativeExcess | 0.005045 | 0.003459 | 0.000459 / 0.011203 | 92.5% | [0.003408, 0.007040] |

Thus the feature-pattern effect was the largest separable component, while the positive interaction and NativeExcess establish additional joint compatibility.

## Donor, timestep, and sequence stability

Across donor sets, median donor CV was 0.155 for NF, 0.137 for FN, and 0.226 for FF. A foreign donor matched or exceeded native KL in only 1.1% of NF pairs, 12.9% of FN pairs, and 3.6% of FF pairs. This rules out dependence on the earlier single donor, although individual donor spread was larger for FF.

| Timestep | NN | NF | FN | FF | Token effect | Feature effect | Interaction | NativeExcess |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.01623 | 0.00428 | 0.00939 | 0.00246 | 0.00433 | 0.00944 | 0.00502 | 0.00684 |
| 0.3 | 0.02354 | 0.00521 | 0.01567 | 0.00406 | 0.00451 | 0.01497 | 0.00673 | 0.00787 |
| 0.5 | 0.02407 | 0.00842 | 0.01888 | 0.00975 | 0.00194 | 0.01239 | 0.00652 | 0.00520 |
| 0.7 | 0.02715 | 0.01283 | 0.02279 | 0.01370 | 0.00175 | 0.01171 | 0.00524 | 0.00436 |
| 0.9 | 0.02208 | 0.01289 | 0.02038 | 0.01254 | 0.00102 | 0.00851 | 0.00135 | 0.00170 |

All five timesteps retained the ordering `NN > FN > NF/FF` and positive NativeExcess. The joint excess weakened late but did not reverse. All eight sequence means had positive token effect, feature effect, interaction, and NativeExcess; sequence 12 was strongest, but no conclusion depends on it.

## Downstream magnitude controls

Mean final-hidden relative energy was 0.002004 (NN), 0.002064 (NF), 0.002168 (FN), and 0.002123 (FF). Mean masked-logit relative energy was 0.002803, 0.003567, 0.002464, and 0.002829 respectively. Hence NF had much lower KL than NN despite *higher* final-hidden and logit energy; FF had essentially matched logit energy but only 37.6% retained KL. KL differences were only weakly related to final-energy differences (Spearman NN-minus-condition: -0.14 to -0.39). FN's lower logit energy explained some pairwise variation (Spearman 0.385), but it cannot explain the NF/FF collapse or the overall interaction.

## Statistical associations

Token-allocation cosine was associated with FF KL (Spearman 0.682) and more moderately with FN KL (0.442), while token rank correlation was essentially unrelated to FN KL (0.020). Receiver-weighted feature-pattern cosine was associated with NF KL (0.516) and FF KL (0.599). Similarity was negatively associated with the corresponding main effects (-0.299 allocation; -0.198 feature), consistent with less-compatible donors producing larger native advantages. These are descriptive associations, not harmful-subspace metrics.

## Interpretation

**Outcome C — Joint token × feature compatibility.** Hidden-feature pattern carries the larger independent share: retaining native directions alone preserves 77.0% of native KL, whereas retaining native allocation alone preserves 38.6%. Nevertheless, NN remains 29.8% more damaging than FN and 159% more damaging than NF; the interaction and NativeExcess are positive across receivers, sequences, and timesteps with cluster-bootstrap intervals excluding zero. Similar or larger downstream perturbation magnitudes in the hybrids rule out a simple propagation-norm explanation.

The supported mechanism is therefore not a scalar token allocation alone or a state-general feature direction alone. D31 failure is strongest when receiver-native token allocation and token-conditioned hidden-feature directions occur together.

## Speculation boundary

These interventions establish compatibility of two observed factors, but do not identify a Jacobian, decision boundary, semantic token role, or correct pruning modification. They also do not show that either factor can be estimated before pruning.

## Exactly one recommended next diagnostic

Run a preregistered **token-row pairing control** using the same frozen perturbations: preserve native allocation and the native set of feature-direction rows, but deterministically re-pair direction rows with token positions within each state. This would test whether the interaction requires the exact token-to-direction correspondence rather than only the two marginal native components. Do not launch it automatically.
