# CGQ-SparseGPT calibration-seed variance audit

## Question

Can the observed `CGQ - Plain = -12/1,267` WinoGrande result be explained by
the random choice of the 16 calibration states, and is running Plain on two or
three additional draws enough to establish a calibration-noise band?

## Existing result

- Plain and CGQ used the **same** 16 cached corrupted states, with a shared
  state digest, and the same WinoGrande documents and evaluation seeds
  ([downstream report](results/report.md),
  [runner](../../cgq_sparsegpt_downstream.py#L245-L316)).
- The observed outcome was Plain `875/1,267`, CGQ `863/1,267`, with 61
  Plain-wrong/CGQ-correct and 73 Plain-correct/CGQ-wrong examples
  ([downstream report](results/report.md#paired-outcomes)).
- One calibration bundle contains four clean WikiText-2 spans, each corrupted
  at four fixed timesteps. The corruption seed is deterministic per
  `(timestep, sequence)` ([state builder](../../cgq_sparsegpt.py#L40-L59)).

## Primary-source evidence

Ji et al. repeat calibration sampling 20 times with different seeds and report
average downstream performance specifically to mitigate sampling randomness
(Section 3.1, p. 4). Their Figure 3 reports the standard deviation over those
20 seeds and finds that it decreases as calibration size increases (Section
3.3, p. 5). This establishes calibration sampling as a variance component that
should be measured, but it does **not** estimate its size here: their main
analysis uses autoregressive DCLM-7B, 128 sequences of 2,048 tokens, and
Wanda/DSnoT/OWL; its reported score averages seven tasks, including
WinoGrande. It is not a LLaDA, SparseGPT, 16-state, or WinoGrande-only result.
The paper therefore does not support the specific claims that dLLM pruning is
unusually seed-sensitive or that a `12/1,267` swing is plausible in this setup
([Ji et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/2ede933e10afa991a10b6f36b6522129-Paper-Conference.pdf)).

The original SparseGPT paper reports five random calibration draws for a 50%
sparse OPT-2.7B model and a raw-WikiText2 perplexity standard deviation of
`0.075` (Appendix A, p. 12). That is evidence of robustness in a different
128-by-2,048-token autoregressive setting, not evidence about this experiment's
WinoGrande variance ([Frantar & Alistarh, ICML
2023](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf)).

## Statistical interpretation

For method `m`, calibration bundle `s`, and WinoGrande item `i`, let

\[
A_{m,s}=\frac{1}{1267}\sum_i Y_{m,s,i}.
\]

- `sd_s(A_Plain,s)` measures **marginal Plain instability** across calibration
  draws.
- `Delta_CGQ,s = A_CGQ,s - A_Plain,s` and
  `Delta_Inv,s = A_Inv,s - A_Plain,s` measure the method effects conditional on
  the same draw. Their across-draw spread measures the relevant
  **method-by-calibration interaction**.
- `D_s = A_Inv,s - A_CGQ,s` is the direct direction-reversal contrast.

Because Plain and CGQ shared the current draw, the `-12` result is attributable
to the methods' different pruning decisions **for that fixed bundle**; it is
not caused by a calibration imbalance between them. Calibration randomness may
still make that conditional effect fail to generalize to a new bundle.

Plain-only repeats cannot estimate this generalization uncertainty. In
particular,

\[
\operatorname{Var}(A_{CGQ}-A_{Plain})
=\operatorname{Var}(A_{CGQ})+\operatorname{Var}(A_{Plain})
-2\operatorname{Cov}(A_{CGQ},A_{Plain}),
\]

and a Plain-only run measures neither the CGQ variance nor the covariance. A
large Plain range can coexist with a stable paired effect, and a small Plain
range can coexist with an unstable CGQ effect.

McNemar's test on the 1,267 paired predictions measures item-level uncertainty
for two fixed sparse models. It does not measure the outer uncertainty induced
by drawing a different calibration bundle. Treating all item predictions from
multiple bundles as independent would likewise be pseudoreplication.

## Minimal useful design

Use calibration bundle as the blocking variable:

1. Keep the existing seed-0 bundle as the **discovery** block. Because its
   result motivated the inverse hypothesis, do not present it as independent
   confirmation.
2. Predeclare three new independent calibration bundles. Save each bundle's
   clean token IDs, corrupted states/masks, and digest; a seed alone is not a
   sufficient audit record.
3. On every new bundle, prune all three methods from the same dense revision:
   Plain, frozen original CGQ, and energy-normalized inverse CGQ. Keep model,
   timesteps, sequence length, sparsity, pruning range, damping, block size,
   and the complete WinoGrande evaluation protocol fixed.
4. Report, per bundle, `A_m,s`, `Delta_CGQ,s`, `Delta_Inv,s`, and `D_s`, in
   both percentage points and numbers of correct items. Also retain paired
   document outcomes.
5. Use the three new bundles only as a **pilot**: inspect sign consistency and
   the range/SD of paired effects. Three bundles cannot provide a stable
   variance estimate or a useful significance test. If a confirmatory claim is
   needed, use the pilot SD of the paired contrast to plan more independent
   bundles rather than calling the three-run range a 95% noise band.

If compute only permits Plain on two or three new draws, those runs are still a
cheap gross-instability diagnostic, but they answer only whether absolute Plain
accuracy moves. They cannot determine whether the confidence direction effect
is calibration noise.

For inference conditional on the fixed WinoGrande validation set, the
across-bundle distribution of paired contrasts is the relevant target. To
generalize over both calibration bundles and WinoGrande items, a later analysis
must account for the crossed seed and item structure; neither a seed-only SD
nor a single-seed McNemar test covers both sources.

## Decision

Run paired Plain/Original-CGQ/Normalized-Inverse blocks on new calibration
bundles. Plain-only replication is optional screening, not the noise band for
the method contrast. The existing `-12` result remains a valid conditional
observation, while its calibration-draw generality is currently unknown.

Obsidian `Research-State.md` could not be consulted because no Obsidian tool is
available in this session; this note uses the repository's checked-in CGQ
records instead.
