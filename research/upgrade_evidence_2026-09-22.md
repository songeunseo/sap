# A+C pruning idea: independent evidence audit and upgrade

Date: 2026-09-22  
Scope: read-only audit of the 2026-09-18 proposal and the completed
`experiments/dlm_context_response50` mini experiment. No model forward, GPU
run, mask generation, or Obsidian write was performed.

## Verdict

A+C remains a development candidate, but the current result does not justify
selecting it as the method. The strongest positive observation is within one
controlled native mask family: A+C gives 61/100 GSM8K versus 55/100 for A-only
and 54/100 for native Uniform, with 9 A-only wrong → A+C correct and 3
regressions. The paired A+C versus A test is inconclusive (exact McNemar
`p=0.145996`), and neither candidate has held-out NELBO evidence.

The first upgrade should preserve the current natural nested pair bank and
compare all arms on that same bank, with a fixed solver and independent
quality split. A matched-count bank is a separate diagnostic: it must be
applied to both A and A+C, not substituted only for the A+C arm. The question
is whether C adds quality beyond endpoint A, rather than whether a new pair
construction changes the result.

## Five pieces of evidence that affect method choice

1. **A+C changes allocation decisions under a fixed support family.** All
   candidates use the same model revision, exact 50% budget, native
   sparse-prefix Wanda support and frozen activation rankings. A versus A+C
   changes 27/32 ideal block rates. The frozen CPU reanalysis reproduces all
   33 receipt hashes and gives A-versus-A+C cost Spearman `0.942815`, mean
   absolute ideal-rate shift `0.7863` percentage points, and maximum shift
   `2.2581` points. C-versus-A cost Spearman is only `0.153592`, so C changes
   the ranking signal, but the resulting perturbation is small. These are
   32 block-level probe costs expanded to 224 projection masks by the grouped
   rate/row-count construction; they are not 224 independently measured
   response costs.

2. **The downstream direction is suggestive but weak.** On reused GSM8K
   mini-100, A+C has 9 rescued and 3 regressed examples relative to A (net
   +6), with `p=0.145996`. Against native Uniform, A+C has 13 rescued and 6
   regressed examples (`p=0.167068`, Holm-adjusted `p=0.334136`). This keeps C
   alive as a hypothesis but does not establish a method effect.

3. **The surrogate improves, but the improvement is not an independent C
   attribution.** Dense-teacher log-odds distortion on the same 80 pairs is
   Uniform `A=0.8240, C=0.3786, A+C=1.2026`, A `0.6653, 0.3763, 1.0416`, and
   A+C `0.6590, 0.3670, 1.0260`. The CPU audit’s 6.55% figure is only an
   additive accounting of the measured C component’s decrease in the composite
   Uniform→A+C loss. It is **not** a causal percentage contribution: A+C
   changes the masks and therefore changes both measured A and C. These values
   are also used for mask selection on the same eight calibration spans, so
   they demonstrate internal consistency, not prediction of GSM8K or NELBO.

4. **Functional allocation is worth testing in this setting, but provenance
   does not transfer automatically.** Historical projection-capacity
   allocation reached full GSM8K `250/1319` versus `139/1319` for its
   historical Uniform and had lower held-out DLM KL. This supports measuring
   functional effects rather than assuming reconstruction is sufficient.
   Historical Uniform is nevertheless `62/100` on the cached mini while native
   Uniform is `54/100`, and 217/224 projection masks differ between the two
   construction families.

5. **The target phenomenon is credible while the pruning claim remains open.**
   Small DLM work reports causal use of visible context in both directions, and
   diffusion-compression work reports cases where sensitivity matching adds to
   output matching. These results motivate a context-response target, but do
   not establish that scalar gold-logodds response in LLaDA-8B is task-relevant
   after static pruning. Project role audits also warn that local proxy gains
   can fail functionally: 40.52% of bundle-state cases with both local
   reconstruction costs improved still had worse functional KL.

The decision from these facts is narrow: retain C as an ablation to test, but
do not call the current result evidence that A+C is preferred.

## Confounds and counterexamples

- **Natural pairs change progress state.** `make_pairs` reveals gold tokens
  from masked positions, so the after input has fewer masked positions and
  more visible evidence. The scored `query` indices are the after-state
  remaining masked positions and are used for both before and after logits;
  thus the scored query positions are held fixed within each pair. The
  revealed positions are no longer scored. Reveal count is 13 in 64/80 pairs
  but varies from 1 to 12 in the other 16 because it is capped by one quarter
  of the masked positions at low mask rates. C therefore mixes context use
  with progress/confidence response and a changing mask count.
- **C is algebraically tied to A.** For endpoint errors `e0,e1`,
  `A=0.5(||e0||²+||e1||²)` and `C=||e1-e0||²`. C reweights endpoint error
  directions; it is not an independent teacher. Endpoint preservation can
  already preserve response, while a response term can also hurt under limited
  capacity.
- **The readout is incomplete.** Gold-vs-rest log-odds ignores wrong-vs-wrong
  ordering and much of the vocabulary distribution. A candidate can improve
  the recorded scalar while damaging a distinction used by denoising.
- **Local probe-to-final extrapolation is a live failure mode.** Costs are
  measured at 48% and 52% in a frozen Uniform50 background, rank-normalized,
  and mapped to 45–55% rates. Rank mapping discards magnitude; 32 block costs
  are then expanded across the 224 projection masks and combined in a jointly
  sparse model. The saved response audit found that analogous local linear
  predictions did not match finite A/AC outcomes.
- **Calibration and selection are not independent.** The same 80 pairs from
  eight spans determine the probe ranks and final allocations. The stability
  audit is useful but in-calibration only: leave-one-span-out ranks remain
  correlated with the full-bank ranks (roughly `0.923–0.987`), yet the
  resulting leave-one-span masks were not evaluated. This is not held-out
  mask validation.
- **Support and allocator effects are bounded by design.** Frozen Wanda
  support isolates between-projection allocation, but cannot show that C fixes
  within-row ranking or transfers to another support family. Historical
  Uniform62 and native Uniform54 are distinct controls.
- **Calibration loss and capability can disagree.** Prior role work observed a
  lower held-out KL for one allocator while another had higher mini GSM8K
  accuracy. Lower A+C distortion is therefore not sufficient as the success
  criterion.
- **No 65%-level A+C comparison exists.** The prior Role65 results do not
  measure A+C, so they cannot support a forecast that A+C will behave like
  Role at 65% or establish any direct Role-versus-A+C relationship.

## Most realistic way A+C can be wrong

The likeliest failure is that C mostly measures the response to revealing more
of the sequence, not the value of a particular context token. That signal can
reduce the pair-bank surrogate while shifting block rates in a way that fails
after the 224-mask sparse model is assembled. The +6 mini net can then be
sampling noise or an endpoint-allocation effect. This explains how lower
calibration A+C, changed allocation, and a non-significant mini gain can
coexist without a real C contribution.

## Two failures the next design must avoid

1. **Do not silently replace the natural nested bank with a matched-count bank
   in only one arm.** The first solver/allocator comparison should keep the
   natural nested pair bank fixed for Uniform, A and A+C, so all arms see the
   same endpoints, query construction and progress distribution. A
   matched-count bank is a separate diagnostic factorial: both A and A+C must
   use it, and the original natural bank remains the primary deployment-like
   condition.

2. **Do not claim an arbitrary shuffled pair is a semantic null.** Count-fixed
   pairs can still differ in which context positions are revealed and which
   evidence is available. Cross-sequence shuffling can also break alignment
   between query positions, endpoints and usable context. A valid broken-pair
   null requires an explicitly constructed query-aligned bank with preserved
   endpoint/query marginals; until that bank exists, there is no clean null for
   variable-query-count pairs. Keep pair distortion diagnostic and validate
   hard masks on disjoint documents with held-out NELBO.

## Minimum high-information comparison

The primary screen should retain the natural nested bank and compare these
same-bank arms under one fixed solver, exact budget, support family and model:

1. native Uniform;
2. endpoint-only A;
3. natural nested A+C; and
4. a strong full-vocabulary endpoint-fidelity control.

Use disjoint calibration and validation documents, one predeclared sparsity
level, one fixed coefficient (`lambda=1` only if retained before seeing the
result), identical candidate proposals and matched calibration compute. Report
full-vocabulary output KL or centered-logit error, held-out WikiText NELBO,
GSM8K only as a later capability check, and wall time/forward count.

Then run the matched-count diagnostic as a separate 2×2 comparison: A versus
A+C, each on both the natural bank and a newly constructed query-aligned
matched-count bank. If a broken-pair condition is added, construct it to
preserve exact query and endpoint alignment; do not call a generic
cross-sequence shuffle causal isolation.

Interpretation is straightforward. A+C must improve held-out NELBO over A on
the same natural bank before C is considered useful. A matched-count result can
then show sensitivity to progress confounding, but cannot by itself prove that
the effect is semantic. If pair distortion improves while NELBO does not, stop
treating the surrogate as a method signal. If the full-vocabulary control wins,
the scalar gold-logodds readout is the wrong upgrade target.

## Repetition versus a possible new contribution

Already proposed or already present:

- paired context-response preservation and the A+C decomposition;
- natural gold-reveal pairs, scalar log-odds readout, rank-based 48/52→45–55%
  allocation, and the A-only control;
- exact-budget static allocation over frozen Wanda support;
- direct functional KL/capacity allocation as a baseline/backend, including
  the historical capacity experiment;
- generic exact-budget search and endpoint/output matching from the reviewed
  pruning and compression literature.

These ingredients are useful, but their combination is not established as
novel merely because it is applied to LLaDA.

A defensible new contribution could be demonstrated if the natural-bank,
same-solver comparison shows that conditional response adds held-out NELBO or
cost efficiency over A, Uniform and full-vocabulary fidelity, and the separate
query-aligned matched-count diagnostic clarifies what portion is progress
response. That would be a narrow empirical framework contribution. It would
still not establish AR-versus-DLM specificity, an optimal optimizer, or
universal superiority without further controlled evidence.

## Decision

Do not promote A+C to the main method from the existing mini result. Preserve
it as a preregistered development candidate. First compare the natural nested
bank with a fixed solver and independent NELBO; only then use a constructed,
query-aligned matched-count bank as a separate diagnostic. Decide from hard-mask
behavior and held-out quality, while treating additive A/C loss accounting as
descriptive rather than causal.
