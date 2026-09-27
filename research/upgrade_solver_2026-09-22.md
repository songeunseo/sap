# A+C allocator solver upgrade: measured exact-budget exchanges

Date: 2026-09-22  
Status: implementation design only. No GPU forward, new mask, benchmark, or
experiment was run for this note.

## Decision

Replace the current one-shot `48/52% probe -> rank -> rate mapping -> rounding`
path with a bounded hard-mask exchange search. The exchange is evaluated in the
current jointly sparse model, with the original weight values and the within-row
Wanda order frozen. Every proposed mask is an integer-count mask with the exact
global budget before it is scored. A measured full-bank A+C improvement is the
only acceptance signal.

The useful change is the evaluation context and the budget handling, rather than
a claim of a new optimizer. The current probes are collected in a Uniform50
sparse-prefix background and prune all seven projections of one block at 48% or
52%; they are not marginal costs at the current A or A+C mask. A probe-derived
rank therefore cannot be treated as a local objective for the final jointly
sparse mask. The new search may use a cheap proposal score, including a gradient
score, but it must measure and accept the actual hard candidate.

Keep the current scalar gold-vs-rest A+C readout for the first solver comparison.
The full centered-vocabulary-logit A+C readout is a separate objective arm. This
separation is necessary because changing the readout and the allocator together
would not identify which change affected the result.

## What the existing implementation leaves exposed

1. `experiments/dlm_context_response50/run.py` collects a block probe by
   restoring the seven dense projections in that block, applying a common 48%
   or 52% row count, and scoring all 80 pairs. Earlier blocks remain in their
   Uniform50 mask. The resulting finite difference is thus a whole-block
   perturbation in a particular sparse prefix, not a candidate exchange in the
   final A+C background. After probing, `rank_rates` discards the magnitude and
   maps the 32 signed costs to a fixed 45--55% interval.

2. `exact_row_counts` is an exact DP for a narrow weighted rounding problem. It
   selects floor-minus-one, floor, or floor-plus-one counts for 64
   block/input-width groups while minimizing rate-rounding error. It does not
   optimize the A+C loss. If it repairs a local proposal by changing another
   group, the resulting mask is a different exchange and must be evaluated as
   that mask. It cannot be reused unchanged for independent 224-projection
   variables.

3. `score` calls `margins` twice per pair and has no incumbent or candidate
   residual cache. A complete 80-pair candidate therefore costs
   `2 * 80 = 160` sparse state evaluations, even though the dense references are
   already cached. The current allocation is selected without a hard objective
   acceptance loop, a sealed search holdout, or a guarantee that a changed
   probe remains informative after earlier changes.

The stability audit is consistent with measuring the current background. Across
leave-one-span recomputations, AC rank correlations are high, but the response
component is much less aligned with A and the AC-vs-A ideal rates differ in
27/32 layers (mean absolute shift 0.786 percentage points, maximum 2.258 pp).
The response-direction sign agreement is only 66.7--85.2% for those changed
layers. This is descriptive in-calibration evidence, not held-out validation;
it supports refresh and common-bank acceptance, not a claim that AC is robust.

## Fixed problem and exact representation

For pair (i), let (e_b^i(M)) and (e_a^i(M)) be sparse minus dense
gold-vs-rest log-odds errors for the unrevealed and revealed-context states.
The first objective arm is the existing scalar

\[
 L_{AC}(M)=\operatorname{mean}_i\left[
   \tfrac12\left((e_b^i)^2+(e_a^i)^2\right)
   +\left(e_a^i-e_b^i\right)^2\right].
\]

This keeps the current pair bank and coefficient exactly fixed. It remains a
surrogate for downstream NELBO or GSM8K quality.

For an allocation unit (u), store an integer boundary (k_u), not only a
percentage. For a projection (p), (k_p) is the number removed from each
output row, and

\[
 n_p(k_p)=\operatorname{rows}_p k_p, \qquad
 0\leq k_p\leq\operatorname{width}_p.
\]

### 32-layer template lattice correction

The 32-layer arm must be anchored to a starting mask rather than replacing
its seven counts by a rounded common rate. Retain the starting mask's two
width-class counts and define a move for layer `l` by

```text
six 4096-input projections: k <- k + d_l
one 12288-input projection: k <- k + 3*d_l
```

The six 4096-input projections have 40,960 output rows in total and the
12288-input projection has 4,096 rows, so a template step changes exactly
`40,960*d_l + 4,096*(3*d_l) = 53,248*d_l` weights. Pairing `d_r=-d_d`
preserves the global budget exactly, including for a historical A or A+C
starting mask. The old 64-group rounding DP permits the two width-class
counts to differ independently by floor-minus-one/floor/plus-one; historical
A/AC masks therefore need not lie on the tied 32-layer lattice. This anchored
search preserves their residue and is a restricted warm-start family, not a
reproduction of the old A/AC family. An independent 64-group exact-count arm
can represent historical masks but enlarges the allowed decisions; it is not
automatically an identical candidate family. To retain the old family, use
the same 32-rate-to-DP mapping and evaluate every rounding-induced change.
The 224-projection arm uses the projection count formula above.

For the 32-layer arm, a unit option (q) must expand to a declared template
of all seven projection counts in that layer. Store its actual removed count
`n_layer(q)` after expansion. For the 224-projection arm, use the projection
formula above. A mask is feasible only if

\[
  \sum_u n_u(k_u)=K_{50}=3{,}489{,}660{,}928,
\]

and every count is within its declared 45--55% trust interval (or the exact
interval recorded for the arm). The mask hash, per-unit counts, and total count
are part of every candidate receipt.

At an exchange, one unit increases its removed boundary and another restores
the original dense weights at its boundary. For candidate (M'), require

\[
 n_d(k'_d)-n_d(k_d)
 = n_r(k_r)-n_r(k'_r) > 0.
\]

Equal percentage changes are not sufficient: projections have different row
counts and widths. If no one-donor/one-receiver pair has equal integer deltas,
enumerate the smallest declared two- or four-unit bundle whose deltas sum to
zero. Never round a percentage and silently repair an unrelated unit. A
candidate failing this integer test is infeasible and is not forwarded to the
model.

The first search should use the anchored 32-layer template arm to keep the
candidate space and interpretation controlled. Report the 64-group arm only as
the independent-count control when matching historical masks matters. Repeat
the selected backend at 224 projection units as a separate granularity/capacity
arm. The finer arm must not be described as a better solver merely because it
has more degrees of freedom.

## Recommended bounded search

Use the existing jointly sparse mask as the warm-start arm. For the clean
solver comparison, also initialize the same search at the native Uniform50
mask; this distinguishes a solver effect from path dependence of an A+C
warm-start. Both arms retain the same objective, mask family, rate interval,
pair bank, and exact budget.

The following is an accounting schedule, not an experiment result or a claim
that these caps are optimal:

* At each sweep, enumerate feasible one-exchange proposals at the current
  integer radius. In the 32-template arm a one-step proposal is `d_d=+1,
  d_r=-1` (or the reverse), so its net budget change is exactly zero; in the
  64-group and 224 arms, match the actual weighted count deltas or enumerate a
  small exact bundle. Draw a fixed mix of layer/type-stratified and seeded
  random proposals so a cheap proposer cannot permanently hide an unscored
  direction. Cap the poll at 32 proposals for the 32-unit arm. At 224 units,
  use the same cap and report the much larger unpolled space.
* Screen each proposal on a fixed document-stratified subset of 16 pairs. This
  costs `2 * 16 = 32` state evaluations per screened candidate. Fully score the
  four best screen candidates and the incumbent on all 80 pairs. Each full
  candidate contains `2 * 80 = 160` state evaluations; when the 32 screen
  states are cached, only 128 additional states are needed per finalist.
  Thus 32 proposals plus four finalists cost
  `32*32 + 4*128 = 1,536` state evaluations per sweep, plus the initial
  160-state incumbent evaluation. If screen outputs are not reused, the upper
  accounting is `32*32 + 4*160 = 1,664`. Screening is a priority mechanism,
  not a reduction in the full acceptance measurement.
* The screen only prioritizes. The full bank decides acceptance, and the
  incumbent is always included. If the best screen candidate fails on the full
  bank, try the next finalist; do not accept a screen-only improvement.
* Accept the best full candidate only if
  `L_AC(candidate) < L_AC(incumbent) - epsilon`, where `epsilon` is fixed from
  a repeated-incumbent numerical/stochastic measurement before search. With
  deterministic inference it is a numerical tolerance, not a post-hoc effect
  threshold. On acceptance, update the hard mask and refresh all proposal
  scores and residuals from this actual mask.
* Keep the radius after a successful move and optionally expand it one declared
  level. After a failed full poll, halve the radius. Stop at the minimum
  radius after a failed pair poll in the primary first pass. A two-exchange
  bundle poll is a separate follow-up only if the pair search repeatedly
  plateaus and its extra budget is approved; it is an interaction diagnostic,
  not a default beam.

The old saved probe may be used only to stratify the initial proposal pool. It
must not veto an unprobed exchange and must be invalidated as an acceptance
score after any accepted move. This matters because a sparse-context refresh
previously failed as a downstream method; the present design does not assume
that refresh helps, but it requires refresh for the candidate value to match
the model being optimized.

### Pseudocode

```text
freeze dense weights W0, row Wanda orders, pair bank, dense references
M <- exact starting mask (warm-start A+C or matched Uniform50)
assert count(M) == K50 and rates(M) in [45%, 55%]
cache L(M), A(M), C(M), per-pair residuals on the search bank
radius <- declared initial integer quantum

while evaluations < budget and radius >= minimum:
    pool <- feasible_exact_exchanges(M, radius, rate_bounds)
    pool <- fixed stratified/random poll plus optional gradient shortlist
    screened <- rank_by_same_objective_on_16_pair_subset(pool)
    finalists <- top four screened plus M

    for candidate in finalists:
        materialize original-weight hard mask candidate
        assert exact_count(candidate) == K50
        measure A, C, AC on all 80 pairs (160 state evaluations)

    best <- lowest full-bank AC candidate satisfying optional anchor rule
    if best.AC < M.AC - epsilon:
        M <- best
        refresh A/C residuals, boundary priorities, and affected-prefix caches
        radius <- min(next_radius(radius), maximum)
        continue

    if radius == minimum:
        break
    radius <- halve(radius)

evaluate M and the starting mask on sealed document holdout, then NELBO/GSM8K
```

The incumbent's per-pair vectors may be cached. A candidate that changes block
(b) may reuse a byte-identical prefix only through block (b-1); because the
model attention is bidirectional, recompute the entire changed suffix. Never
reuse changed-layer KV or freeze visible-token states. The current code has no
such runner, so full forwards are the safe baseline and state-evaluation counts
remain the audit quantity.

## Anchor-versus-response constraint (small, explicit extension)

The AC audit shows that its response component changes decisions but is less
stable than endpoint A. A useful framework extension is an explicit endpoint
anchor guard, evaluated only on the same full candidate bank:

\[
 A(M') \le A(M)+\tau_A, \qquad
 L_{AC}(M') < L_{AC}(M)-\epsilon.
\]

Here (	au_A) is declared before search from the incumbent repeatability
floor. The guard prevents an apparent response gain from paying an uncontrolled
endpoint regression. It is a constrained-objective variant, not a free solver
optimization; therefore compare it separately against unconstrained AC exchange
and report how many proposals violate the guard. If no endpoint regression is
observed, omit the guard from the primary arm rather than adding complexity.
Do not promote this constraint from the in-calibration audit to a scientific
claim without a held-out result.

## Full-vector centered-logit objective arm

After the scalar solver comparison, replace the readout while keeping the
exchange schedule and candidate family fixed. For centered vocabulary logits,

\[
 e_t=P(z_M^t-z_D^t), \quad P=I-\mathbf1\mathbf1^T/V,
\]

use

\[
 L_{vec}(M)=\operatorname{mean}_i[\tfrac12(\|e_b^i\|^2+\|e_a^i\|^2)
 +\lambda\|e_a^i-e_b^i\|^2].
\]

With post-final-norm hidden difference (delta h) and frozen head (U), the
exact squared centered-logit term is (delta h^TQdelta h), where
`Q = U.T @ P @ U` (divide by vocabulary size if the metric is a vocabulary
mean). Verify Q against direct centered logits on a small fixed batch before
using it. Q is about 64 MiB FP32 for width 4096 and forming it from the
126,464-row head is a substantial setup cost; a hidden-only forward is needed
to save head projection work. A hook that captures hidden states while still
computing logits does not provide that saving. This arm changes the objective,
not the solver, and must use the same exact-count hard acceptance.

## Competing alternative: hard-forward gradient/STE proposer

Use a gradient only to prioritize feasible exchanges. At the incumbent, form
the gradient of the same fixed A+C loss with respect to effective weights
`z_j = gate_j * W0_j`. For a boundary weight use the original dense value in
the proposal score, `s_j = W0_j * dL/dz_j`; multiplying by the already-zeroed
deployed weight would incorrectly erase restore utility. Project the proposed
gates or thresholds to the same exact integer count, then materialize the hard
mask and run the same full candidate acceptance. A soft loss or STE prediction
never accepts a candidate by itself.

This alternative adds one incumbent backward pass and its activation memory,
plus the same hard candidate forwards. Compare it under the same proposal cap,
screen subset, exact budget, starting mask, and full-bank acceptance. Record
forward/backward counts, wall time, peak memory, shortlist coverage, and the
number of predicted improvements rejected by hard evaluation. No quality or
global-optimum guarantee follows from the STE; it is a proposal mechanism.

## Minimal comparison that separates objective and solver

Run development only after implementation checks, with a fixed model revision,
state/pair bank, seed, 45--55% interval, exact 50% budget, and candidate budget.
Use document-held-out data for final selection evidence.

| Arm | Objective | Backend | Purpose |
|---|---|---|---|
| Existing rank allocator | scalar gold-vs-rest A+C | current probe/rank/DP | historical control |
| Hard exchange | scalar gold-vs-rest A+C | measured exact hard exchange | solver effect at fixed objective |
| Hard exchange + anchor guard | scalar A+C with explicit A guard | same exchange | constrained tradeoff diagnostic |
| Gradient/STE proposer | scalar gold-vs-rest A+C | same hard exchange/acceptance | proposal efficiency alternative |
| Vector readout arm | centered-logit A+C | same hard exchange | objective effect at fixed solver |

Start with the anchored 32-layer template units. Repeat only the selected
backend at 224 projection units as a capacity/granularity comparison. Include
the current A+C warm-start and matched Uniform50 cold-start as separate paths;
the old rank-generated A/AC masks remain external controls for the anchored
template family. Do not attribute warm-start differences to the readout.
Compare search-bank loss, sealed-bank loss, NELBO,
mask XOR, exact count, accepted moves, measured candidates, forward/backward
work, wall time, and peak memory. Mini100 remains development evidence.

## What the algorithm guarantees

With strict measured acceptance, every accepted mask is feasible and the
search-bank loss is monotone non-increasing up to the declared numerical
tolerance. A full poll that exhaustively measures every feasible exchange at a
fixed radius establishes no improving exchange in that measured neighborhood.
A sampled poll or screen establishes only no improvement among the proposed and
fully evaluated candidates. A finite candidate cap, repeated refresh, or STE
does not establish a global optimum. Global optimality would require exhaustive
enumeration of the finite mask space, which is not practical here.

For scale, all directed one-exchange pairs are 992 at 32 units and 49,952 at
224 units for one step size. At 80 pairs this is 158,720 and 7,992,320 state
evaluations respectively, before bundles. The bounded poll is therefore an
accounting choice, not an exact solver claim. The stopping condition guarantees
termination after the declared evaluation budget, or after a failed minimum-
radius pair poll in the primary pass. It guarantees exact budget and measured
monotonicity, not held-out downstream success, NELBO preservation, or a global
best mask.

## Decision and next step

Implement the exact-count candidate representation, hard evaluator, and receipt
checks first. Add current-mask residual refresh and a bounded screen/full
acceptance loop at 32 layers. Keep the scalar A+C objective unchanged for this
solver test. Only after that comparison should the endpoint anchor guard,
gradient proposer, 224-unit arm, or centered-logit Q readout be enabled as
separately labeled controls. Existing sparse-context refresh failures, weak
coverage evidence, and the failed wider J-based extrapolation remain reasons
to measure the assembled hard model; they are not grounds for claiming this
search backend is novel or globally optimal.

## Related local records

- `research/allocation_design_review_2026-09-18.md`
- `research/design_discrete_2026-09-18.md`
- `research/design_relaxation_2026-09-18.md`
- `research/design_code_audit_2026-09-18.md`
- `research/four_axis_saved_probe_audit_2026-09-18.json`
- `research/ac_probe_stability_2026-09-22.json`
- `experiments/dlm_context_response50/core.py`
- `experiments/dlm_context_response50/run.py`
