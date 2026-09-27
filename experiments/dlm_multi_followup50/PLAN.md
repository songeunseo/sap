# Multi allocation follow-up — frozen development screen

## Objective and hypothesis
Test two separate changes to the existing Multi allocator: document-bootstrap
mean ranks, and smaller allocation displacement. These may reduce sampling
instability or extrapolation error; improved task accuracy is not assumed.

## Fixed setup
- Reuse the verified 32 block probes (48/52%), dense readouts, both 128-state
  banks, native sparse-prefix Wanda activation/ranking, and original weights
  from `experiments/dlm_multiscale_ac50/output`.
- Objective remains A + (C1+C2+C4)/3. No new probes or teacher forwards.
- 32 block rates, tied across seven projections before integer rounding;
  exact 3,489,660,928 / 6,979,321,856 removed weights using the existing DP.
- Same model revision, BF16, seeds, 5-shot GSM8K, 256 steps/length/block length,
  temperature 0, strict exact match, identical development document IDs 0..99.
- Old sources and outputs are immutable. All new artifacts go in this folder.

## Candidates fixed before GPU execution
1. Multi-Bag: resample the eight calibration spans WITH replacement 200 times,
   NumPy default_rng seed 20260924. A draw is shared across all 32 blocks.
   Average each draw's rank-mapped rates; DO NOT rank the averaged ranks again.
   Chain/query/edge observations stay within their span. Reuse signed costs.
2. Multi-R2: s = .5 + .4 * (s_Multi - .5), ideal range 48–52%.
3. Multi-R3.5: s = .5 + .7 * (s_Multi - .5), ideal range 46.5–53.5%.
   Integer DP rounding may move physical rates by approximately one row count;
   report achieved rates, never claim exact per-block endpoints.

No Bag+radius combination, no scale coefficient changes, no 16-span bank,
no candidate-dependent activation refresh, no model search or new loss.
The reference is completed Multi (63/100); it is not regenerated.

## Evaluation and decision
For EVERY distinct candidate mask, measure both calibration and diagnostic
Multi/A/C1/C2/C4 and then all 100 development questions. Do not select candidates
using diagnostic loss or mini100 scores, and do not stop an arm for poor scores.
Report the three paired candidate-minus-Multi comparisons with exact McNemar
and Holm over all three once complete. Diagnostic input is reused development
evidence, not a new final test. No full GSM8K, confirmation, tuning, or automatic
follow-up. An identical physical mask may reuse only identity-verified outputs.

## Cost and execution
An uninterrupted run uses at most 3 * 256 = 768 candidate state forwards and
300 * 256 = 76,800 generation forwards, excluding initialization/identity checks.
Each generated request must use exactly the frozen 256 steps; each state stage
is capped at 128 forwards. No automatic retries. Explicitly resumed attempts
reuse checkpoints; any repeated work is separately included in cumulative
attempt costs and excess above these nominal ceilings. SIGTERM records costs
where possible; forced-kill receipts are marked incomplete and totals become
lower bounds, never silently reported as complete. Existing probes are reused,
not free end-to-end preparation. Cap numerical CPU threads at one.
Use previously authorized GPUs 1,2,3 only after idle checks; workers run in
tmux with per-document checkpoints, source hashes, exact mask identities,
process ownership records, stage/attempt costs and explicit completion receipts.
No GPU forward before independent audit and CPU validation pass.

## Verification gates
Recompute probe losses from raw saved readouts and reproduce the old Multi
allocation exactly. Test span pairing, ties/negative costs, mean budget,
radius scaling, exact integer budget, resume identity and controller failures.
An independent GPT-6 Luna/xhigh auditor checks source vs this plan and reviews
CPU evidence. Fix material findings and obtain a fresh review before launch.

## Authority
User approved implementation and this bounded mini100 follow-up; requested one
independent subagent. New experiment notes record setup before launch, then
observed progress. No mutation of the previous frozen experiment.
