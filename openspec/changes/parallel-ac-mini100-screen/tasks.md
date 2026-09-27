# Tasks

## 1. Screen manifest and legacy import

- [x] 1.1 Add the isolated `experiments/dlm_ac_screen50/` package and CPU-only command wrapper; verify imports/help do not load the 8B model or initialize/query CUDA with mocked forbidden calls.
- [x] 1.2 Implement the frozen ten-arm manifest, source/model/ranking/request identities, seeds, numerical tolerances and separate GPU execution metadata; verify schema tests reject missing controls, extra arms and scientific-setting changes after start.
- [ ] 1.3 Implement legacy validation/import receipts for 64 probes, five allocations, three references and existing document checkpoints; verify current correct counts reproduce 54/55/61 and partial Short/Multi 33/31 out of 56 only if receipts still match, while corrupt/mismatched fixtures fail with field-specific errors.
- [ ] 1.4 Document preparation, legacy ownership and reference naming in the new README; verify CPU prepare/dry-run prints computed missing-work counts and leaves protected old code/config/completed-output hashes unchanged.

- [x] 1.5 Add source-ledger and objective-contract metadata from design §0 without rewriting imported receipts; verify each arm declares edge sets/weights, readout/reduction versions, hypothesis/control and defaults, and that DKD is documented as a distinct deferred candidate rather than an executable Vector alias.
- [x] 1.6 Add CPU theory/import compatibility checks using the frozen Multi bank/core: verify 128 states, seeds, exact stored probabilities, zero-change retention, upfront queries, C1/C2/C4/Path/All definitions, the 7/3 toy loss, population-variance and graph bounds, and a small finite-enumeration cross-bias identity. Document that Path has unequal node exposure and the legacy sign statistic is not the theoretical bound's event; keep legacy source hashes unchanged.

## 2. Four-state bank and scalar objectives

- [x] 2.1 Implement the seeded 8-span × 5-quartet bank and separately frozen diagnostic bank from design section 3; verify the exact split seeds and independently derived query/base/group RNG streams, reproducibility, disjoint reveal groups, fixed masked queries, all four state relationships, group-size caps and degenerate-quartet handling on CPU.
- [x] 2.2 Implement Square-A and Square-AC with query/quartet/span means; verify the exact [00,10,01,11] order and four edge means, constant-error and [1,-1,-1,1] examples (A=1,C=4,AC=5), zero loss for identical dense/sparse responses, equal endpoint exposure and rejection of invalid/nonfinite readouts.
- [x] 2.3 Add per-state atomic scalar stores bound to bank/teacher/mask identities; verify crash/resume and stale-teacher rejection with temporary-file tests, and document why new Square results are not a legacy-pair-only ablation.

- [x] 2.4 Expose averaged squared residual modes and squared mixed residual from already collected Square outputs; verify A=u²+a²+b²+c², C=2a²+2b²+4c² and mixed squared residual=16c², no extra loss penalty, path-total telescoping, and K=4 abstract graph equivalence without treating nested and branched banks as identical. Document random-group/count confounds.

## 3. Vector readout and storage

- [x] 3.1 Implement design §0.5 full-vocabulary A/C with raw FP32 emitted-logit storage and FP64 centering/residual reductions, 1/V normalization and equal pair means; verify full/chunked CPU equivalence at atol=1e-10, rtol=1e-9, independent endpoint common-shift invariance, pairwise-margin identity, nonuniform query counts, and the scalar-blind wrong-token redistribution example. Include a fixture where per-chunk centering incorrectly erases between-chunk differences.
- [x] 3.2 Implement immutable dense-query-logit teacher cache and complete-pair metric checkpoints, using the original calibration pair/query manifest; verify interruption before/after commit does not skip half a pair or mix teacher identities.
- [x] 3.3 Implement the exact Vector diagnostic recipe in design §4: eight clean spans from the 40-state source, ten stored calibration mask probabilities per span, CPU mask seed 20260928+i and legacy reveal seed 20260929; verify 80 pairs/160 states, query alignment, split token-interval separation, no resampling on invalid counts and deterministic regeneration without consulting predictions. Preserve the original calibration pair/query bytes.
- [x] 3.4 Add disk/RAM estimation, storage reservation checks and chunk-size accounting; verify insufficient space fails before partial teacher production and document actual storage for both teacher banks plus the no-hidden-MSE/no-top-k fallback rule.

## 4. Shared probes and final candidate construction

- [ ] 4.1 Implement family-specific GPU worker jobs around existing native ranking/model loaders with bounded block shards; verify tiny CPU fixture outputs reproduce reference scalar/vector metrics and that Square/Vector A and AC use one shared collection per sparse condition.
- [x] 4.2 Compute signed 48/52 marginal costs and call the unchanged rank mapping/exact-quota backend; verify negative costs, ties, the larger-cost-to-less-pruning sign, actual parameter-weighted counts and exact 3,489,660,928 budget on saved-shape fixtures.
- [ ] 4.3 Materialize and hash final masks from original weights, preserving row rankings and surviving values; verify 224-matrix identity/count checks and deduplication only for equal physical masks under equal requests.
- [x] 4.4 Document probe and final-diagnostic job contracts and compute formulas; verify generated job manifests account for 10,240 probe state forwards per new family and the full 11,360-state nominal family budget from design §8, separately from generation, cache reuse, failures and smoke work.

- [ ] 4.5 Add actual full-mask calibration/diagnostic scoring and new-family Uniform calibration references; verify final losses are obtained from assembled masks rather than independent-probe extrapolation, decomposition diagnostics match the measured total, and deduplication preserves bank-specific weights. Document the rank map/rounding DP as a heuristic, not a functional optimizer.

## 5. Bounded hard exchange

- [x] 5.1 Implement two-block tied quota increments from the legacy AC anchor with d=41 and exact shape-derived quanta; verify every feasible move conserves weight count, changes only the selected blocks, preserves rounding offsets and respects per-matrix bounds.
- [ ] 5.2 Implement deterministic frozen-cost proposal ordering, eight proposals per round, three rounds and physical-mask cache keys; verify donor/receiver gain sign DeltaN*(g_receiver-g_donor), shortlist order/ties, no duplicate new evaluations, and no dependence on GSM8K or diagnostic values; document that frozen costs/caps are bounded-screen defaults rather than guarantees from the source papers.
- [ ] 5.3 Implement full-bank candidate scoring, twice-measured initial reproducibility, epsilon fixed from the first finite L0 before its repeat and measured-only acceptance; verify synthetic loss fixtures cover improvement, tie, no feasible move, nonfinite failure and budget termination.
- [ ] 5.4 Persist atomic incumbent/round decisions and cumulative budgets; verify kill/resume at each acceptance boundary cannot apply an exchange twice or reset the 24-candidate budget, and document limits of the fixed neighborhood.

- [ ] 5.5 Add Exchange anchor/final scalar diagnostics on the frozen Vector diagnostic inputs without changing the calibration objective; verify consistent FP32 scalar extraction from shared raw teacher logits, single-producer fallback when no teacher cache exists, unchanged-mask reuse, and at most 4,640 standalone state forwards before smoke/failures. Diagnostics must not affect the search trajectory.

## 6. Parallel execution, stop and status

- [x] 6.1 Build the persisted dependency job graph and fair ready-job scheduler for any explicit GPU list; verify fake-worker tests for one/two/four devices, no duplicate device assignment, shared teacher single ownership and overlap of resumed evaluation with new-family probes.
- [ ] 6.2 Integrate screen/legacy-root leases and original worker entry points without editing frozen modules; verify competing owners are rejected and valid existing document checkpoints are skipped rather than rewritten.
- [x] 6.3 Implement tmux launch, requested-device occupancy checks, process-group ownership and signal cleanup; verify mocked launch refuses occupied devices, dry-run never polls/spawns, and local fake-worker stop/failure tests leave unrelated processes untouched.
- [ ] 6.4 Implement CPU-only status/JSON and measured per-stage ETA with log/PID/device links; verify partial, failed, interrupted and completed fixtures render distinct counters and unknown ETA before measurements.
- [ ] 6.5 Add local research lifecycle records, remote-note receipt/pending fields and documented Obsidian launch/reconciliation procedure; verify planning does not create running records and simulated MCP verification failure remains separate from a successful write.
- [x] 6.6 Document exact prepare/validate/dry-run/launch/watch/stop/resume commands and thread caps; verify CPU command examples run as documented, bounded smoke is capped at 32 state forwards, no automatic full/confirmation phase is scheduled, and a combined implementation-and-run request does not require redundant permission after CPU checks.

## 7. Native mini100 evaluation and reports

- [ ] 7.1 Reuse native request validation, generation and strict-match grading with atomic document checkpoints; verify legacy raw texts regrade identically and changed prompt/token/protocol/model identity is rejected.
- [x] 7.2 Implement matched-control tables and seven predeclared paired comparisons with exact McNemar/Holm; verify hand-counted gain/loss fixtures, incomplete-family behavior, aliases, and common-ID subsets without extrapolation.
- [x] 7.3 Implement per-stage cost summaries and development report sections with raw prediction links; verify scalar/vector objectives are not ranked together, reused work is separated from new cost, and missing timing stays unknown.
- [x] 7.4 Document historical Uniform families, EM-versus-reasoning limits and future independent confirmation; verify the sample report labels MS-A/legacy A and row-wise/layer-global Uniform distinctly, does not equate legacy sign-flip statistics with the theoretical bound, and does not claim that Holm removes prior development-set reuse.

- [x] 7.5 Add theory-linked family diagnostics and the hypothesis/control/limitation matrix to the report; verify Multi's component losses, Square's squared modes, Vector's normalization, Exchange coverage/termination and calibration-versus-diagnostic provenance on fixed fixtures. Preserve negative results and label cross-family gains as end-to-end rather than isolated causal effects.

## 8. CPU integration and execution handoff

- [x] 8.1 Run all new CPU tests plus relevant unchanged legacy CPU regression tests under one-thread limits and CUDA hidden; verify scientific formulas, exact budgets, legacy fingerprints and interrupted/resumed outcomes remain consistent.
- [ ] 8.2 Exercise a full tiny CPU/fake-worker screen through prepare, partial evaluation, stop, resume and reporting; verify resumed outputs equal uninterrupted outputs, all ten arms are represented and protected old source/config hashes remain unchanged.
- [x] 8.3 Generate a real-artifact CPU preparation/validation/dry-run receipt and implementation handoff with measured cache sizes, inferred missing work and future bounded GPU smoke procedure; verify it explicitly records no GPU/model execution and no real benchmark results from implementation alone.

This task list covers implementation readiness; the current document-revision request does not execute any task or model. Actual GPU smoke, tmux launch and screen completion require a user request that includes execution. A combined implementation-and-run request supplies that scope without a redundant second approval after the CPU handoff. Do not mark real experiment completion from CPU checks. Launch must record actual GPUs and the running research note before model work.


## 2026-09-26 audited execution handoff

58 CPU tests pass; evidence: `experiments/dlm_ac_screen50/output/audits/cpu_regression.json`.
The checked subset has direct CPU/math/integrity/report evidence. Unchecked items
are not declared fully accepted merely because implementation exists; expanded
fault-injection or actual GPU/full-candidate verification may still be pending.
The user explicitly authorized execution on GPUs2,3. The immutable prelaunch
revision and serialized scientific-contract comparison are preserved under
`output/audits/`; source/config/completed legacy document hashes are unchanged.
GPU smoke and complete ten-arm results are separate from CPU readiness.
