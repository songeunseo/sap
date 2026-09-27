# Multi / A-only / matched Uniform: separate 100-question validation

## Objective and hypothesis
Check whether the original Multi advantage survives a separate set of 100
questions. Primary scientific contrast: Multi minus A-only, which tests adding
multiscale response terms with the same bank and allocation backend. Secondary
contrasts: Multi minus Uniform and A-only minus Uniform. Report all three paired
exact McNemar comparisons and Holm correction over this fixed family of three.
No superiority is assumed.

## Fixed candidates and evaluation
Reuse the existing physical Multi and A masks from dlm_multiscale_ac50 and the
native sparse-prefix Uniform mask named by that run's config. These scored
63,60,54 on development IDs0–99. A-only means the multiscale-bank A model, not
historical legacy A. Uniform means the matched native model, not the historical
62-point dense-calibration baseline. Verify each sparse model SHA after applying
its existing mask to fresh dense weights. Exact global sparsity50% for all.

Use the 100 confirmation requests already frozen on2026-09-22. The original rule
selected the smallest100 SHA256(multiscale-ac-confirm:20260922:<doc_id>) among
200..1318, then sorted by original ID. Verify this rule independently and verify
no overlap with development IDs0–99. Keep the frozen prompt/token hashes and
original5-shot/256steps/256generation/block length/temperature0/strict-match.
Historical project use of these questions is not ruled out. This is held aside
from the current Multi development selection, not a pristine new benchmark.

The initial tentative100–199 suggestion is replaced before any execution by
this previously frozen sample. The original plan's conditional competitor
selection is superseded by the user's newly approved fixed Multi/A/Uniform
comparison. No score-dependent arm or sample selection occurs.

## Budget and execution
Exactly3x100 generations; nominal maximum76,800 model forwards. No new probes,
calibration, mask optimization or diagnostic-bank forwards. All three candidates
finish regardless of intermediate scores. No automatic retries or further runs.
Explicit resumptions reuse per-document checkpoints and account for retry costs.
Record wall time, forward counts, memory and incomplete forced-kill attempts.
Run in dedicated tmux using previously authorized idle GPUs1,2,3. Do not useGPU0.

## Gates
Before launch: frozen source hashes, all physical mask hashes and counts,
independent reconstruction of selected IDs, same candidate model identities,
CPU tests for sample identity, stale/tampered results and interrupted resume.
Require a matching successful CPU receipt in launcher, controller and worker.
The existing process-ownership controller is reused through an explicit adapter.
All new outputs are isolated here; previous experiments remain read-only.

## Reporting and decision
Report this100-question set separately from the reused development results.
Any combined200 score is descriptive and not the primary validation comparison.
A small non-significant gap does not establish equivalence or superiority.
Do not extend to full GSM8K or change masks based on observed validation scores.
