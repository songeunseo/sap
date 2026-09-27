# Writing handoff — 2026-09-27

## Status

Writing review is complete for the direct JSON evidence currently available. `paper-v1.md` and `claim-map-v1.md` revise the manuscript around the measured primary result and all three fixed contrasts. The independent numeric audit passed: primary scores, gains/losses, percentage effects, exact McNemar p-values, Holm adjustment, and 10,000-draw paired bootstrap intervals recomputed from the four per-question prediction files match both `report.json` and `closeout.json` within 1e-10.

## Read and applied

- `/home/tmluser1/sap/AGENTS.md`
- Original role: `/home/tmluser1/.codex/attachments/07806206-ed65-4894-8b99-f0eb2a0d114d/writing.md`
- Project role: `/home/tmluser1/sap/writing/dlm_pruning/roles/writing.md`
- `/home/tmluser1/sap/writing/dlm_pruning/contract-v1.json`
- `/home/tmluser1/.codex/attachments/07806206-ed65-4894-8b99-f0eb2a0d114d/writing-guidelines.md`
- `/home/tmluser1/.codex/attachments/07806206-ed65-4894-8b99-f0eb2a0d114d/writing-style-guide.md`
- `/home/tmluser1/.codex/attachments/0e37c59b-ba95-4b1a-854b-8efec235c12c/Writing a good scientific paper.md`
- Obsidian `Research/DLM-Pruning/Research-State.md` and sync status; connection verified as connected.

The original Depth-AR examples, provisional-number workflow, four-page/three-hour target, hidden provisional values, Overleaf pushes, and automatic follow-up rules were not applied because the project role explicitly adapts them away for this DLM update.

## Direct sources and provenance

- Closeout: `/home/tmluser1/sap/writing/dlm_pruning/closeout.json` (sha256 `8a076550ce16f2e81d4a8a2e6b5a72b85d8f8adcf887765c15c12261a2ece2a3`).
- Experiment report: `/home/tmluser1/sap/experiments/dlm_crosschain_control50/output/report.json` (sha256 `0857b0f7931f5d019f282c5e6984159b014aca0b91747ff8e9bf077bb53c0a10`).
- Contract: `/home/tmluser1/sap/writing/dlm_pruning/contract-v1.json` (current sha256 `4f861672a53c8ea173be4e15793a71857d84b33df6ae267555d591d0ae82c9fb`).
- Per-question source: `experiments/dlm_crosschain_control50/output/gsm8k/{A,Multi,Cross,CrossMatched}/predictions.json`.
- Primary split: `experiments/dlm_crosschain_control50/output/request_split.json`.

The direct key mapping is in `source-key-inventory.json`; the independent computation is reproducible with `recompute_primary.py` and recorded in `numeric-audit.json`.

## Scientific interpretation carried into v1

- Multi−A is measured as −0.894 pp (64/74; exact p=.443715; Holm p=1; unadjusted bootstrap [−2.949, 1.162] pp).
- Multi−Cross is measured as −0.089 pp (74/75; exact p=1; Holm p=1; [−2.234, 1.966] pp).
- Multi−CrossMatched is measured as +0.179 pp (67/65; exact p=.930684; Holm p=1; [−1.787, 2.145] pp).
- The manuscript says the tested added benefit and pairing advantage are unestablished. It makes no equivalence, universal failure, causal mechanism, practical superiority, or generalization claim.
- Fresh lower response losses for Cross and CrossMatched are reported as diagnostics only, without causal attribution.
- The constructed response example is confined to Appendix A and explicitly labelled as non-observed.

## Scope and unresolved integration dependency

Only the requested paths were written: `paper-v1.md`, `claim-map-v1.md`, and `review_2026-09-27/writing/` (`recompute_primary.py`, `numeric-audit.json`, `source-key-inventory.json`, this handoff). No model was loaded, no GPU or experiment was run, no child agent was spawned, and no source experiment, closeout, v0, template, old claim map, or Obsidian note was modified.

At handoff time, the experiment agent's promised files were not yet present:

- `review_2026-09-27/experiment/verification.json`
- `review_2026-09-27/experiment/report.md`
- `review_2026-09-27/experiment/primary-effects.png`
- `review_2026-09-27/experiment/primary-effects.pdf`
- `review_2026-09-27/experiment/plotted-data.json`

`paper-v1.md` already points Figure 1 to the requested PNG path and describes its measured provenance. Master should perform final integration when those files arrive, verify that the figure matches `numeric-audit.json`, and retain the explicit dependency if the experiment handoff remains unavailable. The constructed Figure 2 is already available and remains separate from observed evidence.

## Master review corrections applied

- Replaced "preregistered" with "prespecified" because the project has frozen plans but no verified public registry.
- Corrected the discussion direction: the source-direction contrasts are Multi−Cross=−0.089 pp and Multi−CrossMatched=+0.179 pp; equivalently, Cross is +0.089 pp and CrossMatched is −0.179 pp relative to Multi.
- Split cost accounting: 8.236 wall-clock hours is the current run only; 1,351,936 forward calls include the current run plus the preserved first attempt.
- Replaced the phrase "negative-to-inconclusive" and the claim that the experiment establishes the distinction with wording that the mathematical objective distinguishes endpoint and response errors, while the completed comparison did not establish incremental benefit.
- Added consistent captions for Table 1 (primary scores), Table 2 (fixed contrasts), Table 3 (historical screens), and Table 4 (fresh diagnostics); Figure 1 now points to Table 2.
