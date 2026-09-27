# Claim–evidence map v1

Research question: under one frozen DLM checkpoint, calibration bank, Wanda ranking, local allocator, and exact 50% sparsity budget, does coupling endpoint errors across paired context reveals improve downstream GSM8K accuracy?

## Interpretation rule

The manuscript separates algebraic properties, measured outcomes, diagnostic associations, and unresolved hypotheses. A nonsignificant contrast is reported with its estimate and interval; it is not described as equivalence. A lower calibration loss is not treated as a causal explanation for a task result. The constructed residual example is confined to Appendix A and is explicitly marked as non-observed.

## Claims and evidence

| ID | Manuscript claim | Status | Direct evidence and key | Permitted scope |
|---|---|---|---|---|
| C0 | A response error for a fixed query is the difference of sparse and dense endpoint errors. | Established algebraically | `paper-v1.md`, Sec. 3.1; construction follows `figures/response-example.json` and `make_response_figure.py` | Definition and identity only; no task guarantee |
| C1 | Natural-chain Multi adds finite response penalties at scales 1, 2, and 4 to endpoint A. | Established setup | `config.json#/plan/edges`, `config.json#/plan/bank/weighting`; `paper-v1.md`, Sec. 3.2 | This implemented objective and weighting |
| C2 | Four arms share the source bank, native sparse-prefix Wanda ranking, and exact 50% budget. | Verified setup | `config.json#/pruning`, `config.json#/crosschain_control/arms`; `closeout.json#/sources`; `audit_pass.json` | One frozen experiment |
| C3 | On primary1119, A=646, Multi=636, Cross=637, CrossMatched=634. | Measured result | `report.json#/scores/primary_remaining_1119`; `closeout.json#/full/scores/primary_remaining_1119`; independent `numeric-audit.json#/scores_primary_remaining_1119` | Frozen sample only |
| C4 | Multi−A is −10 net answers, −0.893655 pp, gains/losses 64/74, exact p=.443715, Holm p=1, bootstrap [−2.949062, 1.161752] pp. | Measured result | `report.json#/comparisons/primary_remaining_1119/Multi-A`; `closeout.json#/full/comparisons/primary_remaining_1119/Multi-A`; `numeric-audit.json#/comparisons_primary_remaining_1119/Multi-A` | No added benefit established; no equivalence claim |
| C5 | Multi−Cross is −1 net answer, −0.089366 pp, gains/losses 74/75, exact p=1, Holm p=1, bootstrap [−2.234138, 1.966041] pp. | Measured result | `report.json#/comparisons/primary_remaining_1119/Multi-Cross`; `closeout.json#/full/comparisons/primary_remaining_1119/Multi-Cross`; `numeric-audit.json#/comparisons_primary_remaining_1119/Multi-Cross` | Tested cross control only |
| C6 | Multi−CrossMatched is +2 net answers, +0.178731 pp, gains/losses 67/65, exact p=.930684, Holm p=1, bootstrap [−1.787310, 2.144772] pp. | Measured result | `report.json#/comparisons/primary_remaining_1119/Multi-CrossMatched`; `closeout.json#/full/comparisons/primary_remaining_1119/Multi-CrossMatched`; `numeric-audit.json#/comparisons_primary_remaining_1119/Multi-CrossMatched` | Tested scale-matched cross control only |
| C7 | The three fixed contrasts do not demonstrate an added downstream benefit for Multi or a natural-chain advantage. | Interpretation bounded by C4–C6 | All three primary comparison keys and measured-effects figure; experiment handoff `verification.json` when present | One bank, allocator, model, sparsity, benchmark |
| C8 | Primary nonsignificance does not establish equivalence. | Statistical interpretation | C4–C6 intervals include both directions; `contract-v1.json#/quality_gates` | Do not infer equivalence |
| C9 | Full1319 scores are A=764, Multi=758, Cross=757, CrossMatched=755. | Descriptive result | `report.json#/scores/full_1319`; `closeout.json#/full/scores/full_1319` | Descriptive full sample; not a new inferential sample |
| C10 | Previously seen 200 scores are 118, 122, 120, 121 in A, Multi, Cross, CrossMatched order. | Descriptive result | `report.json#/scores/previously_seen_200`; `closeout.json#/full/scores/previously_seen_200` | Historical exposure is retained; no independent confirmation |
| C11 | Separate100 Multi−A is +1 point with 6/5 gains/losses and exact/Holm p=1; Multi and A exceed Uniform descriptively. | Historical screen | `closeout.json#/mini/scores`, `closeout.json#/mini/paired`, `closeout.json#/mini/development`; source `experiments/dlm_multi_validation10050/output/report.json` | Context only; does not override primary1119 |
| C12 | Fresh diagnostics measure endpoint A, natural C, cross C, query CE, and response-sign flips on eight spans. | Measured diagnostic | `report.json#/diagnostics/fresh`; `closeout.json#/full/fresh_diagnostics`; span uncertainty `report.json#/fresh_diagnostic_span_comparisons` | Calibration diagnostics, not task performance |
| C13 | Cross and CrossMatched have lower fresh response losses than Multi; Multi has higher fresh A and query CE. | Measured association | `report.json#/diagnostics/fresh`; `closeout.json#/full/fresh_diagnostics` | No causal mechanism or mediation claim |
| C14 | Diagnostic differences do not explain GSM8K differences. | Interpretation | C12–C13 plus primary C4–C6; direct comparison in `paper-v1.md`, Sec. 5.3–6 | Descriptive mismatch only |
| C15 | The readout is gold-versus-rest log-odds, so it does not track redistribution among incorrect alternatives. | Method limitation | `config.json#/plan/readout`; `paper-v1.md`, Sec. 3.1 | Readout scope |
| C16 | Gold-context chains may differ from generated denoising contexts. | Limitation | `config.json#/plan/bank/context_rule`; `closeout.json#/full/limitations` | No generated-context generalization claim |
| C17 | The local block-probe allocator is heuristic and changes multiple blocks together. | Method limitation | `config.json#/pruning/mapping`, `config.json#/pruning/probe_rates`; `paper-v1.md`, Sec. 3.3 | Result constrains criterion plus allocator |
| C18 | One checkpoint, one bank, one sparsity level, and one benchmark do not establish generalization. | Scope limitation | `config.json#/model`, `config.json#/banks`, `config.json#/pruning/sparsity`, `config.json#/evaluation/task`; `closeout.json#/full/limitations` | No replication/generalization claim |
| C19 | Recorded cost is 1,351,936 forward calls and 8.236 wall-clock hours, excluding reused artifacts. | Measured accounting | `closeout.json#/full/costs`, `closeout.json#/full/wall_hours_current_run`; `report.json#/costs` | Not an end-to-end calibration cost |
| C20 | The current experiment did not compare competitive DLM baselines or runtime speed. | Scope limitation | `closeout.json#/full/limitations`; `config.json#/evaluation` | No practical superiority claim |

## Figure provenance

- Figure 1: measured primary effects, generated by the experiment agent from verified per-question outputs. Expected source: `review_2026-09-27/experiment/primary-effects.png`, `primary-effects.pdf`, and `plotted-data.json`; verification: `review_2026-09-27/experiment/verification.json`.
- Figure 2: constructed response residuals, `figures/response-example.png` and `figures/response-example.json`; no model output. It remains in Appendix A and must not be used as observed evidence.

## Independent numeric audit

`review_2026-09-27/writing/recompute_primary.py` loads the four frozen per-question prediction files and `request_split.json`, recomputes the primary scores and all three fixed paired contrasts, then matches `report.json` and `closeout.json` within 1e-10. Its output is `numeric-audit.json`. `source-key-inventory.json` maps manuscript values to direct JSON keys.

## Unresolved questions

1. Does a response-preserving objective help under a separately preregistered calibration replication?
2. Would a different readout, context bank, or joint allocator change the result without changing the primary question?
3. Do any calibration diagnostics predict downstream changes across independent masks and tasks?

These are future hypotheses, not claims made by paper-v1.
