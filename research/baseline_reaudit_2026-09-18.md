# Baseline re-audit: 50% pruning premise

Date: 2026-09-18

Scope: read-only audit of the existing masks, JSON receipts, reports, and source code. No masks, model scores, GPU runs, or shared Obsidian research-state writes were made. `Research/DLM-Pruning/Research-State.md` and sync status were read before this audit.

## Finding

The premise “all prior methods are worse than Uniform” is false for the completed 50% WikiText validation comparison. Under the final native sparse-prefix pipeline, DSA (layer and projection variants) and EvoPress have lower token NELBO than row-quota Uniform. OWL projection is close but still worse by point estimate; OWL layer, LSA, Alpha, DLP, and layer-global Uniform are worse.

That conclusion is about the frozen WikiText likelihood proxy and its current mask family. It does not establish a GSM8K win, and it must not be combined with the historical GSM8K Uniform=62 result as though the masks were identical. The historical 62 and current context Uniform=54 are different ranking/mask-construction families.

## Completed full WikiText-2 validation at 50%

All rows below are completed receipts from the validation split: 551 article-local chunks, 268,163 tokens, sequence length 512, MC128 exact-k masking, seed 2025, model `GSAI-ML/LLaDA-8B-Base` revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, 224 prunable projections, and nominal budget `3,489,660,928 / 6,979,321,856`. NELBO and the PPL column are the saved `token_nelbo` and `ppl_upper_bound_estimate`; lower is better. The PPL value is a finite Monte Carlo upper-bound estimate, not a guaranteed bound and not corpus-generalization uncertainty.

| Method / receipt | Ranking or allocation family; mask engine | NELBO | PPL estimate | MC SE | Actual sparsity | Config / mask receipt |
|---|---|---:|---:|---:|---:|---|
| DSA / layer (`dlm_ppl50/dsa`) | Bounded DLM adaptation; layer allocation from public-operator search on development40 masked CE; row-wise Wanda masks | 2.530142 | 12.555285 | .007441 | 50.000000% | `a73b0aaf` / `44573728` |
| EvoPress (`dlm_ppl50/evopress`) | Official FastOBC/EvoPress machinery with explicit LLaDA sparse-prefix adaptation; not the Wanda engine | 2.540893 | 12.690996 | .007411 | 50.000026% | `4d6fb4eb` / `064029b8` |
| DSA / projection (`dlm_ppl50_projection/dsa_projection`) | Bounded DLM adaptation; projection allocation from public-operator search; row-wise Wanda masks | 2.544038 | 12.730979 | .007438 | 50.000000% | `5d4dc560` / `d5d0aa6a` |
| Uniform / row quota (`dlm_ppl50/uniform`) | Current native sparse-prefix Wanda; exact per-row 50% quota in every projection | 2.550805 | 12.817419 | .007440 | 50.000000% | `d96da7ff` / `3970b7a5` |
| OWL / projection (`dlm_ppl50_projection/owl_projection`) | OWL projection allocation; row-wise Wanda masks | 2.557026 | 12.897403 | .007447 | 50.000000% | `2cbddd1a` / `a5c4722f` |
| OWL / layer (`dlm_ppl50/owl`) | OWL layer allocation; row-wise Wanda masks | 2.569351 | 13.057348 | .007440 | 50.000000% | `66542450` / `4cea0b80` |
| LSA / layer (`dlm_ppl50/lsa_layer`) | LSA layer allocation; row-wise Wanda masks | 2.576876 | 13.155969 | .007448 | 50.000000% | `29f79b18` / `0fa5f65c` |
| LSA / projection (`dlm_ppl50/lsa_projection`) | LSA projection allocation; row-wise Wanda masks | 2.579124 | 13.185582 | .007453 | 50.000000% | `8b9e481e` / `4038537f` |
| Uniform / layer-global (`dlm_ppl50_uniform_layer`) | Same Wanda score, but each block globally prunes 50% across its seven matrices; no row/projection quota | 2.663777 | 14.350387 | .007499 | 50.000000% | `376bb059` / `cbc8d611` |
| Alpha (`dlm_ppl50/alpha`) | Alpha allocation; row-wise Wanda masks | 2.670538 | 14.447742 | .007453 | 50.000000% | `1496ae5b` / `a8802c9e` |
| DLP (`dlm_ppl50/dlp`) | DLP allocation; row-wise Wanda masks | 2.698444 | 14.856601 | .007515 | 50.000000% | `bea48cce` / `cc709510` |

The mechanism review independently verified all 11 manifests. Early-to-late block sparsity also differs: DSA layer is 56.58% in blocks 0–7 versus 47.01% in blocks 24–31; EvoPress is 50.78% versus 48.20%; DSA projection is 52.84% versus 49.16%; row-quota Uniform is 50% in both. Layer-global Uniform demonstrates that keeping every block at 50% does not control within-block allocation: its projection sparsities range from 15.63% to 90.26%, and its NELBO is much worse than row-quota Uniform.

The table therefore separates three effects that an “all methods versus Uniform” sentence would conflate: the ranking/proxy, the allocation granularity, and the mask engine. DSA is a bounded local adaptation rather than the official full DSA search. EvoPress is the clearest engine confound because it uses FastOBC. The other allocation candidates reuse provenance-checked dense surveys and their frozen parameters; they are not all the same ranking objective. None of these WikiText rows is a full GSM8K result.

## Full GSM8K evidence at 50%

No current `dlm_ppl50` candidate has a completed full 1,319-example GSM8K evaluation. The current A and A+C context-response full runs were stopped at 107/1,319 and 147/1,319 respectively, with no checkpointed predictions or final score.

The completed historical full GSM8K table is EXP-002, which uses its frozen 50% masks and the repository protocol (5-shot, strict exact match, temperature 0, length/block length/denoising steps 256):

| Historical method / receipt | Correct / 1,319 | Accuracy |
|---|---:|---:|
| Dense | 938 | 71.1145% |
| DLM-ABS | 745 | 56.4822% |
| DLM-SQUARE | 701 | 53.1463% |
| Wanda | 677 | 51.3268% |
| SparseGPT | 584 | 44.2760% |
| DLM-SUM | 0 | 0.0000% |

EXP-002 records DLM masks from its frozen `dlm_8x10x256` calibration (80 states) and Wanda/SparseGPT from clean `8x256` calibration; that is already a calibration-compute mismatch. The Fisher-Geometry prototype is another module-specific historical 50% artifact: it kept Standard Wanda on 223 matrices and changed only `block_31.ff_out`, obtaining 694/1,319 versus Standard Wanda 677/1,319, paired p=.303. It does not provide an all-methods comparison.

## GSM8K mini evidence and the 62 versus 54 discrepancy

The historical cached Uniform mini result is `experiments/uniform_wanda_sparsity_sweep/evaluation_results.json`, identifier `V0_50_uniform`: 62/100, exact 50%, mask SHA prefix `0c0a00d1`, prediction SHA prefix `ff202744`, protocol hash `1c27fe99`.

The current context-response pipeline reuses the new native sparse-prefix Uniform mask family and gives:

| Current context-response candidate | Correct / 100 | Paired comparison |
|---|---:|---|
| Uniform | 54 | current sparse-prefix baseline |
| A | 55 | A vs Uniform net +1; exact McNemar p=1.0 |
| A+C / AC | 61 | AC vs Uniform net +7; p=.167068; AC vs A net +6; p=.145996 |

These are paired gold-vs-rest context-response surrogate results, not NELBO. The report explicitly limits them: the paired gold reveal is not generated context, wrong-vs-wrong ranking is omitted, finite marginal effects may not compose, and there is no fresh generalization claim. The full A/AC runs were stopped by the user (`full_progress_A.json`: 107/1319; `full_progress_AC.json`: 147/1319; predictions not checkpointed).

The 62 versus 54 change is explained by remeasurement/reimplementation, not by a contradiction in one fixed Uniform method. The historical `V0_50_uniform` mask is from the prior cached one-shot activation/ranking family. The current `experiments/dlm_ppl50/sequential.py` does the following for each block: native batch-one prefix forward, collect that block's activation, compute the Wanda mask, apply it, then proceed to the next block. Its first-block control checks the historical Wanda ranking, but later blocks are intentionally remeasured after earlier blocks are sparse. Comparing the saved manifests shows all seven masks in block 00 match and all seven masks in each block 01–31 differ: 7/224 unchanged, 217/224 changed. Thus the current Uniform=54 is a different full-model mask and ranking trajectory. A/AC are allocations built on this current Uniform background.

## 65% evidence, kept separate

The earlier allocation-only 65% report is a post-hoc offline analysis using the old fixed ranking family. Its mini100 results were Uniform 12, Role 24, Aggregate 19, OWL 5, DLP 4, DSA 7, Alpha 1, and LSA 6. Those numbers supported the frozen-65% observation that the old methods were poor in that setup; they did not test native sparse-prefix remeasurement.

The corrected 2026-09-15 sequential 65% pipeline uses the same native prefix remeasurement idea as the later PPL50 pipeline and retains the exact 65% global budget. Its completed mini100 receipts are:

| Sequential 65% method | Correct / 100 | Versus current sequential Uniform |
|---|---:|---|
| Uniform | 12 | baseline |
| OWL | 9 | net −3; p=.507812 |
| DLP | 1 | net −11; p=.003418 |
| Alpha | 0 | net −12; p=.000488 |
| LSA layer | 2 | net −10; p=.006348 |
| DSA layer | 17 | net +5; p=.266846 |
| LSA projection (LSAC) | 12 | net 0; p=1.0 |

This is contextual evidence only: it is mini100 development data, no full GSM8K was run for these sequential candidates, and DSA was searched with a bounded custom controller. It shows that remeasurement changed the ranking picture at 65% (DSA 17 versus Uniform 12), so the earlier “all worse” pattern cannot be transferred to the final 50% PPL table.

## Version/provenance sequence

1. The historical one-shot Uniform50 receipt (`V0_50_uniform`) preserved the 62/100 GSM8K mini result.
2. The 65% sequential correction was created on 2026-09-15. Its README states that every block is collected after earlier blocks have already been sparsified; its receipts complete Uniform, OWL, DLP, Alpha, LSA, and DSA mini100.
3. The full PPL50 run was assembled on 2026-09-16 under the new default likelihood metric. Its README freezes the common WikiText validation protocol, current sparse-prefix Wanda integration, bounded DSA search, and explicit LLaDA EvoPress adaptation. The 11 validation results above completed with shared model/config assumptions.
4. Context-response mini100 completed on 2026-09-17 using the current Uniform family. The attempted full A and AC GSM8K runs were stopped before producing final results.

## Decision

The corrected premise is: **Uniform is not a universal winner; DSA and EvoPress beat row-quota Uniform on the completed 50% WikiText NELBO proxy, while GSM8K superiority remains unestablished because the current full A/AC runs stopped and no all-method full GSM table exists.** Historical Uniform62, current Uniform54, A55, and AC61 should remain separate records until a single mask/ranking family is evaluated under one declared downstream protocol.

Primary artifacts audited: `experiments/dlm_ppl50/*/validation/results.json`, `experiments/dlm_ppl50_projection/*/validation/results.json`, `experiments/dlm_ppl50_uniform_layer/validation/results.json`, `experiments/dlm_ppl50_mechanism_review/report.md`, `experiments/dlm_ppl50/config.json`, `experiments/dlm_ppl50/sequential.py`, `experiments/dlm_loss_aggregation/exp002/report.md`, `experiments/dlm_loss_aggregation/exp002/results/gsm8k.csv`, `experiments/uniform_wanda_sparsity_sweep/evaluation_results.json`, `experiments/dlm_context_response50/{report.md,results.json,full_progress_A.json,full_progress_AC.json}`, and `experiments/dlm_allocation_sequential65/*/results.json`.
