# EXP-005 Symmetric Token-Mass Reallocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Re-test REVEAL versus REMAIN token direction with equal-strength, opposite-sign perturbations around UNIFORM while keeping ABS aggregation fixed.

**Architecture:** Reuse the EXP-004 frozen calibration manifest, dense partition, scoring graph, row-wise mask writer, evaluator, and paired-statistics helpers. Add a symmetric contrast/alpha path and an EXP-005 runner/config that produces only SYM-REVEAL-ABS and SYM-REMAIN-ABS masks, evaluations, diagnostics, and report artifacts.

**Tech Stack:** Python, PyTorch, existing DLM pruning utilities, repository LLaDAEvalHarness, pytest, tmux for long runs, Obsidian experiment notes.

**Spec:** User-provided EXP-005 — Symmetric Token-Mass Reallocation design in the conversation.

## Global Constraints

- Reuse exactly the EXP-001/004 model revision, 80 frozen corrupted states, sequence length 256, 224 matrices, and 50% exact row-wise pruning.
- Use `q_s=|R_s|/|M_s|`, `c_R=1`, `c_U=-q_s/(1-q_s)`, and fixed `rho=0.5`.
- Use `alpha_sym_reveal=1+rho*c` and `alpha_sym_remain=1-rho*c`; do not use EXP-004 raw 2:1/1:2 weighting.
- Assert per state `mean(alpha)=1`, equal `mean(abs(alpha-1))`, equal `mean((alpha-1)^2)`, and exact mirror relation.
- Use ABS only; reuse frozen UNIFORM-ABS mask/result. Do not run SQUARE, random-token control, ratio sweeps, extra benchmarks, or follow-up experiments.
- Primary comparison is SYM-REVEAL-ABS vs SYM-REMAIN-ABS; secondary comparisons are each versus UNIFORM-ABS.
- Use exact two-sided McNemar tests on the same 1,319 GSM8K examples; no new multiple-comparison family beyond the three preregistered comparisons.
- Run long scoring/evaluation in a dedicated tmux session and create/update the Obsidian EXP-005 note before launch.

### Task 1: Add failing symmetric-alpha tests

**Files:**
- Modify: `tests/test_dlm_aggregation_exp004.py` or create `tests/test_dlm_aggregation_exp005.py`
- Test: same file

- [ ] Step 1: Add tests for `symmetric_token_weights` with a mask containing one reveal and three remain positions. Assert reveal alpha `1.5`, remain alpha `1 - 0.5*(1/3)`, mean alpha `1`, and the remain-up output is the exact mirror around one.
- [ ] Step 2: Add tests for equal perturbation magnitude and zero-mean contrast for multiple reveal fractions, including two reveal of four masked positions.
- [ ] Step 3: Run the new tests and confirm they fail because the symmetric helper does not exist.
- [ ] Step 4: Commit only the failing-test change if the repository policy requires a red checkpoint; otherwise continue directly to Task 2 after recording the expected failure.

### Task 2: Implement symmetric weighting and EXP-005 configuration

**Files:**
- Modify: `experiments/dlm_loss_aggregation/exp004/run.py` only if the helper is safely reusable
- Create: `experiments/dlm_loss_aggregation/exp005/config.json`
- Create: `experiments/dlm_loss_aggregation/exp005/run.py`
- Create: `experiments/dlm_loss_aggregation/exp005/run.sh`

**Interfaces:**
- `symmetric_token_weights(mask, reveal_mask, rho=0.5) -> tuple[torch.Tensor, torch.Tensor]`
- Runner consumes EXP-004 `calibration_state_manifest.json`, `token_partition_summary.json`, and frozen EXP-001/002 provenance.
- Runner produces `sym_reveal_abs_scores`, `sym_remain_abs_scores`, masks, diagnostics, evaluation records, paired comparisons, and `report.md`.

- [ ] Step 1: Implement the helper using masked-token fraction `q`; reject empty/all-reveal masks and any `rho` that would make alpha non-positive.
- [ ] Step 2: Add assertions that the two returned alpha tensors have mean one and satisfy `alpha_reveal + alpha_remain == 2` on every masked position.
- [ ] Step 3: Freeze EXP-004 partition indices; do not recompute dense decoding or token partition per condition.
- [ ] Step 4: Configure exactly two new methods, ABS aggregation only, with `rho=0.5`, and retain evaluation hash assertions and disk preflight.
- [ ] Step 5: Run the new tests and the existing EXP-004 tests; confirm green.
- [ ] Step 6: Commit `feat: add EXP-005 symmetric token weighting`.

### Task 3: Run scoring, mask diagnostics, and evaluation

**Files:**
- Create: EXP-005 output directory artifacts and logs
- Modify: Obsidian EXP-005 running note before launch

- [ ] Step 1: Run preflight and verify disk capacity for two masks plus temporary checkpoint materialization.
- [ ] Step 2: Run partition readback checks and symmetric alpha diagnostics before any model scoring.
- [ ] Step 3: Launch scoring in `tmux new -s exp005_symmetric_token_mass`; verify no parameter updates, independent gradients, exact row-wise 50%, and frozen UNIFORM-ABS reuse.
- [ ] Step 4: Materialize/evaluate SYM-REVEAL-ABS and SYM-REMAIN-ABS sequentially under the EXP-002 evaluator hash; checksum and release each intermediate payload.
- [ ] Step 5: Generate global/layer/module diagnostics and paired counts for the three comparisons.
- [ ] Step 6: Commit the implementation/results milestone before report polishing if artifacts are complete.

### Task 4: Generate and verify Korean report

**Files:**
- Create: `experiments/dlm_loss_aggregation/exp005/report.md`
- Create: required EXP-005 JSON/JSONL artifacts
- Modify: Obsidian EXP-005 note with completion and decision

- [ ] Step 1: Report objective, hypotheses, exact symmetric equations, observed q/alpha distributions, sanity checks, mask diagnostics, GSM8K results, paired tests, limitations, and decision.
- [ ] Step 2: Explicitly state that EXP-004 asymmetry is the motivation, ABS is fixed, and no random control or sweep was run.
- [ ] Step 3: Run the full relevant regression suite, artifact hash/readback audit, and report-value audit.
- [ ] Step 4: Commit `docs: record EXP-005 symmetric token-mass results`.
- [ ] Step 5: Do not launch another experiment automatically; wait for explicit instruction.
