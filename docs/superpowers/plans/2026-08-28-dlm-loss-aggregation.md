# DLM-Loss Aggregation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate the disk-efficient DLM SUM/ABS/SQUARE scorer through calibration digest verification, one-module smoke testing, and one-state resource profiling without starting the full 32-block run.

**Architecture:** A pure experiment module owns effect accumulation, stable row masks, compact masks, and diagnostics. A runner regenerates the pinned 80-state manifest, performs block-scoped hooked backward passes, profiles Spearman, and exposes smoke/full commands. The existing LLaDA evaluator gains only an optional in-memory model/tokenizer seam.

**Tech Stack:** Python 3.10, PyTorch 2.8, NumPy, PyYAML, Transformers, lm-eval.

**Spec:** `docs/superpowers/specs/2026-08-28-dlm-loss-aggregation-design.md`

## Global Constraints

- Never use, modify, or report Sink-Aware pruning.
- Pin `GSAI-ML/LLaDA-8B-Base` to revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`.
- Require historical state SHA-256 `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df` before scoring.
- SUM, ABS, and SQUARE consume the same effect from every backward pass with uniform state weight 1.
- Keep no per-state gradients, full score artifacts, permanent masks, or permanent checkpoints.
- Treat identical scores/masks and one-sign SUM as diagnostics, never failures.
- Do not start the full 32-block run in this plan.

---

### Task 1: Pure aggregation, masks, and diagnostics

**Files:**
- Create: `experiments/dlm_loss_aggregation/__init__.py`
- Create: `experiments/dlm_loss_aggregation/core.py`
- Create: `tests/test_dlm_loss_aggregation.py`

**Interfaces:**
- Produces: `EffectAccumulator(shape)`, `rowwise_mask(score, sparsity)`, `pack_mask(mask)`, `unpack_mask(payload)`, `mask_sha256(payload)`, `pairwise_diagnostics(scores, spearman_sample_size=None, seed=0)`.

- [ ] **Step 1: Write failing synthetic equation and weighting tests**

Use literal effects `[[1, -2], [-3, 4]]` and `[[-1, 2], [3, -4]]`. Assert signed mean is all zero, mean absolute is `[[1,2],[3,4]]`, and mean square is `[[1,4],[9,16]]`. Add a weighted two-state literal proving the scalar state-weight seam. Run:

```bash
pytest -q tests/test_dlm_loss_aggregation.py -k 'accumulator'
```

Expected: collection/import failure because `core.py` does not exist.

- [ ] **Step 2: Implement the minimal FP32 accumulator and pass the tests**

The accumulator updates signed first, converts the temporary effect to absolute in place, then updates absolute and square. It stores scalar observation count and total weight and validates shape/finite values.

- [ ] **Step 3: Write failing stable signed-mask and compact-mask tests**

Assert negative SUM values are selected before positive values, every row prunes `floor(width*sparsity)`, a one-state ABS/SQUARE pair yields identical masks without error, and packed masks round-trip with a stable hash.

- [ ] **Step 4: Implement minimal stable row masks and NumPy bit packing**

Run:

```bash
pytest -q tests/test_dlm_loss_aggregation.py -k 'mask'
```

Expected: pass.

- [ ] **Step 5: Write failing diagnostic tests**

Use hand-ranked small matrices to assert pairwise Spearman, exact XOR, negative-SUM fraction, sign-consistency summaries, spike-ratio summaries, shared deterministic sample indices, and recorded undefined constant-vector correlations.

- [ ] **Step 6: Implement diagnostics and pass all pure tests**

Exact Spearman ranks flattened matrices with tie averaging. Sampled mode selects at most one million shared indices using a seed-local generator. XOR always compares complete boolean masks.

- [ ] **Step 7: Commit**

```bash
git add experiments/dlm_loss_aggregation/__init__.py experiments/dlm_loss_aggregation/core.py tests/test_dlm_loss_aggregation.py
git commit -m "feat: aggregate DLM pruning effects"
```

### Task 2: Frozen calibration manifest and block collector

**Files:**
- Create: `experiments/dlm_loss_aggregation/config.yaml`
- Create: `experiments/dlm_loss_aggregation/run.py`
- Modify: `tests/test_dlm_loss_aggregation.py`

**Interfaces:**
- Consumes: Task 1 core functions.
- Produces: `build_calibration_manifest(clean_ids, mask_id, config)`, `historical_state_digest(manifest)`, `BlockEffectCollector`, CLI commands `manifest`, `smoke`, and `profile`.

- [ ] **Step 1: Write failing deterministic manifest tests**

Use tiny token tensors to assert state ordering, timestep/sample seeds, retry metadata, clean/noisy IDs, masked indices, `p_mask`, global seed, and a literal canonical digest. Assert a mismatched expected digest raises before model scoring.

- [ ] **Step 2: Implement manifest generation using the existing official loss/corruption semantics**

The persisted manifest adds model/dataset/global metadata, while `historical_state_digest` hashes the canonical legacy projection `{"version":1,"states":[...]}` so it can be compared to `1e44d30e...`.

- [ ] **Step 3: Write failing hook-collector tests with a tiny two-linear model**

Assert one backward updates every accumulator from the same gradients, clears `.grad`, preserves dense weights, counts each state once, and rejects missing/non-finite gradients.

- [ ] **Step 4: Implement block hooks and block-scoped LLaDA forward helpers**

Only one block's seven linear weights require gradients. Hooks move one gradient at a time to CPU FP32, form `-w*g`, update all three statistics, set `.grad = None`, and verify every module was seen before advancing the state.

- [ ] **Step 5: Add config validation and smoke gates**

Validate the exact model revision, 8x256 spans, ten timesteps, uniform alpha, 224-module expectation, 50% pruning, digest, and 30 GiB CUDA ceiling. Smoke output records but does not gate on correlations, negative-SUM fraction, or XOR.

- [ ] **Step 6: Run focused and repository tests**

```bash
pytest -q tests/test_dlm_loss_aggregation.py
pytest -q tests/test_dlm_gradient_sensitivity.py tests/test_eval_llada.py
```

- [ ] **Step 7: Commit**

```bash
git add experiments/dlm_loss_aggregation/config.yaml experiments/dlm_loss_aggregation/run.py tests/test_dlm_loss_aggregation.py
git commit -m "feat: collect blockwise DLM aggregation masks"
```

### Task 3: In-memory evaluation and sequential orchestration

**Files:**
- Modify: `eval_llada.py`
- Modify: `experiments/dlm_loss_aggregation/run.py`
- Modify: `tests/test_eval_llada.py`
- Modify: `tests/test_dlm_loss_aggregation.py`

**Interfaces:**
- Produces: optional `model` and `tokenizer` constructor inputs on `LLaDAEvalHarness`; runner evaluation order Dense, SUM, ABS, SQUARE, Wanda, SparseGPT.

- [ ] **Step 1: Write a failing evaluator injection test**

Instantiate the harness with a tiny real `nn.Module` and tokenizer double. Assert it does not call `from_pretrained`, does not move an already placed injected model, and preserves the existing path-loading behavior when injection is absent.

- [ ] **Step 2: Implement the two optional constructor inputs**

Keep CLI behavior unchanged. Direct evaluation passes the existing model and tokenizer, then calls lm-eval with GSM8K, 5-shot, and the fixed generation arguments.

- [ ] **Step 3: Write failing orchestration-order and mask-release tests**

Use small real packed masks and injected load/prune/evaluate callables. Assert each method starts from a new dense model, DLM packed masks are freed immediately after verified evaluation, and Wanda/SparseGPT are labeled reference baselines.

- [ ] **Step 4: Implement minimal sequential orchestration and CSV/readback checks**

Persist only method, sparsity, accuracy, example count, elapsed seconds, model revision, calibration label, mask hash, and checkpoint value `in-memory`.

- [ ] **Step 5: Run tests and commit**

```bash
pytest -q tests/test_eval_llada.py tests/test_dlm_loss_aggregation.py
git add eval_llada.py experiments/dlm_loss_aggregation/run.py tests/test_eval_llada.py tests/test_dlm_loss_aggregation.py
git commit -m "feat: evaluate pruned LLaDA models in memory"
```

### Task 4: Pre-full-run validation on RTX 5090

**Files:**
- Generate: `experiments/dlm_loss_aggregation/calibration_manifest.json`
- Generate: `experiments/dlm_loss_aggregation/diagnostics/spearman_profile.json`
- Generate: `experiments/dlm_loss_aggregation/logs/manifest.log`
- Generate: `experiments/dlm_loss_aggregation/logs/smoke.log`
- Generate: `experiments/dlm_loss_aggregation/logs/profile.log`
- Update: `experiments/dlm_loss_aggregation/report.md`

**Interfaces:**
- Consumes: Task 2 CLI.
- Produces: the requested evidence checkpoint; no full-run artifacts.

- [ ] **Step 1: Regenerate and validate the real manifest**

```bash
python -m experiments.dlm_loss_aggregation.run manifest --config experiments/dlm_loss_aggregation/config.yaml
```

Require exact digest `1e44d30e...`; stop on mismatch.

- [ ] **Step 2: Run one-module tiny-calibration smoke**

```bash
python -m experiments.dlm_loss_aggregation.run smoke --config experiments/dlm_loss_aggregation/config.yaml
```

Verify finite loss/gradients/scores, exact shapes/counts, nonnegative ABS/SQUARE, unchanged dense weights, exact row sparsity, and record correlations/negative-SUM/XOR without identity gates.

- [ ] **Step 3: Profile one state and exact Spearman on the largest matrix**

```bash
python -m experiments.dlm_loss_aggregation.run profile --config experiments/dlm_loss_aggregation/config.yaml
```

Record CUDA allocated/reserved peaks, process RSS, state seconds, exact rank seconds, largest shape, and projected full scoring time. Select exact Spearman only when the recorded profile is comfortable; otherwise record deterministic one-million-weight sampling.

- [ ] **Step 4: Run final pre-full verification**

```bash
pytest -q tests/test_dlm_loss_aggregation.py tests/test_eval_llada.py tests/test_dlm_gradient_sensitivity.py
git status --short
```

- [ ] **Step 5: Record and commit compact validation artifacts**

Stage only the manifest, compact JSON/log/report files, and source changes related to this experiment. Do not commit or run full masks, scores, checkpoints, or 32-block scoring.
