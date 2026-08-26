# Time-Risk DLM Gradient-Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, at bounded cost, whether timestep dispersion in the official LLaDA-loss gradient provides a reproducible pruning signal beyond mean gradient sensitivity.

**Architecture:** Add one tested statistics module and one experiment CLI. The CLI reuses the repository's LLaDA loader, WikiText2 loader, Wanda implementation, model saver, and GSM8K harness. It processes one transformer block at a time, updates full and split-half timestep statistics only after all eight sequence gradients for a timestep are complete, writes bit-packed masks atomically, and materializes only one checkpoint at a time.

**Tech Stack:** Python 3.10, PyTorch 2.8, Transformers 4.49, NumPy, pytest, JSON, Safetensors checkpoints, lm-evaluation-harness 0.4.8

**Spec:** `docs/superpowers/specs/2026-08-18-time-risk-dlm-fisher-experiment-design.md`

## Global Constraints

- Call the method `DLM gradient sensitivity`, `Mean DLM-Grad`, or `Time-Risk DLM-Grad`; do not claim a formal Fisher/Hessian derivation.
- Use the official loss exactly: masked-token CE divided by `p_mask(t)`, summed and divided by full sequence length, with `p_mask(t) = (1 - 1e-3) * t + 1e-3`.
- Use eight calibration sequences, length 256, seed 0, midpoint timesteps `0.05, 0.15, ..., 0.95`, and one deterministic mask per `(sequence, timestep)`.
- Compute one backward per sequence. Never square a batch-averaged gradient.
- Complete `F(t) = mean_x[w^2 g(x,t)^2]` before a Welford update. Full, split-A, and split-B accumulators must each receive exactly ten updates, not eighty.
- Keep sequences 0–3 in split A and 4–7 in split B. Derive split reliability from the same backward passes.
- Cast gradients to FP32 before squaring and move only the block-local squared gradient to CPU.
- Prune the same seven linear weights per block as Wanda. Exclude embeddings, final vocabulary projection, and norms.
- Use the existing row-wise unstructured geometry: `floor(input_width * sparsity)` zeros in every output row.
- Do not z-score `mu` or `sigma`. Select one global lambda from `{0.25, 0.5, 1.0}` using 50%, 60%, and 70%; keep 75% observational.
- Do not add token scoring, CVaR, channel 3848 protection, new dependencies, distributed execution, or a generic experiment framework.
- Run tests as `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ...`; the installed third-party pytest plugin is incompatible with PyTorch 2.8.
- Commit implementation/configuration before every expensive experiment. Stop at a failed gate and record the failure; do not silently widen the lambda grid or calibration budget.
- Do not push commits.

---

## File Map

- `lib/dlm_gradient_sensitivity.py`: official loss, deterministic states, timestep accumulator, row-wise masks, bit packing, diagnostics, runtime projection, paired loss summaries, and model-mask application.
- `tests/test_dlm_gradient_sensitivity.py`: focused mathematical, serialization, suffix-gradient, selection, and bootstrap checks.
- `dlm_gradient_sensitivity.py`: the only new CLI, with `feasibility`, `score`, `materialize`, `loss`, and `summarize-loss` subcommands.
- `main_llada.py`: add the missing `--seqlen` override so the Wanda calibration uses the same length-256 sequences.
- `codex/time_risk_sensitivity/config.json`: immutable pilot constants and model revision; committed before GPU work.
- `codex/time_risk_sensitivity/results/`: ignored generated masks, temporary checkpoints, state files, metrics, and logs; force-add only compact JSON/Markdown reports.
- `codex/time_risk_sensitivity/results/report.md`: final gate decisions, commands, commit hashes, artifacts, metrics, and conclusion.

No pruning formula or model implementation changes are planned. `dlm_gradient_sensitivity.py` imports `get_llm` and `copy_llada_support_files` from `main_llada.py`, `find_layers`/`check_sparsity` from `lib/prune_llada.py`, and `get_loaders` from `lib/data.py`.

### Task 1: Official DLM state and loss

**Files:**
- Create: `tests/test_dlm_gradient_sensitivity.py`
- Create: `lib/dlm_gradient_sensitivity.py`

**Interfaces:**
- Produces: `midpoint_timesteps(count: int = 10) -> tuple[float, ...]`
- Produces: `mask_probability(timestep: float, eps: float = 1e-3) -> float`
- Produces: `make_masked_state(clean_ids: Tensor, timestep: float, mask_id: int, seed: int) -> tuple[Tensor, Tensor, float]`
- Produces: `official_dlm_loss(logits: Tensor, clean_ids: Tensor, mask: Tensor, p_mask: float) -> Tensor`

- [ ] **Step 1: Write the failing midpoint, masking, and loss tests**

```python
def test_official_loss_uses_masked_sum_over_full_sequence_and_p_mask():
    logits = torch.zeros(1, 4, 2)
    clean = torch.tensor([[0, 1, 0, 1]])
    mask = torch.tensor([[True, False, True, False]])
    loss = official_dlm_loss(logits, clean, mask, p_mask=0.5)
    assert loss.item() == pytest.approx(math.log(2.0))

def test_fixed_mask_state_is_reproducible_and_nonempty():
    clean = torch.arange(256).unsqueeze(0)
    first = make_masked_state(clean, 0.05, mask_id=999, seed=7)
    second = make_masked_state(clean, 0.05, mask_id=999, seed=7)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[1].any()
```

Also assert that the midpoint tuple is exactly `(0.05, 0.15, ..., 0.95)`, only masked positions change, and a mask-free draw is deterministically retried with incremented seed.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Expected: collection fails because `lib.dlm_gradient_sensitivity` does not exist.

- [ ] **Step 3: Implement the four pure functions**

```python
def official_dlm_loss(logits, clean_ids, mask, p_mask):
    if not mask.any():
        raise ValueError("official DLM loss requires at least one masked token")
    token_loss = F.cross_entropy(logits[mask].float(), clean_ids[mask], reduction="sum")
    return token_loss / p_mask / clean_ids.numel()
```

Use a local `torch.Generator(device=clean_ids.device)` seeded by the supplied integer. Do not mutate global RNG state. Validate `0 < p_mask <= 1`, shape equality, and a positive sequence length.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Expected: all Task 1 tests pass.

### Task 2: Timestep-only full and split-half statistics

**Files:**
- Modify: `tests/test_dlm_gradient_sensitivity.py`
- Modify: `lib/dlm_gradient_sensitivity.py`

**Interface:**
- Produces: `TimestepSensitivityAccumulator(weights: dict[str, Tensor], split_size: int = 4)`
- Methods: `add_state(grads: dict[str, Tensor], split: str)`, `finish_timestep()`, `finalize() -> dict`

- [ ] **Step 1: Write the invariant test before implementation**

Use one scalar weight `w=2`, two sequences per split, and two timesteps:

```python
# t1: A gradients [1, 3], B gradients [5, 7]
# F_A=20, F_B=148, F=84
# t2: A gradients [2, 4], B gradients [6, 8]
# F_A=40, F_B=200, F=120
assert result["full"]["mu"]["w"].item() == 102
assert result["full"]["sigma"]["w"].item() == 18
assert result["a"]["sigma"]["w"].item() == 10
assert result["b"]["sigma"]["w"].item() == 26
assert result["update_count"] == 2
```

Add rejection tests for an early `finish_timestep()`, a ninth state in a four-plus-four timestep, an unknown split, missing gradient keys, non-finite gradients, and `finalize()` while a timestep is incomplete.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py -k accumulator`

Expected: FAIL because the accumulator is absent.

- [ ] **Step 3: Implement the minimum state machine**

Keep CPU FP32 tensors for current split sums plus Welford `mean`/`M2` for full, A, and B. In `finish_timestep()`:

```python
f_a = weight_sq * sum_sq_a / split_size
f_b = weight_sq * sum_sq_b / split_size
f_full = (f_a + f_b) / 2
for group, value in (("full", f_full), ("a", f_a), ("b", f_b)):
    delta = value - mean[group]
    mean[group].add_(delta / next_update_count)
    m2[group].add_(delta * (value - mean[group]))
```

Reset current sums and counts only after all modules update successfully. `finalize()` returns population standard deviation `sqrt(M2 / update_count)` and requires exactly ten updates in production; accept an `expected_timesteps` constructor argument so the two-timestep unit test stays literal.

- [ ] **Step 4: Run the full focused suite and verify GREEN**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Expected: all tests pass, including `update_count == 2` in the synthetic case.

- [ ] **Step 5: Review and commit Tasks 1–2 together**

Run: `git status --short && git diff --check && git diff -- lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py`

Commit: `git add lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py && git commit -m "feat: add timestep DLM gradient statistics"`

### Task 3: Row-wise masks, bit packing, and diagnostics

**Files:**
- Modify: `tests/test_dlm_gradient_sensitivity.py`
- Modify: `lib/dlm_gradient_sensitivity.py`

**Interfaces:**
- Produces: `rowwise_prune_mask(score: Tensor, sparsity: float) -> Tensor`
- Produces: `pack_mask(mask: Tensor) -> dict`
- Produces: `unpack_mask(artifact: dict) -> Tensor`
- Produces: `save_mask_block(path: Path, masks: dict, metadata: dict) -> None`
- Produces: `load_mask_block(path: Path) -> tuple[dict, dict]`
- Produces: `sampled_spearman(left: Tensor, right: Tensor, sample_size: int, seed: int) -> float`
- Produces: `jaccard(left: Tensor, right: Tensor) -> float`

- [ ] **Step 1: Add failing geometry and round-trip tests**

Assert, for several rectangular score tensors and all four sparsities, that every row prunes exactly `floor(width * sparsity)` weights. Assert stable tie handling by column index. Pack a non-byte-aligned `3 x 11` mask with NumPy `packbits`, reload it, and require bit-exact equality.

Add tests that a changed byte or mismatched shape fails SHA-256 validation, and that a write interrupted before `Path.replace()` cannot replace a valid artifact.

- [ ] **Step 2: Add failing diagnostic tests**

Use small tensors with known identical, reversed, and partially tied ranks. Assert Spearman values, deterministic sample indices, mask Jaccard, row-change fraction, and top-1%-sigma overlap. Return `NaN` with a recorded reason for a constant vector instead of hiding it as zero.

- [ ] **Step 3: Run tests and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py -k "mask or pack or spearman or jaccard"`

Expected: FAIL because mask artifact and diagnostic helpers are absent.

- [ ] **Step 4: Implement with existing dependencies only**

Use `torch.argsort(..., stable=True)` for row-wise selection, `numpy.packbits`/`unpackbits` for storage, `hashlib.sha256` for checksums, and `tempfile.NamedTemporaryFile` followed by `Path.replace()` for atomic writes. Implement average ranks for ties directly in PyTorch; do not add SciPy.

Store, per module, shape, dtype, bit order, prune count, per-row prune count, byte length, and checksum. Reject unknown artifact versions and trailing or missing bits.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Expected: all tests pass.

- [ ] **Step 6: Review and commit**

Run: `git status --short && git diff --check && git diff -- lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py`

Commit: `git add lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py && git commit -m "feat: persist DLM sensitivity masks"`

### Task 4: Exact block-suffix gradient engine and Stage 0 CLI

**Files:**
- Modify: `tests/test_dlm_gradient_sensitivity.py`
- Modify: `lib/dlm_gradient_sensitivity.py`
- Create: `dlm_gradient_sensitivity.py`
- Create: `codex/time_risk_sensitivity/config.json`

**Interfaces:**
- Produces: `embed_state(model, noisy_ids: Tensor) -> Tensor`
- Produces: `suffix_logits(model, hidden: Tensor, start_block: int) -> Tensor`
- Produces: `block_state_gradients(model, block_index: int, hidden: Tensor, clean_ids: Tensor, mask: Tensor, p_mask: float) -> tuple[dict[str, Tensor], Tensor]`
- Produces: `project_scoring_seconds(block_31_seconds: float, block_0_seconds: float, blocks: int = 32, states: int = 80) -> dict`
- CLI: `python dlm_gradient_sensitivity.py feasibility ...`

- [ ] **Step 1: Write a tiny-LLaDA suffix-equivalence test**

Instantiate a small `LLaDAModelLM` from the repository configuration with two blocks, short hidden width, and small vocabulary. In eval mode, assert:

```python
full = model(noisy_ids).logits
cached = embed_state(model, noisy_ids)
suffix = suffix_logits(model, cached, start_block=0)
torch.testing.assert_close(suffix, full)
```

Repeat with the cached dense output of block 0 and `start_block=1`. This test must exercise the real LLaDA embedding scaling, position handling, final norm, weight tying, and logit scaling paths.

- [ ] **Step 2: Write an exact-gradient and immutability test**

Enable gradients only for the target block's seven `nn.Linear` weights. Compare gradients from a normal full forward to `block_state_gradients()` for the same fixed state. Assert tensorwise closeness, finite values, no gradients on frozen parameters, and byte-identical parameters before and after both passes.

- [ ] **Step 3: Write runtime-projection and resource-gate tests**

For `block31=2s` and `block0=33s`, assert `fixed=2`, `suffix=1`, and projected scoring time `80 * sum(2 + 31 - b for b in range(32)) = 44,800s`. Reject a negative fitted suffix slope. Test the gate at the exact 30 GiB and 24-hour boundaries.

- [ ] **Step 4: Run tests and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py -k "suffix or gradient or project or gate"`

Expected: FAIL because the suffix engine and feasibility functions are absent.

- [ ] **Step 5: Implement the exact suffix path**

Mirror only the executed lines of `LLaDAModel.forward`: token/position embeddings and embedding dropout before block 0; block calls with `attention_bias=None`, `layer_past=None`, `use_cache=False`; final norm; tied `F.linear` or `ff_out`; and configured logit scaling. Keep all frozen suffix operations in the autograd graph because gradients must reach the target block output.

Temporarily set `requires_grad` on target block linear weights, restore every original flag in `finally`, call `model.zero_grad(set_to_none=True)`, and return detached FP32 CPU gradients plus the detached target-block output. That output becomes the cached input for the next block, avoiding a duplicate dense block pass. Never mutate weight data.

- [ ] **Step 6: Implement the feasibility subcommand**

The command loads one fixed WikiText2 state, caches its dense block inputs, and measures blocks 31 then 0. Each measurement includes forward, backward, FP32 gradient cast/square, and the real per-state split-sum update. Outside the per-state timer, replay that detached gradient into the remaining seven accumulator slots, then separately time the real `finish_timestep()` and add ten such updates per block to the projection; label the replay as a timing device, never as eight observations. This exercises the full working-buffer allocation without duplicating it. Record `time.perf_counter()`, `torch.cuda.max_memory_allocated()`, `torch.cuda.max_memory_reserved()`, `/proc/self/status` `VmRSS`/`VmSwap` before and after, gradient finite/nonzero counts, fitted runtime, statistic-finalization overhead, and every gate decision in JSON.

CLI command:

```bash
python dlm_gradient_sensitivity.py feasibility \
  --config codex/time_risk_sensitivity/config.json \
  --output codex/time_risk_sensitivity/results/stage0.json
```

The process exits 0 only when both blocks use at most 30 GiB, create no additional swap, have finite nonzero gradients, and project at most 24 GPU hours. It exits 2 for a scientifically valid gate failure and writes the report before exiting.

- [ ] **Step 7: Commit immutable pilot configuration**

`pilot.json` records the exact model ID/revision, WikiText2, calibration seed/index count, sequence length, mask ID source, epsilon, ten timesteps, A/B membership, lambdas, sparsities, Stage 0 bounds, reliability thresholds, validation slice offsets, bootstrap seed/resamples, and GSM8K limits. It contains no machine-specific output path.

- [ ] **Step 8: Run tests, review, and commit before GPU work**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Run: `git status --short && git diff --check && git diff -- lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py codex/time_risk_sensitivity/config.json`

Commit: `git add lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py codex/time_risk_sensitivity/config.json && git commit -m "feat: add blockwise DLM sensitivity probe"`

### Task 5: Execute and record Stage 0

**Files:**
- Generate: `codex/time_risk_sensitivity/results/stage0.json`
- Modify: `codex/time_risk_sensitivity/results/report.md`

- [ ] **Step 1: Record environment and run the committed command**

Record `git rev-parse HEAD`, `nvidia-smi`, available RAM/swap/disk, package versions, full command, start/end timestamps, and exit code. Run the Task 4 command without modifying code during the measurement.

- [ ] **Step 2: Independently verify the projection and gate**

Recompute the two-point fit from the raw block timings in the JSON. Confirm that gradient casting/statistics were inside the timed region and that swap delta is zero.

- [ ] **Step 3: Stop or proceed**

If Stage 0 fails, append the exact failed condition to `report.md`, commit the report, and stop. Do not enable checkpointing or alter the 24-hour bound without a new review. If it passes, record `GO: Stage 1`.

- [ ] **Step 4: Commit the compact feasibility result**

Run: `git status --short && git diff --check && git diff -- codex/time_risk_sensitivity/results/stage0.json codex/time_risk_sensitivity/results/report.md`

Commit: `git add -f codex/time_risk_sensitivity/results/stage0.json codex/time_risk_sensitivity/results/report.md && git commit -m "exp: record DLM sensitivity feasibility"`

### Task 6: Full blockwise scorer, resume, and reliability diagnostics

**Files:**
- Modify: `tests/test_dlm_gradient_sensitivity.py`
- Modify: `lib/dlm_gradient_sensitivity.py`
- Modify: `dlm_gradient_sensitivity.py`

**Interfaces:**
- Produces: `score_block(model, block_index, cached_states, config) -> tuple[dict, dict]`
- Produces: `score` CLI with atomic block artifacts and a manifest

- [ ] **Step 1: Write a one-block end-to-end toy test**

Use the tiny LLaDA model and eight fixed states over two test timesteps. Compare the scorer's `mu`, `sigma`, `sigma_A`, and `sigma_B` against a direct Python loop over saved per-sequence gradients. Assert exactly two Welford updates, then assert every lambda/sparsity mask equals `rowwise_prune_mask(mu + lambda * sigma, sparsity)`.

- [ ] **Step 2: Write resume and corruption tests**

Run a two-block toy score into a temporary directory, interrupt after block 0, and resume. Assert verified block 0 is not recomputed and the resumed output is byte-identical to an uninterrupted run. Corrupt a block checksum and require a hard failure rather than a silent skip.

- [ ] **Step 3: Run tests and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py -k "score_block or resume"`

Expected: FAIL because orchestration is absent.

- [ ] **Step 4: Implement the scoring loop**

Precompute and save the 80 masked token states first. Cache their dense embedding outputs on CPU in model dtype. For block `b`, move one cached state to GPU, calculate target-block gradients through the frozen suffix, and feed its seven gradients to the accumulator. Call `finish_timestep()` only after A4+B4 are present. After ten updates:

- build full `mu`, full `sigma`, split `sigma_A`, and split `sigma_B`;
- compute full and split masks while each module score is resident;
- write the sixteen full masks plus split masks needed for reliability;
- record `rho(mu,sigma)`, `rho(sigma_A,sigma_B)`, `sigma/(mu+eps)` quantiles, top-1% overlap, exact mask disagreement/Jaccard, and changed-row fraction;
- atomically finalize the block artifact and manifest entry;
- replace each cached state with the detached target-block output already returned by its gradient pass.

Delete split masks after their exact overlaps are recorded; retain only their checksums and diagnostics. This avoids tripling the final artifact footprint.

Before the 32-block run, repeat block 31's first state immediately and record maximum absolute/relative gradient-square differences. Fail on non-finite output or tolerance violation. Mask determinism remains covered by the exact toy and resume tests.

- [ ] **Step 5: Add deterministic Stage 1 gates**

Aggregate module diagnostics with unweighted module medians/means exactly as the spec states. Pass reliability only when median module split-sigma Spearman is at least 0.5 and mean split-mask Jaccard at lambda 1.0 is at least 0.95 separately at 50%, 60%, and 70%. Stop if all three nonzero-lambda full masks are byte-identical to Mean at those sparsities.

- [ ] **Step 6: Run tests, review, and commit before full scoring**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_channel_constraint.py tests/test_activation_diagnostic.py`

Run: `git status --short && git diff --check && git diff -- lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py`

Commit: `git add lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py && git commit -m "feat: score timestep DLM sensitivity"`

### Task 7: Execute Stage 1 and record the reliability decision

**Files:**
- Generate: `codex/time_risk_sensitivity/results/masks/manifest.json`
- Generate: `codex/time_risk_sensitivity/results/stage1.json`
- Modify: `codex/time_risk_sensitivity/results/report.md`

- [ ] **Step 1: Run the committed scorer**

```bash
python dlm_gradient_sensitivity.py score \
  --config codex/time_risk_sensitivity/config.json \
  --artifact-dir codex/time_risk_sensitivity/results/masks \
  --output codex/time_risk_sensitivity/results/stage1.json
```

Capture stdout/stderr and peak resources. Resume only from checksum-verified completed blocks.

- [ ] **Step 2: Audit artifacts before interpreting metrics**

Assert 32 completed blocks, 224 module entries, ten full/A/B Welford updates per module, sixteen full masks, expected per-row prune counts, unique model revision/config digest, and valid SHA-256 for every packed block.

- [ ] **Step 3: Apply the Stage 1 gate exactly once**

Record module-level distributions and aggregate thresholds. If reliability fails or all useful masks are identical, write `NO-GO` and stop before checkpoint materialization. Do not tune the sampling seed, Spearman sample size, lambda, or overlap threshold after seeing results.

- [ ] **Step 4: Commit only compact results**

Force-add `stage1.json`, the small manifest, and `report.md`; leave packed masks ignored.

Commit: `git add -f codex/time_risk_sensitivity/results/stage1.json codex/time_risk_sensitivity/results/masks/manifest.json codex/time_risk_sensitivity/results/report.md && git commit -m "exp: record DLM sensitivity diagnostics"`

### Task 8: Packed-mask application and exact sparsity validation

**Files:**
- Modify: `tests/test_dlm_gradient_sensitivity.py`
- Modify: `lib/dlm_gradient_sensitivity.py`
- Modify: `dlm_gradient_sensitivity.py`
- Modify: `main_llada.py`

**Interfaces:**
- Produces: `apply_packed_mask(model, artifact_dir: Path, method: str, sparsity: float) -> dict`
- CLI: `python dlm_gradient_sensitivity.py materialize ...`

- [ ] **Step 1: Write failing application tests**

Apply a packed toy mask to a dense toy model. Assert selected weights become exactly zero, survivors remain byte-identical, all artifact modules are consumed exactly once, unknown/missing modules fail, and the reported zero count equals the decision-mask count. For every method at one sparsity, assert identical per-module and per-row prune counts.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py -k apply_packed`

Expected: FAIL because application is absent.

- [ ] **Step 3: Implement application and one-at-a-time materialization**

Load one block artifact at a time, validate its config/model digest, unpack only the requested method/sparsity mask, and set `weight.data[mask] = 0`. Run the existing `check_sparsity()`, save with `model.save_pretrained()`, and reuse `copy_llada_support_files()` from `main_llada.py`. Refuse a nonempty output directory.

Add `--seqlen` to `main_llada.py` with the existing model maximum as its default and assign `model.seqlen = args.seqlen` immediately after loading. This is the only Wanda integration change; do not touch Wanda scoring or masking.

Example:

```bash
python dlm_gradient_sensitivity.py materialize \
  --config codex/time_risk_sensitivity/config.json \
  --artifact-dir codex/time_risk_sensitivity/results/masks \
  --method mean --sparsity 0.60 \
  --output-dir codex/time_risk_sensitivity/results/tmp/mean-60
```

- [ ] **Step 4: Run tests, review, and commit**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Run: `git status --short && git diff --check && git diff -- lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py main_llada.py`

Commit: `git add lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py main_llada.py && git commit -m "feat: apply DLM masks at matched sequence length"`

### Task 9: Independent held-out loss selection and confirmation

**Files:**
- Modify: `tests/test_dlm_gradient_sensitivity.py`
- Modify: `lib/dlm_gradient_sensitivity.py`
- Modify: `dlm_gradient_sensitivity.py`

**Interfaces:**
- Produces: `validation_states(test_ids: Tensor, offset: int, count: int, length: int, ...) -> list[dict]`
- Produces: `choose_global_lambda(selection_metrics: dict) -> float`
- Produces: `paired_cluster_bootstrap(differences: Tensor, seed: int = 0, resamples: int = 10_000) -> tuple[float, float]`
- Produces: `loss_gate(selection: dict, confirmation: dict, selected_lambda: float) -> dict`
- CLI: `loss` and `summarize-loss`

- [ ] **Step 1: Write selection and non-overlap tests**

Use a synthetic token stream to assert selection uses chunks 0–7 while confirmation uses chunks 8–15, with no token-index overlap. Feed literal metrics where lambda 1.0 wins only at 75% and assert it is not selected. Assert ties select the smaller lambda. Dataset loading itself remains an integration action, not a network-dependent unit test.

- [ ] **Step 2: Write paired-bootstrap and gate tests**

Use a `[8 sequences, 10 timesteps]` difference tensor. Assert seed-0 output is deterministic, resampling preserves each sequence's ten-timestep cluster, and the returned 2.5/97.5 percentiles match a direct NumPy calculation. Cover all three point gates: at least two wins, mean relative improvement at least 1%, and no individual degradation beyond 1%, on both datasets.

- [ ] **Step 3: Run tests and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py -k "validation or lambda or bootstrap or loss_gate"`

Expected: FAIL because selection/confirmation helpers are absent.

- [ ] **Step 4: Implement reusable fixed validation states**

Load `Salesforce/wikitext`, `wikitext-2-raw-v1`, split `validation` directly through the already-installed Datasets package. Save clean token IDs, original token offsets, timestep, mask seed, `p_mask`, and packed mask once. The `loss` command reads those states unchanged for every model and writes per-sequence/per-timestep official losses. The selection set evaluates Wanda, Mean, and all three lambdas; the confirmation set rejects methods other than Mean and the already selected lambda.

- [ ] **Step 5: Implement selection, bootstrap, and summary**

`summarize-loss` reads completed method files, chooses one global lambda from only 50/60/70, emits paired differences for the separate confirmation states, computes the seed-0 10,000-resample sequence-cluster interval, and writes an explicit `GO` or `NO-GO` with every predicate value. The confidence interval is descriptive and cannot override a failed point gate.

- [ ] **Step 6: Run tests, review, and commit before evaluation**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_dlm_gradient_sensitivity.py`

Run: `git status --short && git diff --check && git diff -- lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py`

Commit: `git add lib/dlm_gradient_sensitivity.py tests/test_dlm_gradient_sensitivity.py dlm_gradient_sensitivity.py && git commit -m "feat: screen DLM sensitivity variants"`

### Task 10: Execute Stage 2 without retaining checkpoints

**Files:**
- Generate: `codex/time_risk_sensitivity/results/loss/*.json`
- Generate: `codex/time_risk_sensitivity/results/stage2.json`
- Modify: `codex/time_risk_sensitivity/results/report.md`

- [ ] **Step 1: Create Wanda baselines with unchanged repository code**

Materialize Wanda once per sparsity using `main_llada.py` with `--nsamples 8 --seed 0 --seqlen 256 --sparsity_type unstructured --skip_ppl`, then verify row/global sparsity. Do not change Wanda's scoring formula.

- [ ] **Step 2: Evaluate the selection set one checkpoint at a time**

For each sparsity, evaluate Wanda, Mean, and lambdas 0.25/0.5/1.0 against the same saved selection states. Delete each temporary checkpoint only after its loss JSON, model revision, actual sparsity, and command exit code are verified. Packed masks and metrics remain recoverable.

- [ ] **Step 3: Select lambda and evaluate confirmation**

Run `summarize-loss` on selection outputs, then materialize only Mean and the selected lambda for the separate confirmation states. Re-run the summary to add paired differences, bootstrap interval, and final gate.

- [ ] **Step 4: Stop or proceed**

If either selection or confirmation misses any point gate, record `NO-GO`, commit compact results, and do not run GSM8K. Do not try another lambda. Otherwise record `GO: GSM8K-64`.

- [ ] **Step 5: Commit the Stage 2 decision**

Run: `git status --short && git diff --check`

Commit: `git add -f codex/time_risk_sensitivity/results/loss codex/time_risk_sensitivity/results/stage2.json codex/time_risk_sensitivity/results/report.md && git commit -m "exp: record held-out DLM loss screen"`

### Task 11: Paired GSM8K-64 gate

**Files:**
- Generate: `codex/time_risk_sensitivity/results/gsm8k-64/*.json`
- Modify: `codex/time_risk_sensitivity/results/report.md`

- [ ] **Step 1: Freeze evaluation settings before the first run**

Record the existing `eval_llada.py` command, batch size, diffusion steps, generation length, exact-match filters, seed, and the fixed 64 document IDs in `report.md`. Use lm-evaluation-harness sample logging so correctness is recoverable per document.

- [ ] **Step 2: Evaluate twelve variants sequentially**

At each sparsity, run Wanda, Mean, and the single selected Time-Risk lambda. Materialize from the immutable dense model and packed mask each time. Verify actual sparsity before evaluation, save strict/flexible exact match plus sample logs, then remove that temporary checkpoint.

- [ ] **Step 3: Compute paired outcomes and apply the gate**

For Time-Risk versus Mean, record correct-ID sets, wins, losses, ties, and correct-count difference at every sparsity. Pass only if Time-Risk gains at least two correct answers at two or more of 50/60/70 and has a positive summed difference across those three. Keep 75% out of the decision.

- [ ] **Step 4: Commit the compact Stage 3 record**

Commit: `git add -f codex/time_risk_sensitivity/results/gsm8k-64 codex/time_risk_sensitivity/results/report.md && git commit -m "exp: record paired GSM8K-64 gate"`

### Task 12: Conditional confirmation and final handoff

**Files:**
- Generate on Stage 3 GO: `codex/time_risk_sensitivity/results/gsm8k-256/*.json`
- Generate on Stage 3 GO: `codex/time_risk_sensitivity/results/replication/*.json`
- Modify: `codex/time_risk_sensitivity/results/report.md`

- [ ] **Step 1: Select exactly one non-floor sparsity**

Choose the 50/60/70 sparsity with the largest 64-example Time-Risk gain; deterministic tie order is lower sparsity first. Evaluate only Mean and selected Time-Risk on the same fixed 256 GSM8K examples.

- [ ] **Step 2: Compute exact McNemar significance**

Use Python's standard-library `math.comb` to calculate the exact two-sided binomial McNemar p-value from paired discordant counts. Do not add SciPy.

- [ ] **Step 3: Replicate calibration only after GSM8K-256 confirms**

Update a copied config to 32 calibration sequences of length 512, rerun Stage 0 cost projection, then score/evaluate only Mean and the already selected lambda/sparsity. Commit the copied config before the run. Do not reopen lambda selection.

- [ ] **Step 4: Write the final conclusion**

Classify exactly one outcome: no reliable sigma estimate; reliable but no mask change; no held-out loss signal; loss-only signal; inconclusive GSM8K; downstream signal not replicated; or confirmed pilot signal. State that token/CVaR/3848 remain outside this pilot.

- [ ] **Step 5: Run final verification and commit**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`

Run: `git status --short && git diff --check`

Commit Stage 3 GO artifacts when present: `git add -f codex/time_risk_sensitivity/results/gsm8k-256 codex/time_risk_sensitivity/results/replication codex/time_risk_sensitivity/results/report.md && git commit -m "exp: confirm time-risk DLM sensitivity"`

If Stage 3 is not GO, commit only the updated report: `git add -f codex/time_risk_sensitivity/results/report.md && git commit -m "docs: conclude time-risk DLM sensitivity pilot"`

## Execution Checkpoints

1. After Task 4: code and configuration are committed; no expensive work has run.
2. After Task 5: user reviews projected runtime and resource gate before Stage 1.
3. After Task 7: user reviews split-half reliability and mask disagreement before model materialization.
4. After Task 10: Stage 2 decides whether GSM8K is authorized by the predeclared loss gate.
5. After Task 11: Stage 3 decides whether the 256-example and 32x512 replication costs are justified.

At every checkpoint, a failed gate is a completed experimental result, not an implementation failure.
