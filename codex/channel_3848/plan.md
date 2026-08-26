# LLaDA Channel 3848 Causal Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether preserving the LLaDA-8B-Base residual-channel-3848 weight pathway causally recovers GSM8K accuracy at exactly 75% sparsity.

**Architecture:** Generate each pruning method's baseline mask and scores once, record compact independent restore/compensation deltas while those tensors are available, and materialize every variant from the immutable saved baseline. A single mask helper enforces module-local compensation without changing Wanda or Sink-Aware scores.

**Tech Stack:** Python 3.10, PyTorch 2.8, Transformers 4.49, lm-evaluation-harness 0.4.8, pytest, JSON, Safetensors checkpoints

## Global Constraints

- Do not modify Wanda or Sink-Aware importance formulas.
- Do not rank scores across modules or layers.
- Preserve each module's exact baseline prune count and exact 75% global sparsity.
- Derive channel 3848 and all three random-channel controls independently from one immutable Wanda baseline; never accumulate deltas.
- Report `protected_restored / protected_total` and `protected_already_survived_in_baseline / protected_total` alongside GSM8K accuracy.
- Use the repository GSM8K prompt and decoding configuration unchanged.
- Do not push commits.

---

## File Map

- `lib/channel_constraint.py`: architecture mapping, local mask transformation, compact delta artifacts, artifact application, and diagnostics aggregation.
- `tests/test_channel_constraint.py`: focused synthetic checks for protected axes, compensation scope, invariants, serialization, and independent variants.
- `lib/prune_llada.py`: call the shared recorder after the unchanged Wanda/Sink baseline mask is selected and before it is applied.
- `main_llada.py`: expose delta-generation/application arguments and skip unrelated PPL work during experiment checkpoint creation.
- `activation_diagnostic.py`: measure channel-3848 post-block residual magnitudes on fixed calibration samples.
- `eval_llada.py`: guard single-process synchronization only if the existing GSM8K path demonstrates the known `None` accelerator failure.
- `codex/channel_3848/results/`: ignored generated JSON/CSV/checkpoint outputs; the final Markdown report is force-added intentionally.

### Task 1: Core local mask constraint

**Files:**
- Create: `tests/test_channel_constraint.py`
- Create: `lib/channel_constraint.py`

**Interfaces:**
- Produces: `protected_axis(module_name: str) -> str`
- Produces: `build_channel_delta(module_name: str, weight: Tensor, score: Tensor, baseline_mask: Tensor, channel: int) -> dict`
- Produces: `apply_delta_to_mask(baseline_mask: Tensor, delta: dict) -> Tensor`
- Delta keys: `shape`, `restore_indices`, `restore_values`, `compensation_indices`, `stats`

- [ ] **Step 1: Write failing tests for architecture mapping and input-column compensation**

```python
def test_input_column_restores_and_compensates_in_same_rows():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    score = weight.clone()
    baseline = torch.tensor([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
    ], dtype=torch.bool)
    delta = build_channel_delta("q_proj", weight, score, baseline, channel=0)
    constrained = apply_delta_to_mask(baseline, delta)
    assert not constrained[:, 0].any()
    assert torch.equal(constrained.sum(1), baseline.sum(1))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_channel_constraint.py`

Expected: FAIL because `lib.channel_constraint` does not exist.

- [ ] **Step 3: Add output-row, intervention-count, and insufficient-candidate tests**

```python
def test_output_row_compensates_only_in_other_rows():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    score = weight.flip(0).clone()
    baseline = torch.tensor([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
    ], dtype=torch.bool)
    delta = build_channel_delta("attn_out", weight, score, baseline, channel=1)
    constrained = apply_delta_to_mask(baseline, delta)
    assert not constrained[1].any()
    assert constrained.sum() == baseline.sum()
    assert delta["stats"]["protected_restored"] == baseline[1].sum().item()
    assert delta["stats"]["compensation_pruned"] == baseline[1].sum().item()
```

Also assert `protected_total`, already-survived count and ratio, restored ratio,
mask Hamming distance, protected exclusion, and `ValueError` when compensation
candidates are insufficient.

- [ ] **Step 4: Implement the minimum mask transformer**

```python
INPUT_COLUMN_MODULES = {"q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"}
OUTPUT_ROW_MODULES = {"attn_out", "ff_out"}

def build_channel_delta(module_name, weight, score, baseline_mask, channel):
    leaf = module_name.rsplit(".", 1)[-1]
    constrained = baseline_mask.clone()
    if leaf in INPUT_COLUMN_MODULES:
        restore_rows = baseline_mask[:, channel].nonzero().flatten()
        candidates = (~baseline_mask).clone()
        candidates[:, channel] = False
        candidate_scores = score.masked_fill(~candidates, torch.inf)
        compensation_cols = candidate_scores.argmin(dim=1)[restore_rows]
        constrained[restore_rows, channel] = False
        constrained[restore_rows, compensation_cols] = True
    elif leaf in OUTPUT_ROW_MODULES:
        restore = baseline_mask[channel].nonzero().flatten()
        candidates = (~baseline_mask).clone()
        candidates[channel] = False
        candidate_scores = score.masked_fill(~candidates, torch.inf).flatten()
        compensation = torch.topk(candidate_scores, restore.numel(), largest=False).indices
        constrained[channel, restore] = False
        constrained.flatten()[compensation] = True
    else:
        raise ValueError(f"Unsupported prunable module: {module_name}")
    # Validate all invariants, then return only changed indices and restore values.
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run: `pytest -q tests/test_channel_constraint.py`

Expected: all tests pass.

- [ ] **Step 6: Review and commit the coherent helper**

Run: `git status --short && git diff --check && git diff`

Commit: `git add lib/channel_constraint.py tests/test_channel_constraint.py && git commit -m "feat: add channel protection mask constraint"`

### Task 2: Independent compact delta artifacts

**Files:**
- Modify: `tests/test_channel_constraint.py`
- Modify: `lib/channel_constraint.py`

**Interfaces:**
- Produces: `select_experiment_channels(hidden_size: int, protected: int, random_count: int, seed: int) -> list[int]`
- Produces: `new_delta_bundles(channels: list[int]) -> dict[int, dict]`
- Produces: `record_module_deltas(bundles: dict, module_name: str, weight: Tensor, score: Tensor, baseline_mask: Tensor) -> None`
- Produces: `save_delta_bundles(bundles: dict, output_dir: Path, metadata: dict) -> list[Path]`
- Produces: `apply_channel_delta(model: nn.Module, artifact_path: Path) -> dict`

- [ ] **Step 1: Write failing tests for seeded selection and branch independence**

```python
def test_variants_are_independent_branches_from_baseline():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    score = weight.clone()
    baseline = torch.tensor([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
    ], dtype=torch.bool)
    module_name = "model.transformer.blocks.0.q_proj"
    channels = select_experiment_channels(6, protected=4, random_count=2, seed=7)
    bundles = new_delta_bundles(channels)
    record_module_deltas(bundles, "model.transformer.blocks.0.q_proj", weight, score, baseline)
    for channel in channels:
        direct = apply_delta_to_mask(baseline, bundles[channel]["modules"][module_name])
        assert not direct[:, channel].any()
        assert torch.equal(direct.sum(1), baseline.sum(1))
    assert channels[0] == 4
    assert len(set(channels)) == 3
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_channel_constraint.py`

Expected: FAIL because artifact functions are absent.

- [ ] **Step 3: Write failing round-trip/application test**

Create a toy model with a matching named linear module, save a delta with
`torch.save`, apply it to a clone of the baseline-pruned toy model, and assert
that restore values equal the original dense values while compensation indices
are zero. Apply a second channel directly to a fresh baseline clone and assert
that no first-variant changes are present.

- [ ] **Step 4: Implement compact bundle save/load/application**

Use `torch.save` for tensors and write a sibling human-readable JSON file with
global and per-module stats. Before mutation, validate module presence, exact
shape, zero restore targets, and nonzero compensation targets. Aggregate:

```python
protected_already_survived_in_baseline = protected_total - protected_restored
mask_difference_count = protected_restored + compensation_pruned
mask_difference_fraction = mask_difference_count / total_prunable_weights
```

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_channel_constraint.py`

Expected: all tests pass.

- [ ] **Step 6: Review and commit artifact support**

Run: `git status --short && git diff --check && git diff`

Commit: `git add lib/channel_constraint.py tests/test_channel_constraint.py && git commit -m "feat: persist independent channel mask deltas"`

### Task 3: Wire Wanda and Sink-Aware without changing scores

**Files:**
- Modify: `lib/prune_llada.py`
- Modify: `main_llada.py`
- Modify: `tests/test_channel_constraint.py`

**Interfaces:**
- CLI produces baseline artifacts with `--delta_dir`, `--protect_channel`, and `--random_channel_count`.
- CLI materializes a branch with `--apply_channel_delta`.
- `--skip_ppl` skips unrelated PPL evaluation but not sparsity verification.
- Produces: `validate_delta_request(sparsity_ratio: float, sparsity_type: str, use_variant: bool, prune_method: str, channel: int, hidden_size: int) -> None`

- [ ] **Step 1: Add failing request-validation and shared-score-input tests**

Assert that delta generation accepts only unstructured, non-variant, 75%
`wanda`/`sink` requests with an in-range channel. Also pass two different score
tensors through `record_module_deltas` and assert that the same constraint
logic is used while compensation indices follow each supplied score. This
proves Wanda and Sink share mask logic without sharing or modifying scores.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_channel_constraint.py`

Expected: FAIL because `validate_delta_request` does not exist.

- [ ] **Step 3: Add CLI arguments and input validation**

```python
parser.add_argument("--delta_dir")
parser.add_argument("--protect_channel", type=int, default=3848)
parser.add_argument("--random_channel_count", type=int, default=0)
parser.add_argument("--apply_channel_delta")
parser.add_argument("--skip_ppl", action="store_true")
```

Reject delta generation for structured pruning, `use_variant`, methods other
than `wanda`/`sink`, non-75% sparsity, or out-of-range channels.

- [ ] **Step 4: Record deltas immediately after unchanged baseline masks**

In `prune_wanda`, call the recorder after the existing `W_mask` selection and
before `weight.data[W_mask] = 0`. In `prune_sink`, do the same after its existing
`topk`/`scatter_` block. Use canonical names
`model.transformer.blocks.{i}.{name}`. Do not replace score or mask selection
expressions.

- [ ] **Step 5: Add baseline application and reporting flow**

When `--apply_channel_delta` is present, load the requested baseline model,
apply exactly one artifact, run `check_sparsity`, write the application JSON,
and save the variant. Every variant command must point to the immutable baseline
checkpoint rather than a prior variant.

- [ ] **Step 6: Run unit and CLI sanity checks**

Run:

```bash
pytest -q tests/test_channel_constraint.py
python main_llada.py --help
python -m py_compile main_llada.py lib/prune_llada.py lib/channel_constraint.py
```

Expected: tests pass, new flags appear, compilation succeeds.

- [ ] **Step 7: Review and commit pruning integration**

Run: `git status --short && git diff --check && git diff`

Commit: `git add main_llada.py lib/prune_llada.py tests/test_channel_constraint.py && git commit -m "feat: capture pruning channel deltas"`

### Task 4: Fixed-sample activation diagnostic

**Files:**
- Create: `activation_diagnostic.py`
- Modify: `tests/test_channel_constraint.py`

**Interfaces:**
- Produces: `collect_channel_activations(model, batches, channel: int, block_indices: tuple[int, ...]) -> dict[int, float]`
- CLI writes Dense/baseline/protected JSON measurements.

- [ ] **Step 1: Write a failing toy-block hook test**

Feed two known batches through a toy model whose blocks return `(tensor,
None)`. Assert exact post-block `mean(abs(output[:, :, channel]))` at blocks 0,
15, and 31 and assert hooks are removed after collection.

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_channel_constraint.py -k activation`

Expected: FAIL because `activation_diagnostic` does not exist.

- [ ] **Step 3: Implement hooks and the eight-sample CLI**

Reuse `lib.data.get_loaders("wikitext2", nsamples=8, seed=..., seqlen=...,
tokenizer=...)`. Accumulate absolute sums and element counts only for residual
channel 3848 at blocks `(0, 15, 31)`, then write JSON.

- [ ] **Step 4: Run tests and compilation**

Run:

```bash
pytest -q tests/test_channel_constraint.py
python -m py_compile activation_diagnostic.py
```

Expected: all pass.

- [ ] **Step 5: Review and commit diagnostics**

Run: `git status --short && git diff --check && git diff`

Commit: `git add activation_diagnostic.py tests/test_channel_constraint.py && git commit -m "feat: measure protected channel activations"`

### Task 5: Validate the repository GSM8K path

**Files:**
- Modify if necessary: `eval_llada.py`
- Modify if necessary: `tests/test_channel_constraint.py`

**Interfaces:**
- Existing `llada_dist` evaluation protocol remains unchanged.

- [ ] **Step 1: Run a one-example dense GSM8K smoke test**

Run:

```bash
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true accelerate launch eval_llada.py \
  --tasks gsm8k --limit 1 --model llada_dist \
  --model_args model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024 \
  --output_path codex/channel_3848/results/smoke-dense
```

Expected: one sample completes and a result JSON is written.

- [ ] **Step 2: If it fails, use systematic debugging and add the smallest failing regression test**

The currently visible likely failure is the unconditional
`self.accelerator.wait_for_everyone()` call when single-process initialization
sets `self.accelerator = None`. Confirm from the traceback before changing it.

- [ ] **Step 3: Apply only the demonstrated root-cause fix**

```python
if self.accelerator is not None:
    self.accelerator.wait_for_everyone()
```

- [ ] **Step 4: Re-run the regression test and one-example smoke test**

Expected: both pass.

- [ ] **Step 5: Commit only if a code fix was required**

Commit: `git add eval_llada.py tests/test_channel_constraint.py && git commit -m "fix: support single-process GSM8K evaluation"`

### Task 6: Generate and verify the immutable Wanda baseline

**Files:**
- Generated: `codex/channel_3848/pruned_weights/wanda75-baseline/`
- Generated: `codex/channel_3848/results/wanda75-deltas/`
- Generated: `codex/channel_3848/results/wanda75-pruning.log`

**Interfaces:**
- Consumes the committed pruning implementation.
- Produces one baseline checkpoint and four compact channel artifacts.

- [ ] **Step 1: Record preflight resources and model config**

Run `nvidia-smi`, `df -h .`, and capture the official downloaded `config.json`
in the experiment log. Require enough space for the model cache, two retained
baselines, and one temporary variant.

- [ ] **Step 2: Run the immutable Wanda pruning job**

```bash
python main_llada.py \
  --model GSAI-ML/LLaDA-8B-Base \
  --prune_method wanda --sparsity_ratio 0.75 --sparsity_type unstructured \
  --seed 0 --nsamples 128 --protect_channel 3848 --random_channel_count 3 \
  --delta_dir codex/channel_3848/results/wanda75-deltas \
  --save_model codex/channel_3848/pruned_weights/wanda75-baseline --skip_ppl
```

- [ ] **Step 3: Verify artifact and checkpoint invariants**

Check all four artifact JSON files. Require per module and globally:

```text
protected_restored == compensation_pruned
mask_difference_count == 2 * protected_restored
baseline_pruned == constrained_pruned
actual baseline sparsity == 0.75
protected_total == 1,703,936
```

Also verify the three random channels are distinct and exclude 3848.

- [ ] **Step 4: Commit the experiment configuration before evaluation**

Force-add only the small JSON configuration/diagnostic summary, not checkpoints
or tensor artifacts ignored under `results/`.

Commit message: `exp: record Wanda channel constraint masks`

### Task 7: Wanda 64-example sanity and mandatory primary comparison

**Files:**
- Generated: `codex/channel_3848/pruned_weights/wanda75-protect3848/` (temporary)
- Generated: `codex/channel_3848/results/gsm8k-wanda75-64/`
- Generated: `codex/channel_3848/results/gsm8k-wanda75-protect3848-64/`

- [ ] **Step 1: Materialize channel 3848 from the immutable baseline**

```bash
python main_llada.py \
  --model codex/channel_3848/pruned_weights/wanda75-baseline \
  --apply_channel_delta codex/channel_3848/results/wanda75-deltas/channel-3848.pt \
  --save_model codex/channel_3848/pruned_weights/wanda75-protect3848 --skip_ppl
```

- [ ] **Step 2: Verify actual zeros and application report**

Require exactly the same total and module-wise zero counts as the Wanda
baseline and require every protected weight to be nonzero.

- [ ] **Step 3: Evaluate the same fixed 64 GSM8K examples**

Run the repository command twice, changing only `model_path` and `output_path`:

```bash
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true accelerate launch eval_llada.py \
  --tasks gsm8k --limit 64 --model llada_dist \
  --model_args model_path=MODEL_PATH,gen_length=1024,steps=1024,block_length=1024 \
  --output_path OUTPUT_PATH
```

- [ ] **Step 4: Treat results only as sanity**

Record paired counts and inspect generated outputs for gross corruption. Do not
stop merely because protect-3848 ties or loses by one sample.

- [ ] **Step 5: Run full GSM8K for both primary models**

Repeat the exact command without `--limit` for the immutable Wanda baseline and
protect-3848 variant. If measured runtime makes full evaluation temporarily
infeasible, run fixed `--limit 256` for both before deciding; never use 64 as
the efficacy gate.

- [ ] **Step 6: Persist metrics, then remove only the generated temporary variant checkpoint**

Retain the baseline, delta artifact, result JSON, logs, and diagnostic summary.

- [ ] **Step 7: Commit the primary result summary**

Commit message: `exp: record Wanda channel 3848 GSM8K comparison`

### Task 8: Random controls, Dense reference, and Sink-Aware follow-up

**Files:**
- Generated sequentially: one temporary variant checkpoint at a time
- Generated: `codex/channel_3848/results/` evaluation and activation JSON

- [ ] **Step 1: If the primary comparison has a signal, evaluate Dense full GSM8K**

Use the identical repository protocol with
`model_path=GSAI-ML/LLaDA-8B-Base`.

- [ ] **Step 2: Materialize and evaluate three random controls independently**

For each recorded random channel, load
`codex/channel_3848/pruned_weights/wanda75-baseline` fresh, apply only that channel's
artifact, verify invariants, evaluate, persist results, and remove that one
temporary checkpoint before the next channel. Report each result and the mean.

- [ ] **Step 3: Generate one immutable Sink-Aware baseline and channel-3848 delta**

Run `main_llada.py` with `--prune_method sink`, the same 75% ratio, repository
Sink defaults, `--random_channel_count 0`, and a distinct baseline/delta output
path. Verify the same invariants.

- [ ] **Step 4: Evaluate Sink baseline and protect-3848**

Use the same 64-example sanity policy followed by the appropriate full primary
comparison. The constraint helper and artifact application must be identical to
Wanda's.

- [ ] **Step 5: Run activation diagnostics**

Run `activation_diagnostic.py` on Dense, each evaluated baseline, and each
channel-3848 variant using seed 0, eight Wikitext-2 sequences, blocks 0/15/31,
and channel 3848.

- [ ] **Step 6: Commit the reached control and diagnostic summaries**

Commit message: `exp: record channel protection controls`

### Task 9: Final verification and causal report

**Files:**
- Create: `codex/channel_3848/results/report.md` (force-added despite ignore rule)

- [ ] **Step 1: Run fresh code verification**

```bash
pytest -q tests/test_channel_constraint.py
python -m py_compile main_llada.py lib/prune_llada.py lib/channel_constraint.py activation_diagnostic.py eval_llada.py
git diff --check HEAD
```

- [ ] **Step 2: Re-read all stored result JSON and recompute tables**

Do not transcribe console output by hand. The report table must include GSM8K,
absolute percentage-point delta, restored/protected, already-survived/protected,
actual sparsity, mask-difference fraction, random-control mean, and block
activation magnitudes.

- [ ] **Step 3: Assign Case A, B, C, or D conservatively**

Interpret activation preservation only together with downstream accuracy.
Separate a 3848-specific effect from the generic cost of protecting any one
channel using all three random controls.

- [ ] **Step 4: Review and commit the final report**

Run: `git status --short && git diff --check && git diff -- codex/channel_3848/results/report.md`

Commit: `git add -f codex/channel_3848/results/report.md && git commit -m "exp: report LLaDA channel 3848 causal experiment"`
