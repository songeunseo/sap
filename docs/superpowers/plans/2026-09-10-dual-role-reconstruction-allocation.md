# Dual-Role Reconstruction Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the pre-registered analysis-only diagnostic that tests whether masked/unmasked role-bottleneck reconstruction predicts useful projection sparsity allocation better than aggregate reconstruction.

**Architecture:** Add a self-contained `experiments/dlm_dual_role_allocation` package. A GPU collector replays the frozen dense block prefixes and existing six Standard-Wanda masks, stops at the target Linear, and checkpoints masked/unmasked numerator and denominator scalars. A CPU analyzer validates exact reconstruction against historical curves, builds fold-specific aggregate and role allocations, evaluates them with existing per-state oracle KL curves, fits the frozen nested OLS comparison, and emits the pre-registered decision artifacts.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy, unittest, existing LLaDA experiment helpers.

**Spec:** `docs/superpowers/specs/2026-09-10-dual-role-reconstruction-allocation-design.md`

## Global Constraints

- Model is `GSAI-ML/LLaDA-8B-Base` at revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`.
- Projection order is the repository's existing 32 blocks × 7 projections = 224.
- Candidate masks and sparsity grid are frozen to `experiments/projection_capacity_allocation_65/candidate_mask_manifest.json` and 50/55/60/65/70/75%.
- Weight ranking remains the existing 80-state Standard Wanda ranking; this experiment changes no weight mask within a projection/sparsity pair.
- Construction states are the frozen 8 sequences × 10 timesteps = 80 allocation-calibration states.
- Budget is the actual Uniform-65 row-floor count 4,536,008,704 / 6,979,321,856 weights.
- `E_role=max(E_M,E_U)` is fixed; no epsilon, learned coefficient, monotone envelope, module exception, timestep-specific mask, or result-dependent rescue is allowed.
- GPU collection uses GPU 0 only when it is free, runs in tmux, checkpoints per projection, and does not use GPU 1.
- Existing historical artifacts are read-only. Human reporting goes to Obsidian; repository outputs are machine-readable provenance/results.
- No full sparse model, held-out DLM evaluation, or GSM8K is run in this diagnostic.
- Git commit steps are intentionally omitted because the repository has no configured author identity and the user approved proceeding without commits.

---

### Task 1: Role-statistic numerical core

**Files:**
- Create: `experiments/dlm_dual_role_allocation/__init__.py`
- Create: `experiments/dlm_dual_role_allocation/core.py`
- Test: `tests/test_dlm_dual_role_allocation.py`

**Interfaces:**
- Consumes: dense and six sparse Linear outputs shaped `[7, sequence, output]`, a boolean position mask shaped `[sequence]`, six-point per-state raw records, and module shapes.
- Produces: `partition_role_sums(outputs, masked_positions) -> dict`, `pool_role_curves(raw, state_ids) -> dict`, `role_contrast(masked, unmasked) -> ndarray`, `allocation_mask_xor(a, b, manifest) -> dict`, `bootstrap_mean_ci(values, resamples=20000, seed=0) -> dict`.

- [ ] **Step 1: Write failing unit tests for exact token partitioning and zero-policy**

```python
def test_partition_role_sums_reconstructs_aggregate():
    dense = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
    sparse = dense + torch.tensor([[[1., 0.], [0., 2.], [3., 0.]]])
    outputs = torch.cat([dense, sparse.repeat(6, 1, 1)], dim=0)
    result = partition_role_sums(outputs, torch.tensor([True, False, True]))
    assert result["num_masked"][0] + result["num_unmasked"][0] == 10.0
    assert result["den_masked"] + result["den_unmasked"] == 91.0

def test_partition_role_sums_rejects_empty_group_and_zero_denominator():
    outputs = torch.ones(7, 3, 2)
    with pytest.raises(ValueError, match="empty token role"):
        partition_role_sums(outputs, torch.ones(3, dtype=torch.bool))
    outputs[:, 0] = 0
    with pytest.raises(ValueError, match="zero role denominator"):
        partition_role_sums(outputs, torch.tensor([True, False, False]))
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the package is absent**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k partition`

Expected: FAIL during import of `experiments.dlm_dual_role_allocation.core`.

- [ ] **Step 3: Implement role partitioning with FP32 squares and Python/FP64 scalars**

```python
def partition_role_sums(outputs, masked_positions):
    outputs = outputs.float()
    mask = torch.as_tensor(masked_positions, dtype=torch.bool, device=outputs.device)
    if outputs.ndim != 3 or outputs.shape[0] != 7 or mask.shape != (outputs.shape[1],):
        raise ValueError("expected seven variants and one position mask")
    if not bool(mask.any()) or not bool((~mask).any()):
        raise ValueError("empty token role")
    dense = outputs[0]
    delta2 = (outputs[1:] - dense.unsqueeze(0)).square()
    dense2 = dense.square()
    den_m = float(dense2[mask].sum().double().cpu())
    den_u = float(dense2[~mask].sum().double().cpu())
    if den_m == 0.0 or den_u == 0.0:
        raise ValueError("zero role denominator")
    return {
        "num_masked": [float(x.double().cpu()) for x in delta2[:, mask].sum((1, 2))],
        "den_masked": den_m,
        "num_unmasked": [float(x.double().cpu()) for x in delta2[:, ~mask].sum((1, 2))],
        "den_unmasked": den_u,
    }
```

- [ ] **Step 4: Add tests and implementations for pooling, contrast, bootstrap, and exact nested-mask XOR**

```python
def test_role_contrast_has_no_epsilon_and_zero_zero_is_zero():
    got = role_contrast(np.array([0., 3., 2.]), np.array([0., 1., 2.]))
    np.testing.assert_allclose(got, [0., .5, 0.])

def test_pool_role_curves_uses_ratio_of_sums_not_mean_of_ratios():
    raw = synthetic_raw_records()
    pooled = pool_role_curves(raw, state_ids=[0, 1])
    assert pooled["masked"][0, 0] == pytest.approx((1 + 9) / (2 + 18))

def test_nested_mask_xor_uses_selected_pruned_counts():
    manifest = synthetic_nested_manifest()
    result = allocation_mask_xor([0, 2], [1, 1], manifest)
    assert result["xor_pruned_weights"] == 4
    assert result["xor_fraction_of_prunable_weights"] == pytest.approx(4 / 40)
```

Implement `pool_role_curves` by summing `num_masked`, `den_masked`, `num_unmasked`, and `den_unmasked` over the requested states before division. Define zero/zero contrast as exactly zero and otherwise `abs(M-U)/(M+U)`. Compute nested-mask XOR for a module as the absolute difference between the two selected row-floor pruned counts because all six masks share one stable Wanda ordering.

- [ ] **Step 5: Run the numerical-core tests**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k 'partition or contrast or pool or xor or bootstrap'`

Expected: PASS.

---

### Task 2: Frozen-input and checkpoint validation

**Files:**
- Create: `experiments/dlm_dual_role_allocation/io.py`
- Modify: `tests/test_dlm_dual_role_allocation.py`

**Interfaces:**
- Consumes: capacity config/state verification/candidate manifest/curve files and per-projection checkpoints.
- Produces: `load_frozen_inputs(root) -> FrozenInputs`, `validate_checkpoint(payload, expected) -> None`, `atomic_write_json(path, payload) -> None`, `source_receipt(paths) -> dict`.

- [ ] **Step 1: Write failing tests for source ordering, hashes, row-floor counts, and checkpoint identity**

```python
def test_load_frozen_inputs_rejects_module_order_mismatch(tmp_path):
    fixture = write_frozen_fixture(tmp_path)
    fixture["candidate"]["entries"].reverse()
    write_json(fixture["candidate_path"], fixture["candidate"])
    with pytest.raises(RuntimeError, match="module ordering"):
        load_frozen_inputs(tmp_path)

def test_validate_checkpoint_rejects_state_digest_mismatch():
    with pytest.raises(RuntimeError, match="state digest"):
        validate_checkpoint({"state_digest": "bad"}, {"state_digest": "good"})
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k 'frozen or checkpoint'`

Expected: FAIL because `io.py` does not exist.

- [ ] **Step 3: Implement strict frozen-input receipts**

The loader must assert all of the following before returning:

```text
capacity config model id/revision match the spec
state_verification status=verified and disjoint=true
80 calibration states, 8 unique sequences, exact timesteps 0.05..0.95
candidate manifest contains 224 entries in the same name/index/shape order as capacity curves
each entry contains the exact six-point grid and valid mask file/hash metadata
sum of level-3 row-floor counts equals 4,536,008,704
sum of module weights equals 6,979,321,856
all source file SHA-256 values are captured in the returned receipt
```

`validate_checkpoint` must bind every checkpoint to source receipt digest, model digest, state digest, module index/name/shape, grid, and candidate mask hashes. `atomic_write_json` writes a sibling `.tmp` and replaces the destination.

- [ ] **Step 4: Run validation tests**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k 'frozen or checkpoint'`

Expected: PASS.

---

### Task 3: GPU role-reconstruction collector

**Files:**
- Create: `experiments/dlm_dual_role_allocation/collect.py`
- Modify: `tests/test_dlm_dual_role_allocation.py`

**Interfaces:**
- Consumes: `FrozenInputs`, the historical dense model loader, dense block prefixes, existing packed Wanda masks.
- Produces: `collect_projection(model, module, prefix_states, states, masks) -> list[dict]`, `collect_all() -> None`, and `runtime/role_stats/<canonical_name>.json` checkpoints.

- [ ] **Step 1: Write failing tests for target-stop control flow and seven-variant equivalence**

```python
def test_target_stop_collects_output_without_running_suffix():
    block = ToyBlock(counter={"suffix": 0})
    records = collect_projection_from_block(block, block.target, toy_prefixes(), toy_states(), toy_masks())
    assert len(records) == 2
    assert block.counter["suffix"] == 0
    assert records[0]["num_masked"] == expected_masked_numerators()
```

The toy block places a counter immediately after the target Linear. The implementation must stop via a private `_TargetReached` exception raised only after the forward hook has copied the six numerator values and two denominators to CPU.

- [ ] **Step 2: Run the collector unit test and confirm failure**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k target_stop`

Expected: FAIL because collector control flow is absent.

- [ ] **Step 3: Implement the target-stop collector**

For each state:

```python
prefix = prefixes[block_index][state_index].expand(7, -1, -1).contiguous()
masks = [None] + [read_mask(candidate_row, level, module.weight.device) for level in range(6)]

def hook(mod, inp, _out):
    variants = masked_linear_variants(inp[0], mod.weight, mod.bias, masks)
    captured.update(partition_role_sums(variants, persisted_mask[0]))
    raise _TargetReached

try:
    _suffix_logits(model, prefix, block_index)
except _TargetReached:
    pass
finally:
    handle.remove()
```

Require all seven target inputs to be bitwise equal before applying masks. If `_suffix_logits` returns normally, or an unrelated exception occurs, fail. Store state index, sequence index, timestep, `p_mask`, role sums, and reconstructed aggregate error for every state and sparsity level.

- [ ] **Step 4: Implement resumable all-projection collection and provenance checks**

Collect dense prefixes exactly as the capacity experiment does. Before skipping a checkpoint, validate its complete identity and require 80 states × 6 levels. Emit JSONL progress events `dense_prefixes_complete`, `role_projection_complete`, `role_projection_skipped`, and `collection_complete`. Assert `model_sha(model)==DENSE_SHA` before collection, after each projection, and at completion.

- [ ] **Step 5: Run collector tests plus historical helper tests**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py tests/test_wanda_failure_characterization.py tests/test_projection_capacity_allocation65.py`

Expected: PASS.

---

### Task 4: Reconstruction sanity and descriptive analysis

**Files:**
- Create: `experiments/dlm_dual_role_allocation/analyze.py`
- Modify: `tests/test_dlm_dual_role_allocation.py`

**Interfaces:**
- Consumes: validated role checkpoints and historical `capacity_curves_raw.json`.
- Produces: `validate_reconstruction(raw, historical) -> dict`, `describe_roles(...) -> dict`, `role_reconstruction_raw.json`, `role_distributions.json`.

- [ ] **Step 1: Write failing tests for 224×6×80 sanity thresholds and raw distribution fields**

```python
def test_reconstruction_sanity_enforces_both_thresholds():
    historical = np.ones((2, 3, 6))
    reconstructed = historical.copy()
    reconstructed[0, 0, 0] += 2e-7
    with pytest.raises(RuntimeError, match="absolute reconstruction"):
        validate_reconstruction(reconstructed, historical)

def test_describe_roles_counts_dominance_and_negative_marginals():
    report = describe_roles(masked_curves(), unmasked_curves(), metadata())
    assert sum(report["dominance_counts"].values()) == 12
    assert report["negative_marginal_count"] == 1
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k 'reconstruction_sanity or describe_roles'`

Expected: FAIL because analysis helpers are absent.

- [ ] **Step 3: Implement exact historical reconstruction validation**

Build arrays in module × state × level order. Compare `(num_M+num_U)/(den_M+den_U)` to every historical `local_reconstruction_error`. Fail if max absolute error exceeds `1e-7` or max relative error exceeds `1e-5`; for a zero historical value require exact reconstructed zero. Persist maxima and the offending module/state/level indices.

- [ ] **Step 4: Implement pre-registered descriptive outputs**

For each sparsity and marginal increment report min/p10/p25/median/p75/p90/max/mean/std/CV for `E_M`, `E_U`, `E_all`, `E_role`, contrast, and `E_M/E_U`. Report negative marginals, monotonicity violations, layer/type/quartile/sequence/timestep breakdowns, dominance counts, sequence/timestep Spearman stability, bidirectional fold reproducibility, and the top/bottom 20 contrast increments. Preserve raw non-monotone values.

- [ ] **Step 5: Run descriptive analysis tests**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k 'reconstruction_sanity or describe_roles or stability'`

Expected: PASS.

---

### Task 5: Frozen cross-fit allocations and nested OLS gate

**Files:**
- Create: `experiments/dlm_dual_role_allocation/models.py`
- Modify: `experiments/dlm_dual_role_allocation/analyze.py`
- Modify: `tests/test_dlm_dual_role_allocation.py`

**Interfaces:**
- Consumes: role/aggregate curves, module metadata, per-state oracle KL, fixed folds `[0,1,2,3] -> [4,5,6,7]` and reverse.
- Produces: `fit_nested_ols(...) -> dict`, `evaluate_crossfit(...) -> dict`, `apply_diagnostic_gate(...) -> dict`, `redundancy_analysis.json`, `crossfit_allocations.json`, `decision.json`.

- [ ] **Step 1: Write failing tests for construction-only standardization and fixed design columns**

```python
def test_nested_ols_uses_fixed_reference_categories_and_train_scaling():
    fit = fit_nested_ols(train_fixture(), layers(), types())
    assert fit["design_columns"][:3] == ["intercept", "layer_1", "layer_2"]
    assert "layer_0" not in fit["design_columns"]
    assert "type_attn_out" not in fit["design_columns"]
    assert fit["baseline_columns"][-1] == "aggregate_marginal_per_parameter"
    assert fit["extended_columns"][-1] == "role_minus_aggregate_marginal_per_parameter"
```

- [ ] **Step 2: Write failing tests for exact allocation evaluation and all-five gate conjunction**

```python
def test_gate_fails_when_only_mask_xor_is_below_one_percent():
    evidence = passing_gate_fixture()
    evidence["mask_xor_fraction"] = .0099
    result = apply_diagnostic_gate(evidence)
    assert result["decision"] == "NOT SUPPORTED"
    assert result["criteria"]["mask_xor_at_least_one_percent"] is False
```

- [ ] **Step 3: Run the model/gate tests and confirm failure**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k 'nested_ols or gate or crossfit'`

Expected: FAIL because model functions are absent.

- [ ] **Step 4: Implement the frozen nested OLS**

Flatten construction module × state × increment rows. Build one-hot columns for layers 1–31 and six non-reference projection types in sorted repository type order, with `attn_out` omitted. Add standardized aggregate reconstruction marginal cost per nominal parameter to the baseline. Add standardized role-minus-aggregate marginal cost per nominal parameter only to the extended model. Means/scales come from construction rows only; reject a zero scale. Fit with `np.linalg.lstsq`, retain coefficients and scaling, and apply unchanged to validation rows. Report overall and per-sequence RMSE.

- [ ] **Step 5: Implement bidirectional conditional allocation evaluation**

Pool construction role sums for sequences 0–3 and 4–7 separately. Build `E_all` and `E_role` curves and call the existing exact `allocate(curves, shapes)` using raw marginals and repository-order ties. Verify both allocations hit 4,536,008,704 pruned weights. For every validation sequence, sum the historical per-state functional KL at each allocation's selected level over modules, then average its ten timesteps. Record `Role-Aggregate` damage and nested-mask XOR. Repeat in reverse.

- [ ] **Step 6: Implement the exact pre-registered decision**

The decision is `DIAGNOSTIC SUPPORTED` only when reconstruction/hash sanity passes, role additive damage is lower in both directions, the eight sequence damage-difference bootstrap CI upper bound is below zero, exact nested-mask XOR is at least 0.01 of all prunable weights, extended OLS RMSE is lower in both directions, and its eight-sequence RMSE-difference bootstrap CI upper bound is below zero. Persist every Boolean separately and list failed criteria. Do not run alternate formulas.

- [ ] **Step 7: Run all CPU tests**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py tests/test_dlm_allocation_structure.py tests/test_projection_capacity_allocation65.py`

Expected: PASS.

---

### Task 6: CLI, frozen config, and tmux launcher

**Files:**
- Create: `experiments/dlm_dual_role_allocation/run.py`
- Create: `experiments/dlm_dual_role_allocation/run_tmux.sh`
- Modify: `tests/test_dlm_dual_role_allocation.py`

**Interfaces:**
- Consumes: collection and analysis functions from Tasks 3–5.
- Produces: CLI phases `freeze`, `collect`, `analyze`, `all`; the required artifact directory and logs.

- [ ] **Step 1: Write a failing CLI freeze test**

```python
def test_freeze_config_contains_pre_registered_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    config = run.freeze()
    assert config["role_curve"] == "max(E_masked,E_unmasked)"
    assert config["mask_xor_gate"] == .01
    assert config["full_model_evaluation"] is False
```

- [ ] **Step 2: Run the CLI test and confirm failure**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py -k freeze_config`

Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement immutable freeze and phased CLI**

`freeze()` writes `config.json` and `state_verification.json` only if absent; otherwise it requires byte-equivalent semantic content and unchanged source hashes. `collect` requires CUDA device index 0 visibility and writes only checkpoints/raw collection artifacts. `analyze` requires all 224 valid checkpoints and performs no model load. `all` runs collection then analysis. Every phase logs one JSON event per line.

- [ ] **Step 4: Add the tmux launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
exec python -m experiments.dlm_dual_role_allocation.run all
```

The launcher itself does not create a tmux session; the documented invocation is:

```bash
tmux new-session -d -s dlm_dual_role 'bash experiments/dlm_dual_role_allocation/run_tmux.sh > experiments/dlm_dual_role_allocation/logs/all.log 2>&1'
```

- [ ] **Step 5: Run the complete unit test file and syntax checks**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py`

Expected: PASS.

Run: `python -m compileall -q experiments/dlm_dual_role_allocation`

Expected: exit code 0.

---

### Task 7: Freeze, smoke-check, launch, and report

**Files:**
- Create at runtime: `experiments/dlm_dual_role_allocation/config.json`
- Create at runtime: `experiments/dlm_dual_role_allocation/state_verification.json`
- Create at runtime: `experiments/dlm_dual_role_allocation/role_reconstruction_raw.json`
- Create at runtime: `experiments/dlm_dual_role_allocation/role_distributions.json`
- Create at runtime: `experiments/dlm_dual_role_allocation/redundancy_analysis.json`
- Create at runtime: `experiments/dlm_dual_role_allocation/crossfit_allocations.json`
- Create at runtime: `experiments/dlm_dual_role_allocation/decision.json`
- Create after completion: Obsidian `Research/DLM-Pruning/Experiments/2026-09-10 Dual-Role Reconstruction Allocation 결과.md`

**Interfaces:**
- Consumes: completed implementation and frozen historical artifacts.
- Produces: a running/completed tmux experiment and an evidence-backed Korean Obsidian report.

- [ ] **Step 1: Run the full relevant CPU regression suite**

Run: `pytest -q tests/test_dlm_dual_role_allocation.py tests/test_dlm_allocation_structure.py tests/test_projection_capacity_allocation65.py tests/test_wanda_failure_characterization.py`

Expected: all tests pass.

- [ ] **Step 2: Freeze inputs and inspect the receipt**

Run: `python -m experiments.dlm_dual_role_allocation.run freeze`

Expected: `freeze_complete` with 224 projections, 80 states, exact budget, and verified hashes.

- [ ] **Step 3: Run a one-projection smoke collection without accepting it as evidence**

Run: `CUDA_VISIBLE_DEVICES=0 python -m experiments.dlm_dual_role_allocation.run collect --stop-after 1 --smoke-output /tmp/dlm_dual_role_smoke`

Expected: one projection × 80 states × 6 levels, dense hash unchanged, historical reconstruction max absolute error ≤1e-7 and max relative error ≤1e-5. The smoke output is outside the frozen artifact directory and is not resumed into the official run.

- [ ] **Step 4: Verify GPU 0 is free and GPU 1 is not selected**

Run: `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`

Expected: no active process on physical GPU 0. If occupied, do not launch and report the blocker.

- [ ] **Step 5: Launch the official job in tmux**

Run: `mkdir -p experiments/dlm_dual_role_allocation/logs && tmux new-session -d -s dlm_dual_role 'bash experiments/dlm_dual_role_allocation/run_tmux.sh > experiments/dlm_dual_role_allocation/logs/all.log 2>&1'`

Expected: `tmux has-session -t dlm_dual_role` exits 0 and the log begins with frozen-input verification.

- [ ] **Step 6: Monitor checkpoints without polling model output tensors**

Run: `watch -n 5 'find experiments/dlm_dual_role_allocation/runtime/role_stats -name "*.json" 2>/dev/null | wc -l; tail -n 3 experiments/dlm_dual_role_allocation/logs/all.log'`

Expected: checkpoint count increases toward 224; completion emits `collection_complete` followed by `analysis_complete`.

- [ ] **Step 7: Verify artifacts and the decision after completion**

Run: `python -m experiments.dlm_dual_role_allocation.run analyze`

Expected: deterministic byte-equivalent JSON outputs on the second analysis run and a decision containing all five gate criteria.

- [ ] **Step 8: Write the Korean Obsidian report**

The report must state the hypothesis, frozen formula, raw masked/unmasked distributions, redundancy with aggregate reconstruction, exact allocation/XOR, bidirectional additive oracle differences, nested OLS validation, each gate criterion, and exactly one conclusion: `DIAGNOSTIC SUPPORTED` or `NOT SUPPORTED`. It must explicitly state that no full-model or downstream claim follows from this diagnostic.
