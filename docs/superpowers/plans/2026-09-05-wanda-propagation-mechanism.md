# Wanda Propagation Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain where the frozen 50% module-only Wanda perturbation changes from local reconstruction error into final masked-token KL damage.

**Architecture:** Reconstruct the exact persisted masks and reuse the frozen 40 states and three-way suffix batch. Instrument the existing LLaDA block with observation hooks only, reduce sham/50% tensor differences immediately to compact checkpoint statistics, and analyze a compact preregistered propagation chain with type/layer controls.

**Tech Stack:** Python 3.12, PyTorch 2.8, SciPy, pandas, pytest.

**Spec:** Current user request, “Spike — Where Does Wanda's Local Pruning Error Become Functional Damage?”

## Global Constraints

- Preserve `[sham, 50%, 75%]` batch shape and order.
- Reuse the exact model, masks, 40 frozen states, ordering, and prefix/suffix path.
- Stop unless reproduced 50% KL and loss match the persisted artifact.
- Observation only: no weight mutation, new criterion, backward/JVP, random perturbation, or downstream evaluation.
- Analyze the full 224-module 50% map; 75% is execution-path-only.

---

### Task 1: Trace reduction primitives and tests

**Files:**
- Create: `experiments/wanda_failure_characterization/propagation_core.py`
- Modify: `tests/test_wanda_failure_characterization.py`

**Interfaces:**
- Produces `tensor_pair_metrics(sham, variant, mask)` and `propagation_gain(later, baseline)`.

- [ ] Write failing tests proving exact relative-energy, masked enrichment, cosine/alignment, and safe gain semantics.
- [ ] Run the focused test and confirm failure from missing functions.
- [ ] Implement FP32 reductions without retaining full hidden tensors.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Reproduction gate and trace collector

**Files:**
- Create: `experiments/wanda_failure_characterization/run_propagation_trace.py`

**Interfaces:**
- Consumes `heldout_state_manifest.json`, `failure_map.pt`, `wanda_sufficient_statistics.pt`, and the existing model/path helpers.
- Produces `propagation_trace.pt` and `propagation_trace_manifest.json`.

- [ ] Reconstruct both masks and verify every persisted mask hash.
- [ ] Add observers at attention output, post-attention input to `ff_norm`, MLP output, block outputs, final norm output, and logits.
- [ ] Re-run a bounded state/module gate and compare same-path 50% loss/KL to the frozen failure map with exact-or-established tolerance.
- [ ] Stop on mismatch; otherwise collect all 40×224 traces while preserving the three-way path.
- [ ] Store only scalar sufficient statistics and verify model SHA before/after.

### Task 3: Mechanism analysis

**Files:**
- Create: `experiments/wanda_failure_characterization/analyze_propagation.py`

**Interfaces:**
- Consumes `propagation_trace.pt` and frozen `failure_map.pt`.
- Produces `propagation_per_module.csv`, `propagation_associations.csv`, and `propagation_analysis.json`.

- [ ] Aggregate branch, C1, later block, final hidden, and logit metrics across states.
- [ ] Calculate the five-stage correlation chain, within-type correlations, within-type standardized associations, and leave-one-layer-out fixed ridge diagnostics.
- [ ] Produce named traces for block31 ff_out/ff_proj/up_proj, block30 ff_out, and block0 v_proj.
- [ ] Classify exactly one Outcome A--E from full-map evidence.

### Task 4: Report and verification

**Files:**
- Create: `experiments/wanda_failure_characterization/propagation_report.md`

**Interfaces:**
- Consumes all persisted trace/analysis artifacts.
- Produces the required fact/association/interpretation/speculation-separated report.

- [ ] Write the report with exactly one recommended next diagnostic.
- [ ] Run all focused tests, artifact shape/hash checks, reproduction-error checks, and weight-hash checks.
- [ ] Inspect repository status and report only verified outcomes.
