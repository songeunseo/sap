# D31 State-Specificity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether matched D31@L31 damage requires pairing each frozen perturbation with its source state.

**Architecture:** Freeze cyclic and same-timestep donor mappings before evaluation, verify stored delta hashes, reconstruct the previous batch-three L31 sham residual without masks, reproduce native results, then run fixed `[sham,native,foreign]` final-norm/readout batches. Analyze clustered pairing advantages and direction cosines from persisted per-state results.

**Tech Stack:** Python 3.12, PyTorch 2.8, SciPy, pandas, pytest.

**Spec:** Current user request, “Spike — Is the D31 Harmful Perturbation State-Specific or State-General?”

## Global Constraints

- Reuse the exact 40 states, stored delta31 tensors, tau, and L31 location.
- Freeze `donor(i)=(i+1)%40` and same-timestep next-sequence mappings before outcomes.
- Use matched receiver-relative norm and fixed three-row same-path batches.
- Stop unless native results reproduce the previous matched D31@L31 artifact.
- No masks, pruning, random directions, backward/JVP/Jacobian, or downstream evaluation.

---

### Task 1: Mapping primitives

**Files:** Modify `propagation_core.py`; modify `tests/test_wanda_failure_characterization.py`.

- [ ] Add failing tests for cyclic and same-timestep donor maps.
- [ ] Implement deterministic mapping helpers and rerun focused tests.

### Task 2: Freeze mappings and native gate

**Files:** Create `freeze_state_swap_mappings.py`; create `run_state_swap.py`.

- [ ] Persist both mappings with state metadata and hashes before model execution.
- [ ] Verify delta hashes and reconstruct batch-three dense L31 residuals without masks.
- [ ] Reproduce previous per-state native KL/loss/final-hidden/logit results; stop on mismatch.

### Task 3: Causal collection

**Files:** Modify `run_state_swap.py`.

- [ ] Run primary cyclic batches, then same-timestep batches, recording same-path outcomes, norm errors, sham drift, repeatability, cosine, and weight hashes.
- [ ] Persist compact results and manifest.

### Task 4: Cluster analysis and report

**Files:** Create `analyze_state_swap.py`; create `state_swap_report.md`.

- [ ] Compute pairing advantage, retained damage, fixed-seed sequence-cluster bootstrap, timestep/sequence tables, downstream magnitude contrasts, and cosine associations.
- [ ] Assign one Outcome A--E and exactly one next diagnostic.
- [ ] Verify tests, artifact hashes/shapes, mapping freeze, native gate, norms, and unchanged weights.
