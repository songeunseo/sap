# D31 Token-Row Pairing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the frozen analysis-only exhaustive cyclic and masked-class-preserving D31 token-row permutation diagnostic.

**Architecture:** Freeze deterministic permutation manifests before outcomes, reuse the existing L31 injection/readout utilities, and persist scalar sufficient statistics for every state/permutation/cell. Analyze only receiver-aggregated effects with sequence-cluster bootstrap and preregistered contrasts.

**Tech Stack:** Python 3.12, PyTorch 2.8, NumPy, pandas, SciPy, tmux.

**Spec:** Current user request, “Exact Token-Row Pairing in State-Specific D31 Failure.”

## Global Constraints

- Reuse the exact 40 states, delta31, tau, and L31 path; no new state or mask.
- Complete all 10,200 unrestricted conditions plus every deduplicated class-preserving permutation.
- No backward/JVP/Jacobian, downstream benchmark, criterion design, or automatic follow-up.
- Primary outcome is same-path masked-token KL.

---

### Task 1: Freeze permutations and mathematical helpers

**Files:**
- Create: `experiments/wanda_failure_characterization/freeze_token_row_pairing.py`
- Modify: `experiments/wanda_failure_characterization/propagation_core.py`
- Modify: `tests/test_wanda_failure_characterization.py`

**Interfaces:**
- Produces `token_row_pairing_manifest.json` and tested `token_row_pairing_contrasts(cells)`.

- [ ] Add tests for NN/NS/SN/SS contrast formulas and deterministic class-preserving permutations.
- [ ] Run the focused test and confirm the new imports fail.
- [ ] Implement helpers and manifest generation with hashes, zero-row gate, 255 cyclic shifts, and per-state deduplicated class permutations.
- [ ] Run focused tests and freeze the manifest before model evaluation.

### Task 2: Execute exact L31 interventions

**Files:**
- Create: `experiments/wanda_failure_characterization/run_token_row_pairing.py`

**Interfaces:**
- Consumes frozen manifest, `transplant_deltas.pt`, `perturbation_transplant.pt`, and held-out states.
- Produces sharded per-state result files and `token_row_pairing_run_manifest.json`.

- [ ] Implement factorization, NN/NS/SN/SS construction, exact norm matching, and batch-5 evaluation.
- [ ] Gate batch-5 against batch-3 and NN against frozen metrics before any shifted result is accepted.
- [ ] Persist each state atomically so the tmux job is resumable without changing results.
- [ ] Launch the complete run in tmux with a persistent log and inspect initial gates/progress.

### Task 3: Analyze and report

**Files:**
- Create: `experiments/wanda_failure_characterization/analyze_token_row_pairing.py`
- Create: `experiments/wanda_failure_characterization/token_row_pairing_report.md`

**Interfaces:**
- Consumes all completed state shards.
- Produces per-shift/per-receiver tables, bootstrap CIs, stability and magnitude-control tables, hashes, and one Outcome A–E.

- [ ] Aggregate shifts within receiver before inference and run sequence-cluster bootstrap with seed 20260905 and 10,000 replicates.
- [ ] Produce unrestricted/class-preserving, timestep, sequence, distance, retention, and downstream-magnitude results.
- [ ] Write the report with facts, causal results, associations, interpretation, speculation, and exactly one next diagnostic.
- [ ] Run focused tests and artifact/hash/completeness verification before reporting completion.
