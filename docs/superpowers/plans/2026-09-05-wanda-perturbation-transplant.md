# Wanda Perturbation Transplant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate location, direction, and location-by-direction effects for the natural block30/block31 ff_out Wanda perturbations under state-wise matched relative norm.

**Architecture:** Reconstruct the two exact finite residual perturbations from the frozen masks and states, validate native residual injection against the original module-only outputs, then execute two dense three-row suffix batches per state. Persist full frozen perturbations, compact four-cell results, and sequence-cluster bootstrap summaries.

**Tech Stack:** Python 3.12, PyTorch 2.8, SciPy, pandas, pytest.

**Spec:** Current user request, “Spike — Perturbation Transplant: Location vs Direction in Terminal Wanda Failure.”

## Global Constraints

- Use all existing 40 states, existing masks, pinned model, and no new data.
- Preserve fixed three-row same-path batches at both locations.
- Stop unless unmodified native residual injection reproduces logits, KL, and loss.
- Primary cells use only geometric-mean relative-norm matching.
- No new masks, backward/JVP/Jacobian, random directions, criterion, or downstream benchmark.

---

### Task 1: Matched perturbation and factorial primitives

**Files:**
- Modify: `experiments/wanda_failure_characterization/propagation_core.py`
- Modify: `tests/test_wanda_failure_characterization.py`

- [ ] Write failing tests for geometric-mean target norm, achieved norm, and the preregistered factorial contrasts.
- [ ] Implement `matched_perturbation` and `factorial_contrasts` in FP32.
- [ ] Run focused tests.

### Task 2: Frozen-delta reconstruction and native gate

**Files:**
- Create: `experiments/wanda_failure_characterization/run_perturbation_transplant.py`

- [ ] Reconstruct mask hashes and exact state-wise block30/block31 ff_out deltas in the original three-way suffix path.
- [ ] Inject each unmodified delta into its dense native residual location and compare logits/KL/loss to original module-only outputs.
- [ ] Stop on tolerance failure; otherwise persist `transplant_deltas.pt` and hashes.

### Task 3: Matched 2x2 collection

**Files:**
- Modify: `experiments/wanda_failure_characterization/run_perturbation_transplant.py`

- [ ] Compute state-specific `tau=sqrt(r30*r31)` and both unit directions.
- [ ] Run `[sham,D30,D31]` at L30 and L31 with dense suffixes only.
- [ ] Record norm errors, external sham drift, repeatability, KL/loss/top1/confidence, final-hidden/logit magnitudes, and model SHA.

### Task 4: Clustered analysis and report

**Files:**
- Create: `experiments/wanda_failure_characterization/analyze_transplant.py`
- Create: `experiments/wanda_failure_characterization/transplant_report.md`

- [ ] Compute paired contrasts, sequence/timestep summaries, fixed-seed sequence-cluster bootstrap CIs, magnitude-conditioned descriptive ratios, and natural anchors.
- [ ] Assign exactly one Outcome A--E.
- [ ] Persist machine-readable tables/manifests and write one recommended next diagnostic.
- [ ] Run focused tests and artifact/hash/weight/reproduction verification.
