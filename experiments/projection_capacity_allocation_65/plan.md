# Projection Capacity Allocation Implementation Plan

**Goal:** Execute the user's frozen oracle kill gate at nominal 65%, without searching for a new score or allocation rule.

**Spec:** The user's 16-section experiment specification in this conversation.

**Architecture:** An isolated experiment directory contains preflight verification, numerical allocation/analysis helpers, and a resumable GPU runner. Historical modules are imported read-only. Expensive work runs in tmux on GPU 0; historical artifacts are never overwritten.

## Frozen details

- Model revision: `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`.
- 224 projections in `run_failure_map.modules` insertion order.
- Grid: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75.
- Raw curves determine the allocation; no monotone envelope.
- At each step only the next increment of each projection is eligible. Rank by raw delta KL / nominal additional parameter count; break exact ties by module order.
- Match the actual uniform row-floor mask budget. Preserve feasibility of reaching that budget when accepting increments; record any feasibility-driven skips. No downstream outcomes enter this decision.
- Primary mean is the equally weighted mean of state-level masked-token KL means, matching the historical fidelity evaluator. Also report pooled token means and medians.
- Paired bootstrap: 20,000 resamples of 40 states, NumPy seed 0, percentile 95% interval. Strictly negative upper endpoint and majority of 8 sequence / 5 timestep means required.
- Calibration candidate paths use a common dense prefix and a batch of seven (dense sham plus six masks), matching the historical functional Linear intervention. Verify suffix/full equivalence. Full jointly sparse evaluation uses ordinary physical masks and identical batch-one full forwards for dense, sham, and both sparse models.

## Tasks

- [x] Verify manifests, corruption, frozen digest, reconstructed corpus span offsets, and cross-split interval disjointness. Persist `state_verification.json`; stop on failure.
- [x] Freeze `config.json`, source hashes, calibration semantics and actual uniform budget before outcomes.
- [x] Test raw negative marginal costs, parameter weighting, exact budget reachability, and gate consistency on hand-computed fixtures.
- [ ] Build six nested row-wise masks per projection from one Standard Wanda ranking; require 65% hashes to match the historical sweep when using DLM calibration.
- [ ] Collect resumable per-module calibration results with same-path sham checks; persist all 224 x 6 curves and diagnostics.
- [ ] Summarize raw damage and increments, monotonicity, layer/type breakdown; freeze exact allocation and manifests.
- [ ] Evaluate both full sparse models on all 40 held-out states; compute paired gate, descriptive allocation diagnostics, and report.
- [ ] Only after PASS, run fixed mini-100 with exact historical protocol and examples; only after directional mini improvement, run full 1,319 GSM8K for both models.

Validation: CPU numerical tests before launch; real-model hash/mask/sham checks during the run; final artifact and gate consistency checks before completion claims.
