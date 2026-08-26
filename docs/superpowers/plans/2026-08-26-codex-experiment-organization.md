# Codex Experiment Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every Codex-run experiment into a self-contained `codex/` hierarchy and document its Korean-language methods and results.

**Architecture:** Keep shared model and evaluation implementations at the repository root, while each experiment owns its configs, runners, design records, and results under one directory. Preserve ignored raw artifacts by same-filesystem moves and retain tracked compact results through `git mv`.

**Tech Stack:** Git, Bash, Markdown, JSON, pytest

**Spec:** `docs/superpowers/specs/2026-08-26-codex-experiment-organization-design.md`

## Global Constraints

- Move files; do not copy or delete experimental data.
- Keep `dlm_gradient_sensitivity.py`, `lib/`, `main_llada.py`, `eval_llada.py`, and tests at their current shared locations.
- Keep nested `results/` and `pruned_weights/` artifacts ignored unless Git already tracks them.
- Do not modify the primary worktree's untracked `.agents/` or `skills-lock.json`.
- Do not rewrite historical commands embedded in raw result JSON, JSONL, or logs.
- Create focused commits and do not amend, rebase, reset, squash, or push.

---

### Task 1: Move tracked and ignored experiment trees

**Files:**
- Create: `codex/channel_3848/`
- Create: `codex/time_risk_sensitivity/`
- Create: `codex/mean_dlm_sweep/`
- Create: `codex/controlled_winogrande/`
- Create: `codex/calibration_16x512/`
- Move: experiment configs, runners, design records, and result trees listed in the spec

**Interfaces:**
- Consumes: the exact old-to-new mapping in the spec
- Produces: experiment-local paths consumed by runners, tests, and README links

- [ ] **Step 1: Record the pre-move inventory**

Run in the experiment worktree:

```bash
for d in experiments/channel_3848 experiments/time_risk results/channel_3848 results/time_risk results/mean_dlm results/controlled_winogrande results/calibration_16x512; do
  find "$d" -type f -printf '.' | wc -c
  du -sb "$d"
done
```

Expected: all seven source trees exist; `results/time_risk` contains 41 files and about 18GB.

- [ ] **Step 2: Move each tree atomically**

Run:

```bash
mkdir -p codex/channel_3848 codex/time_risk_sensitivity codex/mean_dlm_sweep codex/controlled_winogrande codex/calibration_16x512
git mv experiments/channel_3848 codex/channel_3848/configs
git mv results/channel_3848 codex/channel_3848/results
git mv docs/superpowers/specs/2026-08-13-llada-channel-3848-causal-experiment-design.md codex/channel_3848/design.md
git mv docs/superpowers/plans/2026-08-14-llada-channel-3848-causal-experiment.md codex/channel_3848/plan.md
git mv experiments/time_risk/pilot.json codex/time_risk_sensitivity/config.json
git mv results/time_risk codex/time_risk_sensitivity/results
git mv docs/superpowers/specs/2026-08-18-time-risk-dlm-fisher-experiment-design.md codex/time_risk_sensitivity/design.md
git mv docs/superpowers/plans/2026-08-18-time-risk-dlm-gradient-sensitivity.md codex/time_risk_sensitivity/plan.md
git mv experiments/time_risk/run_mean_sparsity_sweep.sh codex/mean_dlm_sweep/run_gsm8k.sh
git mv experiments/time_risk/run_mean_winogrande_sweep.sh codex/mean_dlm_sweep/run_winogrande.sh
git mv results/mean_dlm codex/mean_dlm_sweep/results
git mv experiments/time_risk/run_controlled_winogrande_comparison.sh codex/controlled_winogrande/run.sh
git mv results/controlled_winogrande codex/controlled_winogrande/results
git mv experiments/time_risk/calibration_16x512.json codex/calibration_16x512/config.json
git mv experiments/time_risk/run_calibration_16x512.sh codex/calibration_16x512/run.sh
git mv results/calibration_16x512 codex/calibration_16x512/results
```

Expected: `experiments/` has no remaining files; ignored block masks and logs moved with their parent directories.

- [ ] **Step 3: Update active paths and moved historical documents**

Apply these exact replacements only to moved Markdown/config/runner files and `tests/test_dlm_gradient_sensitivity.py`, excluding result JSON, JSONL, and logs:

```text
experiments/channel_3848/                    -> codex/channel_3848/configs/
results/channel_3848/                        -> codex/channel_3848/results/
pruned_weights/channel_3848/                 -> codex/channel_3848/pruned_weights/
experiments/time_risk/pilot.json             -> codex/time_risk_sensitivity/config.json
results/time_risk/                           -> codex/time_risk_sensitivity/results/
results/mean_dlm/                            -> codex/mean_dlm_sweep/results/
results/controlled_winogrande/               -> codex/controlled_winogrande/results/
experiments/time_risk/calibration_16x512.json -> codex/calibration_16x512/config.json
results/calibration_16x512/                  -> codex/calibration_16x512/results/
```

Keep every runner's repository-root bootstrap as `cd "$(dirname "$0")/../.."`; the new runners remain exactly two directories below the root.

- [ ] **Step 4: Check moved paths and runnable files**

Run:

```bash
find codex -name '*.json' -not -path '*/results/*' -print0 | xargs -0 -n1 jq empty
find codex -name '*.sh' -print0 | xargs -0 -n1 bash -n
rg -n 'experiments/(channel_3848|time_risk)|results/(channel_3848|time_risk|mean_dlm|controlled_winogrande|calibration_16x512)|pruned_weights/channel_3848' codex tests --glob '!**/results/**'
```

Expected: JSON and shell checks pass; `rg` has no matches except the organization spec's intentional old-to-new mapping outside the searched paths.

- [ ] **Step 5: Run relevant tests**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_dlm_gradient_sensitivity.py tests/test_eval_llada.py tests/test_main_llada.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the organization change**

Review:

```bash
git status --short
git diff --check
git diff --stat
```

Commit only the moves and reference updates:

```bash
git add -u
git add codex tests/test_dlm_gradient_sensitivity.py
git commit -m "refactor: organize Codex experiments by study"
```

### Task 2: Write the Korean experiment README

**Files:**
- Create: `codex/README.md`

**Interfaces:**
- Consumes: committed result JSON, reports, sample JSONL, and the primary worktree's phase1/phase2 reports
- Produces: the single human entry point for all six experiments

- [ ] **Step 1: Extract source-of-truth metrics**

Use `jq` on every `results_*.json` to extract WinoGrande/GSM8K accuracy and standard error. Use the existing reports for temporal-rho, channel-3848, and Time-Risk gate decisions. Compute calibration deltas from paired sample correctness by matching `doc_id`; do not estimate values from console prose.

- [ ] **Step 2: Create `codex/README.md`**

Write Korean sections for:

```text
개요
공통 환경과 평가 조건
디렉터리 안내
Temporal rho 실험
Channel 3848 인과 실험
Time-Risk sensitivity 실험
Mean-DLM sparsity sweep
동일 조건 WinoGrande 비교
16×512 calibration 비교
종합 결론과 해석 한계
재현 방법
```

Every result table must name the benchmark, sparsity, calibration profile, sample count, accuracy, standard error, and incomplete status where applicable. State that Mean-DLM 16×512 at 50% was intentionally not run after the user-requested stop.

- [ ] **Step 3: Validate README links and numbers**

Run a local Markdown-link target check for relative paths and compare all tabulated WinoGrande values to `jq` output. Confirm that no result is described as statistically significant when its paired 95% confidence interval contains zero.

- [ ] **Step 4: Commit the README**

Run:

```bash
git status --short
git diff --check
git diff -- codex/README.md
git add codex/README.md
git commit -m "docs: summarize Codex experiments in Korean"
```

### Task 3: Verify and expose the organized tree in the primary worktree

**Files:**
- Move locally after fast-forward: `results/phase1/`, `results/phase2/`
- Move locally after fast-forward: `pruned_weights/channel_3848/`

**Interfaces:**
- Consumes: verified experiment branch and ignored local-only artifacts
- Produces: `/home/tmluser1/sap/codex/` as the complete user-visible experiment tree

- [ ] **Step 1: Perform final branch verification**

Read and apply `superpowers:verification-before-completion`. Run the relevant test command again, validate all non-result JSON and runners, audit tracked README links, and confirm the experiment worktree is clean.

- [ ] **Step 2: Snapshot primary-only ignored artifacts**

In `/home/tmluser1/sap`, record file counts and `du -sb` for:

```text
results/phase1
results/phase2
pruned_weights/channel_3848
```

Expected current counts: 6, 6, and 56 files respectively.

- [ ] **Step 3: Fast-forward the primary branch**

Run:

```bash
git status --short
git merge --ff-only exp/time-risk-dlm-gradient-sensitivity
```

Expected: only pre-existing `.agents/` and `skills-lock.json` remain untracked; no merge commit is created.

- [ ] **Step 4: Move primary-only ignored artifacts**

Run:

```bash
mkdir -p codex/temporal_rho/results codex/channel_3848
mv results/phase1 codex/temporal_rho/results/phase1
mv results/phase2 codex/temporal_rho/results/phase2
mv pruned_weights/channel_3848 codex/channel_3848/pruned_weights
```

Re-run file counts and `du -sb` at the destinations. Expected: counts and bytes exactly match the Step 2 snapshot; the source paths no longer exist.

- [ ] **Step 5: Report completion**

Report the final `codex/` path, six experiment directories, verification results, local ignored artifact moves, and every commit hash/message created during this task.
