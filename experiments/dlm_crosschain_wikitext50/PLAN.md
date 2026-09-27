# Cross-chain controls on frozen WikiText validation

Status: prepared, before GPU execution. Objective: determine whether the four already fixed 50%-sparse masks that completed Full GSM8K separate on the project's established DLM likelihood metric.

Hypothesis: Natural-Multi may have lower token NELBO than A-only if preserving same-chain response helps distributional prediction, even though Full GSM8K did not show a task benefit. This is exploratory because the validation corpus and GSM8K outcomes have already been observed in this project.

Planned setup: LLaDA-8B-Base revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2; four existing A, Multi, Cross, CrossMatched physical masks and original surviving weights, exact 3,489,660,928 / 6,979,321,856 zeros; filtered WikiText-2 validation 551 chunks/268,163 tokens; article-local unconditional 512-token chunks with tails; original MC128 exact-k shared CPU mask draws, seed 2025, BF16 model/FP32 masked CE. Report token NELBO, exp(NELBO) estimate, MC SE. No mask or coefficient change, no test access, no score-based stopping. Each arm has per-chunk atomic checkpoints, SHA verification, model SHA verification and resumable tmux GPU worker. Old WikiText baselines may use different calibration and allocation, so direct causal claims from their numerical difference are not planned.

Related experiments: 2026-09-27 Full GSM8K Natural/Cross C control; original 50%-sparse WikiText NELBO baseline sweep; original multiscale A+C calibration and diagnostic banks.

Actual setup / progress / results / interpretation / decision: pending. Obsidian MCP tools were absent from this turn's tool catalog; this local preregistration precedes GPU work.

## Actual start

Status: running from 2026-09-27 13:24:55 KST. GPU0 occupied by another process; GPUs 1,2,3 verified idle (0% and 14 MiB each) immediately before launch. Dedicated tmux session: `dlm_crosschain_wikitext50`. Four arms are queued in fixed order A, Multi, Cross, CrossMatched; no outcome-based selection. Source/config SHA validation and 13 existing protocol CPU tests passed. Output root: `experiments/dlm_crosschain_wikitext50/output/`.
