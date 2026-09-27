# Multiscale A+C — implementation ready, GPU experiment not started

Date: 2026-09-22. User instruction: prepare code without using any GPU.

## Objective

Make the frozen multiscale A+C small-evaluation plan executable when GPUs become available.

## Hypothesis

Unchanged from `ac_multiscale_experiment_plan_2026-09-22.md`. Implementation verification does not test the research hypothesis.

## Planned Setup

LLaDA-8B-Base, exact 50% pruning, fixed native Wanda rankings, same block-probe/rank-map/exact-count backend. Five same-bank A/Short/Path/All/Multi objectives, shared 64-probe outputs. Development mini100, optionally a separately launched confirmation100; maximum 800 generated answers over 200 distinct questions. No full GSM8K/PPL.

## Actual Setup

Code: `/home/tmluser1/sap/experiments/dlm_multiscale_ac50/`.

Prepared output: `/home/tmluser1/sap/experiments/dlm_multiscale_ac50/output/`.

Config SHA256: `7041424d35ea8b0ff2db70cc896ce7be8d084b30c1606caa8a3d9dc2f0ea5b15`.

The original plan remains immutable. Its earlier `runner_implemented: false` describes the planning turn; this implementation note and the new prepared config record the subsequent implementation state.

## Progress / Notes

- Implemented CPU bank construction, five objectives, exact allocation, span-resampled allocation stability, native-model workers, two-GPU block sharding and candidate scheduling.
- Implemented atomic state/document checkpoints, fingerprint checks, physical model deduplication, cached-generation reuse, paired reports, a confirmation gate, tmux launch, status/ETA and failure cleanup.
- Frozen two 128-state banks, the selected 200 requests and tokenization hashes, legacy mask/results/ranking hashes, code sources and dependency versions.
- Preserved few-shot RNG by constructing prompts in original dataset order on CPU before selecting the 200 stored requests. No answers were generated.
- Verified the six cached model checkpoint shards exist (16,031,197,112 bytes total); did not load the 8B model.
- Actual GPU memory/performance and the real-model smoke check remain untested. The first GPU launch performs the smoke/reference/mask checks before collecting probes.

## Results

Implementation validation only:

- 17 CPU tests passed, zero failures/errors.
- Existing 300 Uniform/A/AC mini predictions match frozen document/prompt/protocol identities and official strict-match regrading.
- Current lm-eval protocol hash equals the historical hash.
- Preparation completed with `torch.cuda._lazy_init` replaced by a function that raises; `torch.cuda.is_initialized()` remained false.
- Default output is `prepared_not_started`. There is no `started.json`, running experiment record, or model readout directory.
- No GPU query, GPU allocation, tmux launch, or automatic scheduler was performed.

Receipt: `experiments/dlm_multiscale_ac50/output/cpu_validation.json`.

## Interpretation

The input, objective, budget, checkpoint and evaluation-identity paths have CPU coverage. These results are not evidence for multiscale pruning quality or a guarantee that the first real GPU run will pass.

## Decision

Implementation and CPU preparation are complete. Wait for an explicit request to run experiments.

## Next Experiment

When execution is requested, use the README commands. The default launch completes only development and stops. Confirmation is a separate explicit command after inspecting the result and meeting the frozen gate.

## Related Notes

- `research/ac_multiscale_monotone_design_2026-09-22.md`
- `research/ac_multiscale_experiment_plan_2026-09-22.md`
- `experiments/dlm_multiscale_ac50/README.md`

## Obsidian synchronization

Pending local record. The current tool catalog does not expose `mcp__obsidian__get_sync_status` or `mcp__obsidian__read_note`; no new MCP call was possible. This is not a new observation that the server disconnected. The preceding design turn had successfully saved/read its Obsidian note. Sync this implementation note and the planned experiment before a future launch when those tools are callable.
