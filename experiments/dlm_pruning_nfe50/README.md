# Pruning × parallel decoding

Approved scope: Dense and existing A exact50% mask, each at256/64/32steps on the frozen exposed200 GSM8K development questions. Generation/block length256,5shot,temperature0,model revision,seed and strict grader remain fixed. The independent question-level interaction at32steps is primary;64steps is secondary. No new pruning mask or allocator.

The scientific proposal is [proposal.md](/home/tmluser1/sap/research/next_experiment_2026-09-27/proposal.md). `output/config.json` freezes the executable setup. Reused256step answers retain source provenance; new generation and diagnostic costs are separate.

## Roles and authority

The user authorized implementation and execution on2026-09-27 and requested GPT-6 Sol/high for new agents. Master owns preparation, shared state and launching. The experiment executor implements generation/diagnostics; an independent experiment auditor verifies caches, statistics and code. Both agents use the source experiment.md's evidence discipline. Original Depth-AR examples, three-hour deadline, hosts, seeds and publication instructions do not apply. Manuscript work remains deferred.

## Outputs

- `output/gsm8k/{Dense,A}_{256,64,32}`: immutable question checkpoints and cell results.
- Dense32 traces and paired full-distribution diagnostics are produced by `runner.py` and `diagnostic.py`.
- `output/controller.json`: live status; `output/logs/`: worker logs.
- `output/report.json`: final question-paired interaction analysis when all six cells are verified.

## Operation

Run CPU preparation with `bash experiments/dlm_pruning_nfe50/run.sh prepare prepare`. After tests and independent review, freeze code via `... prepare seal` and record the launch audit. Launch `... control --gpus 1,3` inside tmux only after checking both devices and other controllers. Do not edit frozen code/config after launch. Restart only the same verified protocol; checkpoints preserve completed work and attempt receipts preserve failed costs.

Dense32 generation includes trace capture, so its recorded wall time includes instrumentation overhead. These raw times do not establish a matched throughput or sparse-kernel speedup.

No result is implied by this README. Exposed200 findings are exploratory; a wide interval spanning zero is inconclusive. Endpoint KL is not NELBO, and an accuracy interaction alone does not identify dependency as its cause.
