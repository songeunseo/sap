# Prior-method 50% GSM8K mini-100

## Task contract

- target_role: executor
- task_type: research-code / empirical-analysis
- operation: execute
- rigor: standard
- authority: create-new
- goal: Evaluate every completed 50% prior-method mask from the frozen WikiText comparison on the same GSM8K development IDs 0--99.
- source_of_truth: the eleven existing mask manifests and their completed WikiText validation receipts; the frozen LLaDA-8B revision; the existing GSM8K request/protocol artifact.
- analysis_unit: one GSM8K question; paired correctness by identical example ID.
- estimand: strict exact-match correct count out of 100; fixed paired differences versus native row-quota Uniform.
- exclusions: no new pruning, allocation search, mask repair, hyperparameter tuning, full GSM8K, or new question selection.
- quality_gates: 224-module identity, source/payload hashes, physical sparsity counts, dense and sparse model hashes, IDs/prompts/targets/protocol identity, 256 forwards per generation, atomic per-document checkpoints, exact paired McNemar with Holm over ten fixed contrasts.
- stop_conditions: source manifest/payload mismatch, changed protocol/sample, non-idle assigned GPU, or any requested scientific setting change.

## Fixed arms

DSA layer, EvoPress+FastOBC, DSA projection, Uniform row-quota, OWL projection,
OWL layer, LSA layer, LSA projection, Uniform layer-global, AlphaPruning, DLP.

All Wanda-engine arms are exactly 3,489,660,928 / 6,979,321,856 pruned weights.
EvoPress preserves its observed FastOBC artifact sparsity, 3,489,662,724 zeros
(50.0000257332%), rather than silently altering the published artifact.

## Evaluation

LLaDA-8B-Base revision `0f2787f...`, BF16, GSM8K IDs 0--99, 5-shot,
temperature 0, 256 denoising steps, generation/block length 256, strict-match.
This is repeatedly used development evidence, not independent confirmation.
