# Reveal-KL allocation mini100

Hypothesis: protecting dense next-reveal token distributions yields better65%
block allocations than averaging KL over all masked tokens in the same states.
Eight frozen WikiText train spans supply128-token prompts. Dense temp0 decoding
produces256tokens over256steps, capture8,24,...248:128states, K1. First rollout
must match native generate exactly. Probe32entireblocks individually at65% with
verified historical dense-calibrated Wanda masks. Both scores come from exactly
the same full-forward logits. Equal state means within prompt, equal8prompt means.

Higher score -> less sparsity: average-rank mapping fixed60–70%nominal range,
exact same historical4,536,008,704removed weights via existing row-count DP.
Final two candidates use the original80calibrationstates and native sparse-prefix
Wanda. Probe-to-global-context/ranking mismatch is an explicit approximation.
Uniform12/100 uses verified sequential baseline. Two new mini100 runs preserve
all historical five-shot/temp0/256step/seed/strictEM settings. Primary paired
reveal vs all-masked; secondary two vs Uniform use Holm2. This reused mini is a
development screen. Full/PPL/75%/tuning are not part of this experiment.

Run pipeline only inside dedicated tmux. GPU3, one collect then two evaluations.
Metadata/artifacts are local to this directory; history is never overwritten.
Progress: progress.json, each candidate/progress.json, pipeline.log, exit_code.txt.
