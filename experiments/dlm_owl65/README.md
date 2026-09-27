# OWL allocation + frozen DLM-Wanda @ 65%

One untuned OWL allocation transfer, not Role-aware OWL and not a verbatim run
of upstream's sequential pruning pipeline. Frozen LLaDA revision, 224 targets,
80 historical DLM states, historical dense Wanda ranking, same mini100 protocol.

Official source: https://github.com/luuyin/OWL/tree/dddb7a4bffe27c73e4c8cf692b3a5e36401532c8
`lib/prune_all.py` SHA256 3803aee75bfc15de92b7d79d643c6014fdd07898c9e0fc51d32b6a6c76d59a55.
Use README's unstructured example M=5, lambda=.08; NOT argparse default M=3.

For each block concatenate all seven FP32 Standard Wanda score matrices on CPU.
Outlier percentage = 100 * count(score > 5*block_mean)/block_weight_count.
Copy official density formula exactly: minmax(D)*2*lambda, subtract mean, add
1-target. Sparsity = 1-density. No role split, gradients, reconstruction,
5%-grid restriction, coefficient search, downstream feedback or full auto-run.
The code's mean-centered minmax rule is not hard clipping to target +/- lambda.

Port differences, explicit: LLaDA/BF16; historical DLM calibration instead of
128 C4 sequences; frozen dense local ranking instead of upstream sequential
sparse-prefix recalibration; integer budget correction instead of accepting
row-floor drift. 32 blocks have equal parameter counts. Within each block/input
width, use a common row count. DP selects floor(ideal)-1/floor/floor+1 to minimize
parameter-weighted squared deviation, subject to exactly 4,536,008,704 removals.
Save raw and corrected allocation. Stop if infeasible; no manual module changes.

Validate all 224 generated Uniform65 masks against history before evaluation.
Freeze source hashes and validate prediction identities, strict EM, and protocol.
One primary paired comparison vs Uniform12, secondary vs Role24 (descriptive).
Repeated mini100 is development data, not an independent confirmation set.

Run long work only in tmux: `bash experiments/dlm_owl65/run.sh run`.
Status: `/usr/bin/python3 experiments/dlm_owl65/status.py`.
