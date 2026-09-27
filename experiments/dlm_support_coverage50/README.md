# Shared state support coverage50: mini experiment

Hypothesis: an allocation informed by state-specific channel coverage improves over
the same allocation based on pooled channel energy alone.

For projection g and frozen state s, e[s,j]=sum_i W[i,j]^2 * sum_token X[s,token,j]^2.
This is diagonal energy, not exact functional output energy or reconstruction loss.
Sort channels once by descending mean_s e[s,j]. Pooled arm takes the shortest prefix
preserving 90% of pooled energy. Coverage arm takes the shortest SAME prefix preserving
90% in EACH of 80 states. Scores are required prefix size / input width.
This is not a globally minimum common support, and the resulting mask remains
unstructured row-wise Wanda; channel support is used only as an allocation proxy.

Both arms: parameter-weighted centered rank map, clipped45–55%, exact row-count DP,
50% overall, 224 projections. All224 original Uniform50 masks must reproduce before
new masks are accepted. Frozen Uniform50 sparse-prefix Wanda ranking reused for both;
no candidate recalibration. Only allocation differs.

Reuse hash-verified dense input energies for same8 WikiText train spans x10 masks,
256tokens; independent corrupted states, NOT a generated temporal trajectory.
No role/timestep reweighting. Threshold .90 and mapping are frozen before results.
Dense repeat/first-state input-energy control; source/model/mask/protocol hashes.

Uniform50 mini54/100 is reused only after exact manifest/model/protocol verification.
Two new GSM8K mini100 runs: existing first100/5shot/256steps/gen256/block256/temp0,
strictEM, random0 and numpy/torch/fewshot1234. Primary paired coverage vs pooled;
secondary both vs uniform with Holm2. No tuning based on these reused development items.

NELBO diagnostic: same16 previously selected validation article chunks512 × MC128
from scale-shape50, all3 candidates, same masks/draws. Uniform per-draw reproduction
checked. Report paired article differences/CI; this is a mini development subset,
NOT full validation/test. No automatic full evaluation or parameter sweep.

Limitations: channel-to-weight transfer, diagonal surrogate, worst-state sensitivity,
fixed pooled prefix ordering, only8 calibration documents; no proof of DLM specificity.

Run in tmux support_coverage50_gpu1. Progress:

    python3 experiments/dlm_support_coverage50/status.py
    tail -f experiments/dlm_support_coverage50/pipeline.log
