# Sequential allocation baselines on LLaDA @65%

Correct previous allocation-only experiments without modifying historical files.
Same model/revision, ordered224 projections, Standard Wanda score, calibration80,
exact4,536,008,704/6,979,321,856 pruned weights and historical GSM8K mini100.

Each block gathers all seven input-energy statistics by native batch-one LLaDA
prefix forwards, then applies row-wise Wanda masks. Earlier blocks are already
sparse; current/later blocks dense. No causal AR masks or shifted loss. Output
hook stops after current block. Original dense allocation surveys/settings are
reused with hashes; these do not depend on the sparse prefix. This preserves
block-wise Wanda integration, not true-sequential submodule pruning. LSA basic
layer variant only. Uniform is rerun with the same sequential pipeline.

DSA re-searches public graph operators, custom seed0/pop8/gen4 controller.
Every candidate uses sequential Wanda before fitness evaluation. Masked gold CE
fitness uses persisted40 spans verified disjoint from calibration80/heldout40.
These previously evaluated oracle-control states are now development data, NOT
fresh confirmation. Heldout40 is measured only after winner freeze. No GSM8K
search. Unsafe vector matrix operators excluded; lambda=.08 unchanged. Ignored
graph suffix canonicalized; population/RNG saved before evaluation for resume.

Commands: python -m experiments.dlm_allocation_sequential65.run freeze
and python -m experiments.dlm_allocation_sequential65.run run --method uniform
(also owl/dlp/alpha/lsa/dsa). Expensive jobs must run in dedicated tmux.
Status: python3 experiments/dlm_allocation_sequential65/status.py
Masks: /DATA/tmluser1/dlm-allocation-sequential65; metadata/logs here.
No automatic full GSM8K or hyperparameter tuning.
