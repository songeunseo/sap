# LLaDA allocation baselines50: agreed WikiText DLM likelihood

User requested GPU3: Uniform-Wanda / OWL / DLP / AlphaPruning / LSA layer-wise /
LSA projection-wise / DSA / EvoPress, 50% sparsity, previously agreed PPL metric.

Primary: complete frozen filtered WikiText-2 validation, 551 chunks / 268163 tokens,
512-token article-local unconditional chunks, short tails, token weighting,
MC128 exact-k masking, seed2025, shared CPU mask SHA256, BF16/FP32 masked CE.
Report token NELBO, exp(NELBO) PPL-bound estimate, MC SE. Not causal perplexity.
Test remains final held-out data. Same model revision/224 projections/calibration80.
Verified dense allocation surveys reused; new native batch1 sparse-prefix Wanda.
Wanda-based candidates: exact3489660928/6979321856 pruning, floor±1 count DP.
Block0 control reconstructs historical65 masks to verify unchanged ranking.
Parameters: OWL lambda.08/M5, DLP alpha.15, Alpha epsilon.3;
LSA layer pinned alpha[int(.5*10)]=.04, probe.5/group128;
LSA projection official lsac lambda.07/parameter-count correction.
LSA comparison changes both official coefficient and allocation granularity.

DSA: public operators/custom bounded pop8/gen4/seed0/lambda.08 re-search50.
Existing development40 masked-gold CE fitness retained; not full official search.
Development40 WikiText train is verified disjoint from calibration/heldout40;
validation/test exclude shared32-token strings. Historical fitness unchanged.

EvoPress: official commit upstream/evopress/COMMIT. Unmodified FastOBC class
with isolated official utility bindings. BF16 database/damping.01/block128,
17 levels(-8..8)/equal weights_diff524288. Explicit LLaDA adaptation: database
later blocks see uniform50 sparse prefixes via full native replay, no causal wrapper.
Original elitist level-transfer mutations/staged selection/seed0/400generations/
offspring64/survivors8/2/1. Selection2048/8192/10240 tokens from40x256development,
instead of upstream2048/16384/65536. Same-position masked Dense||Sparse KL;
fresh dense teacher, no AR logit shifting. DLM adaptation, not end-to-end AR parity.
Original FastOBC threshold/dead columns may add zeros; report actual sparsity
separately from nominal50; no post-pruning correction. Database approximately237GB
on /DATA, per-file hashes/block checkpoints. SparseGPT compensation preserved.

Dedicated tmux dlm_ppl50_gpu3. Masks/database /DATA/tmluser1/dlm-ppl50.
Logs here; resumable MC chunk/search checkpoints/frozen-source digests.
Status: python3 -m experiments.dlm_ppl50.status using run.sh PYTHONPATH.
Obsidian notes registered before launch; local status is persisted after each method.
No predicted ordering, GSM8K, test access or outcome-dependent coefficient tuning.
