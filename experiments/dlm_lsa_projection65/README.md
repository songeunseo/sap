# Projection-wise LSA at 65% — one GSM8K mini100

User requested lsac, not a new method or hyperparameter sweep. Frozen LLaDA
revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2, ordered 224 targets, 80 DLM
calibration states and historical mini100/5shot/generation/strict EM/seeds.

Reuse the complete provenance-checked dense LSA metric survey. Official commit
1c28cae299cf942b0eb2ef5ca582972506dc0d0c blk_score_global projection path, lambda
0.07 (official lsac README example), hypothetical 50% probe, group128. No block
average, abs, sign inversion, role split, clipping or 5% grid. Allocation uses
parameter counts exactly as source; parity checked by executing original AST
mapping statements. Independently round each projection row count within
floor +/-1, minimizing weighted squared rate deviation with exact global DP.
Exact removed count 4,536,008,704 / 6,979,321,856, matching Uniform65.

Use isolated namespace of the frozen sequential Wanda runner: native batch1
sparse-prefix calibration per block, all seven statistics before block pruning;
unchanged Standard Wanda masks, first-block historical hash gate and physical
mask verification. Cache masks on /DATA/tmluser1/dlm-lsa-projection65.
Mini compares sequential Uniform12 and basic LSA2. These differ in granularity
AND official mapping/lambda (.07 vs .1), not a pure granularity ablation.
Heldout40 CE is a supplementary inherited diagnostic, never selection.
No automatic full GSM8K, PPL or tuning. Existing historical files unchanged.

Run tests and freeze with the project PYTHONPATH, then run.sh in tmux on a free
GPU. Status: python3 experiments/dlm_lsa_projection65/status.py.
Pruning resumes from validated block masks. GSM8K evaluator writes predictions
after all 100 examples; an interruption during generation requires restarting
generation, not pruning. Tqdm progress and ETA are retained in logs/run.log.
