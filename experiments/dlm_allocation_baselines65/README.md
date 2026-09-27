# LLM allocation baselines → frozen LLaDA DLM-Wanda @65%

## Objective / hypothesis
Test whether LSA, AlphaPruning, DLP and the public fixed DSA graph transfer to
LLaDA under the existing controlled mini100 protocol. No assumed winner.

## Frozen setup
GSAI-ML/LLaDA-8B-Base revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2.
Exactly the historical ordered 224 Linear modules. Standard Wanda ranking from
the saved overall_uniform activation statistic on the frozen 80 DLM states.
This is an **allocation transfer**, not verbatim reproduction of each paper's
AR calibration / sequential sparse-prefix pipeline. No role weighting.
All 224 Uniform65 masks must match historical hashes before generation.
All candidates remove exactly 4,536,008,704 / 6,979,321,856 weights (nominal65%).
Continuous block rates use the existing OWL floor±1 exact-budget DP. No grid,
no downstream selection of parameters, no clipping to rescue invalid rates.

## Public settings and explicit quirks
- DLP: get_dlp_ratios mean path, alpha=.15; structure-path median is not used.
- DSA: README W:(ABSLOG)-(VAR)-(ATAN,ASIN)-(7), args.Lamda=.08;
  graph input is pooled Wanda scores. No evolutionary search. Upstream maps
  asin-domain NaNs to zero; flat resulting scores fail the candidate rather
  than creating arbitrary sparsities. Such failure is not an accuracy of 0.
- AlphaPruning: alpha_peak, block_wise, epsilon=.3. Preserve the upstream
  log10 lower-bound comparison in xmin_peak. Full FP32 CUDA gesvd spectrum
  (numeric backend port from CPU), CPU public tail estimator; no approximation.
- LSA: basic layer=lsa, group size128, resp=.5, source schedule alpha=.1 at65%.
  Literal BlockWanda.blk_s recurrence and basic block mapping. Dense batch-one
  H accumulated as the public add_batch; verify diag(H) against 2*saved A.
  No compensation or alternate local selector is applied to the final model.

Sources / commits and file hashes are frozen in config.json. Numerical tests
compare LSA, AlphaPruning and DSA to the pinned public implementations.

## Execution / artifacts
python -m experiments.dlm_allocation_baselines65.run freeze before launch.
GPU0 queue DLP→LSA; GPU1 queue DSA→AlphaPruning, each inside its own tmux.
Large masks/source snapshots: /DATA/tmluser1/dlm_allocation_baselines65/.
Per-method statistics, allocation, mask manifest, receipts, predictions and
paired results stay in this directory. Log files under logs/.
Statistics checkpoint between modules/blocks. Failed candidates are preserved;
queue continues to independent candidates. No automatic full1319 evaluation.

## Evaluation / interpretation
Exact historical GSM8K mini100 examples, 5-shot prompts, seeds, generation and
strict EM evaluator. Historical Uniform12 and Role24 are reused with matching
protocol hashes. Report paired discordances and exact McNemar tests, descriptive
mini100 evidence; four comparisons do not establish general failure of AR methods.
No results yet at preregistration.
