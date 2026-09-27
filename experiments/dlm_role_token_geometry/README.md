# Role damage proxy comparison — fixed Max / Wanda

## Hypotheses

Let y_t be the dense Linear output and z_t(s) the output under the persisted
Wanda candidate mask. Within each masked/unmasked role:

- Original: sum ||z−y||² / sum ||y||² (output-energy-weighted relative error).
- Token-relative: mean_t [||z−y||² / ||y||²]. Tests within-role energy domination.
- Token-angular: mean_t [1−cos(z,y)]. Tests direction vs magnitude after the same
  token balancing. Compare angular with token-relative to isolate the latter change;
  angular vs original changes both geometry and within-role pooling.

All use max of the two pooled role curves and the existing raw-marginal greedy
allocation with exact row-floor65 budget. The Wanda weight ranking is unchanged.
The claim that local angle better predicts functional importance is unproven.
Residual additions/gates can make output magnitude important. This is an ablation,
not an invariance theorem or claimed DLM-specific innovation by itself.

These are NOT the earlier per-token raw error proxy, diagonal error, fitted
raw/relative predictor, Global Minimax, or sparse-context recalibration.
Both collect on the same historical80 dense-background states in one shared pass.
No backward, teacher labels, downstream fitting, new calibration, smoothing,
manual protection, allocation coefficients, or global target change.

## Gates / costs

1. Freeze sources, run tests; collect six candidate Linear outputs per projection/state.
2. Dense batch1 original-error control must reproduce historical Role allocation.
3. Save raw role curves, negative margins, rank comparisons and exact mask differences.
   Only identical selected masks skip/reuse generation; no tuned correlation cutoff.
4. Evaluate up to two distinct candidates on the exact existing mini100, parallel GPU0/1.
   Baseline Role24 is receipt-verified and reused. No automatic full evaluation.

Baseline is already heavily reused development data; a mini win is only a lead.
No claim that role balancing/angle is necessary or generally superior from one screen.

## Numerical definitions

BF16 candidate Linear forward, FP32 token norm/error/dot products, FP64 pooled sums.
Angular cosine uses FP64 denominator and clips numerical cosine to[−1,1].
Zero candidate norm => angular damage1; zero dense norm => stop, no fitted epsilon.
Raw margins remain raw even if angular curves are non-monotone.

Run uses tmux. Progress: `python3 experiments/dlm_role_token_geometry/status.py`.
Collection onGPU0 is shared; only downstream runs use two GPUs. Runtime is measured
from collected states; downstream typically adds roughly35minutes in parallel.
