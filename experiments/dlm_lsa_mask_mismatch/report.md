# LSA clean/corrupted masked likelihood mismatch audit

## Objective / Hypothesis
Compare clean and corrupted LSA proxies against the same corrupted masked-gold CE damage. Hypothesis: corrupted-calibrated LSA ranks damage better. This is a proxy-input comparison, not clean-vs-corrupt functional damage or an AR/DLM comparison.

## Actual setup
Frozen LLaDA-8B-Base revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2. Existing 224 full-H corrupted LSA metrics (80 states, 8 identical 256-token spans ×10 masks) reused with source provenance checks. Clean covariance added from the same 8 spans, H=2/8 sum(XᵀX), original LSA probe50/group128 metric. Fixed historical corrupted-calibrated Standard Wanda rowwise50 masks. Each projection intervened alone in dense background using native batch1 full forwards. Target: signed CE sum/(p_mask*256) change against dense, averaged uniformly over states. Block target is the mean of seven projection-only changes, not joint block damage.

Clean-input CE would reveal gold tokens and was not used as DLM likelihood. No new allocation, generation, full PPL, or tuning.

## Results

| Granularity | Clean proxy vs CE damage rho | Corrupted proxy vs CE damage rho | Difference | Paired conditional 97.5% CI |
|---|---:|---:|---:|---|
| Projection | 0.227495 | 0.236451 | +0.008956 | [-0.002125, +0.013543] |
| Block | 0.439150 | 0.434018 | -0.005132 | [-0.010264, +0.002566] |

Bootstrap: 8 sequence clusters, 2000 resamples, seed0, fixed proxy estimates; 97.5% intervals account for two primary granularities via Bonferroni. Projection difference positive in 4/8 sequences; block positive in 3/8.

Clean/corrupted proxy rho: projection0.992215, block0.999633. Applying the original basic LSA lambda=.1 mapping descriptively gives mean absolute ideal block sparsity difference0.309909pp, max1.398480pp; no actual masks built.

Exploratory projection rank correlations after layer/type rank residualization: clean0.133171, corrupted0.137253.

CE damage is signed: 51.55% of module/state changes and 120/224 module means were negative. These are observed improvements in this finite calibration CE diagnostic, not proof of improved generalization or generation. No absolute-value transformation was used.

## Verification
224×80=17,920 native batch1 intervention forwards; all224 module checks agree with physical weight pruning on one state, and all224 restoration checks reproduce dense CE. End-to-end dense weight SHA matches. All frozen source/input hashes and record coverage checked. Rank correlations independently recomputed using Pearson correlation of ranks. See verification.json for output hashes and descriptive allocation quantities.

## Interpretation / Decision
No established extra damage-ranking information from corrupted rather than clean LSA calibration. Large mask-induced shifts of the original LSA importance ordering are unsupported in this setup. LSA proxies retain some association with damage, with limited projection-level rank correlation; this alone does not diagnose why LSA allocation loses downstream performance.

This is an in-calibration diagnostic with only8 sequences. CIs condition on proxy estimates; LSA surrogate selector differs from the Wanda intervention. The experiment does not establish that mismatch is larger in DLM than AR, nor that masked/unmasked pooling, feature direction, high-sparsity extrapolation or any correction is the causal explanation. Stop this diagnostic; no automatic follow-up.

## Related artifacts
config.json, results.json, verification.json, clean/*.json, damage/*.json, dense_losses.json, run.log.
