# Time-Risk DLM Gradient-Sensitivity Go/No-Go Experiment Design

## Objective

Test whether weight importance in a masked diffusion language model varies enough
across denoising timesteps that preserving high-variance weights improves
post-training pruning over both Wanda and mean DLM-loss gradient sensitivity.

The primary comparison is not against channel 3848 protection. Channel 3848 is
deferred after the fixed-sparsity protection experiment showed no recovery. This
experiment asks whether a DLM-specific pruning signal exists at all.

## Hypothesis and Scores

For clean sequence `x`, timestep `t`, sampled masked state `x_t`, and a prunable
weight `w_i`, define the official LLaDA pre-training loss

```text
L_DLM(x_t, t) = (1 / sequence_length)
                * sum_masked CE(model(x_t), x) / p_mask(t)
```

where `p_mask(t) = (1 - 1e-3) * t + 1e-3`. This matches the official LLaDA
guideline rather than the repository's evaluation-only mean cross-entropy:
`https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md`.

For each of ten fixed timesteps and eight independently masked calibration
sequences,

```text
F_i(t) = w_i^2 * mean_x[(d L_DLM(x_t, t) / d w_i)^2]
mu_i   = mean_t[F_i(t)]
sigma_i = sqrt(mean_t[(F_i(t) - mu_i)^2])
```

The compared importance scores are:

```text
Wanda:         abs(w_i) * sqrt(sum calibration activation_i^2)
Mean DLM-Grad: mu_i
Time-Risk:     mu_i + lambda * sigma_i, lambda in {0.25, 0.5, 1.0}
```

These are empirical sensitivities computed from gradients of the official DLM
loss. The pilot does not claim a formal Fisher- or Hessian-derived objective.
In particular, the official `1 / p_mask(t)` loss factor is also present in the
gradient and therefore enters the squared-gradient score quadratically. Any
formal derivation of that weighting is deferred until the score improves
pruning.

`Std_t` uses population standard deviation (`unbiased=False`). No per-module
z-score or rescaling is added: `mu` and `sigma` already have the same units, and
extra normalization would test another hypothesis. All methods use the existing
row-wise unstructured mask rule so the score is the only intended difference.

## Fixed Pilot Calibration

- Model: `GSAI-ML/LLaDA-8B-Base`, revision
  `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- Calibration data: WikiText2 train, seed 0
- Calibration sequences: 8
- Sequence length: 256 tokens
- Timesteps: midpoint grid `{0.05, 0.15, ..., 0.95}`
- Mask realizations: one deterministic mask per `(sample, timestep)`
- Reliability split: sequences 0–3 are split A and 4–7 are split B; derive
  `sigma_A` and `sigma_B` from the same backward passes
- Gradient microbatch: one sequence, because squaring a batch-averaged gradient
  is not the requested empirical sensitivity
- Scope: the same 224 transformer-block `nn.Linear` weights used by Wanda;
  embeddings and vocabulary head remain excluded
- Numeric accumulation: cast each gradient to FP32 before squaring; maintain
  timestep sum, running mean, and running M2 in FP32 on CPU
- Masking edge case: deterministically resample if a state contains no masked
  tokens

The pilot deliberately uses a small calibration budget. A positive result must
later be replicated with 32 sequences of length 512 before a paper-level claim.

## Feasible Scoring Architecture

LLaDA-8B has 6,979,321,856 prunable weights. Two full FP32 statistics would
require about 56 GB, exceeding the 32 GB GPU and leaving too little of the 35 GB
free filesystem. The scoring pass therefore processes one transformer block at
a time.

For block `b`:

1. Cache the dense hidden inputs for the 80 fixed `(sample, timestep)` states.
2. Freeze every parameter except the seven linear weights in block `b`.
3. For each state, run block `b`, the frozen suffix, final norm, and vocabulary
   head; compute the exact LLaDA loss and backpropagate to block `b`.
4. Accumulate per-weight squared gradients separately for split A and split B.
5. Only after all eight sequences at one timestep are complete, form `F(t)` and
   update full, split-A, and split-B Welford statistics exactly once.
6. Derive all lambda/sparsity masks while the block statistics are resident.
7. Store masks packed to one bit per weight, discard block statistics, and
   advance the cached dense hidden states through block `b` for the next block.

The accumulator must implement this invariant:

```text
F(t) = (1 / 8) * sum_x w^2 * grad(x, t)^2
F_A(t) = (1 / 4) * sum_{x in A} w^2 * grad(x, t)^2
F_B(t) = (1 / 4) * sum_{x in B} w^2 * grad(x, t)^2
mu, M2 updates = exactly 10, once per completed F(t)
```

Updating Welford 80 times with individual states is invalid because it measures
joint sample-and-timestep dispersion. Full and split-half reliability statistics
raise worst-case block-local CPU storage to about 7 GB, still bounded well below
whole-model storage. Prefix caching avoids recomputing blocks before `b`.
Activation checkpointing covers the differentiable suffix if the one-state
feasibility check shows it is needed.

Alternatives rejected for the pilot:

- Whole-model gradient/statistic accumulation: does not fit current memory and
  disk constraints.
- A block-local reconstruction loss: cheaper, but no longer measures the
  requested DLM objective.
- Token-, channel-, CVaR-, or learned-timestep weighting: deferred until the
  mean-plus-standard-deviation proxy shows a signal.

## Mask Artifacts and Exact Sparsity

Scores are computed once. For each `lambda in {0, 0.25, 0.5, 1.0}` and
`sparsity in {0.50, 0.60, 0.70, 0.75}`, the lowest-scoring fraction in every
output row is pruned, exactly matching Wanda's unstructured mask geometry.

The sixteen DLM-Grad masks are bit-packed. One complete mask is about 0.81 GiB;
all sixteen require about 13 GiB. Each artifact records model revision,
calibration indices, seed, timestep grid, mask seeds, score definition, lambda,
module shapes, per-module prune counts, and SHA-256 checksums. Checkpoints are
materialized and evaluated one at a time, then removed; only the dense source,
packed masks, metrics, and logs are retained.

Every materialized model must satisfy:

```text
actual zero count == decision-mask prune count
per-row prune count == floor(input_width * requested_sparsity)
all variants at one sparsity have identical per-module prune counts
```

## Staged Experiment

### Stage 0: One-State Feasibility Gate

Run one length-256 state against block 31 and block 0, including gradient FP32
casting and the real statistic update. Record peak GPU memory, CPU memory,
forward/backward time, gradient finiteness, and nonzero-gradient fraction.

Fit the two-point suffix-cost estimate

```text
seconds(block b, state) = fixed_cost + suffix_cost * (31 - b)
projected_scoring_seconds = 80 * sum_b seconds(block b, state)
```

Continue only if both blocks fit within 30 GiB GPU memory, finish without new
swap activity, produce finite nonzero scores, and project no more than 24 GPU
hours for the complete scoring pass. A larger projection requires explicit
approval before Stage 1.

### Stage 1: Score and Mask Diagnostics

Compute all block statistics and packed masks. Before materializing models,
report for every module and sparsity:

- `sigma / (mu + eps)` median and 90th/99th percentiles
- deterministic-sample Spearman `rho(mu, sigma)` using up to 262,144 weights per
  module
- exact mask disagreement between Mean DLM-Grad and each lambda
- fraction of rows whose mask changes
- split-half Spearman `rho(sigma_A, sigma_B)` on the same deterministic sample
- exact split-A/split-B Time-Risk mask overlap for every lambda and sparsity
- exact overlap of each split's top 1% `sigma` weights
- score and mask determinism under an immediate repeated toy/block check

Reliability passes when the median module-level `rho(sigma_A, sigma_B)` is at
least 0.5 and the mean split-mask Jaccard at lambda 1.0 is at least 0.95 at each
of 50%, 60%, and 70%. If it fails, stop before model materialization and report
the eight-sequence calibration budget as statistically inadequate; increasing
the calibration size requires a revised cost projection, not a wider lambda
search.

If all three lambdas produce masks byte-identical to Mean DLM-Grad at 50%, 60%,
and 70%, stop: the proposed term cannot test the hypothesis at this budget.
Small but nonzero disagreement is reported and proceeds to the loss screen.

### Stage 2: Held-Out DLM-Loss Screen

Use eight WikiText2 validation-selection sequences of length 256, the same ten
timestep grid, and fixed validation masks. Evaluate Wanda, Mean DLM-Grad, and
all three Time-Risk lambdas at 50%, 60%, 70%, and 75% sparsity. Report aggregate
and per-timestep official DLM loss.

Choose one global lambda using only 50%, 60%, and 70%: minimize the mean relative
loss versus Mean DLM-Grad across those three sparsities; ties select the smaller
lambda. The 75% result is excluded from lambda selection because it may be at a
performance floor.

Then evaluate only Mean DLM-Grad and the selected lambda on a separate eight
validation-confirmation sequences. Reuse the exact same masked states for both
models. Record paired per-sequence loss differences and a seed-0, 10,000-resample
cluster bootstrap 95% confidence interval, resampling sequences while keeping
their ten timesteps together.

Proceed to downstream evaluation only if the selected lambda satisfies the
following point-estimate conditions on both the selection and confirmation
sets:

- beats Mean DLM-Grad at least two of 50%, 60%, and 70%;
- improves their mean held-out DLM loss by at least 1%; and
- is not worse than Mean DLM-Grad by more than 1% at any of those sparsities.

The bootstrap interval is reported but is not an additional pass/fail threshold
for this eight-sequence pilot.

Otherwise classify the timestep-risk hypothesis as no-go and do not spend GPU
time on GSM8K generation.

### Stage 3: Paired GSM8K-64 Gate

Evaluate exactly three methods at all four sparsities on the same fixed 64
GSM8K examples and unchanged decoding settings:

```text
Wanda
Mean DLM-Grad
Time-Risk DLM-Grad with the single Stage-2 lambda
```

The primary comparison is Time-Risk versus Mean DLM-Grad. Wanda determines
whether the gradient-sensitivity family is useful at all. Record flexible and
strict exact match, correct-example IDs, paired wins/losses, actual sparsity,
and runtime.

Declare a downstream go only if Time-Risk:

- gains at least two correct answers over Mean DLM-Grad at two or more of 50%,
  60%, and 70%; and
- has a positive total correct-answer difference across those three sparsities.

Do not use 75% alone for a go decision. A tie or one-answer change is
inconclusive, not a positive result.

### Stage 4: Confirmation

If Stage 3 is go, evaluate Mean DLM-Grad and the selected Time-Risk variant on
the same fixed 256 GSM8K examples at the non-floor sparsity with the largest
64-example gain. Confirm with an exact McNemar test over example-level
correctness, then repeat DLM-gradient calibration with 32 sequences of length
512 and re-evaluate that selected pair. Only after both checks should CVaR,
learned `q(t)`, or token-wise heterogeneity be designed.

If Stage 2 passes but Stage 3 fails, report that timestep dispersion improves
the matched DLM loss but not the downstream task, and move to token-level
heterogeneity. If Stage 2 fails, abandon `mu + lambda * sigma` without expanding
the lambda grid.

## Comparison Table

The primary report contains one table per evaluation stage:

| Sparsity | Wanda | Mean DLM-Grad | Risk 0.25 | Risk 0.5 | Risk 1.0 |
|---:|---:|---:|---:|---:|---:|
| 50% | | | | | |
| 60% | | | | | |
| 70% | | | | | |
| 75% | | | | | |

Stage 2 fills all columns with held-out DLM loss. Stage 3 contains only Wanda,
Mean DLM-Grad, and the selected Time-Risk lambda and reports GSM8K correct counts
as well as percentages.

## Reproducibility and Failure Handling

- Store clean calibration token IDs and packed noise masks, not only RNG seeds.
- A resumed run starts at the first absent verified block artifact.
- Write each block artifact atomically and verify its checksum before deleting
  block statistics.
- Fail on NaN/Inf gradients, shape mismatches, missing modules, unexpected
  parameter mutation, non-exact sparsity, or incomplete mask artifacts.
- Persist peak memory and elapsed time per block so the full cost is measurable.
- Commit implementation and configuration before any expensive scoring or
  evaluation run; bugs found by an experiment receive a separate fix commit
  before rerunning.

## Explicitly Deferred

- Token-wise `S_i(t, k)`
- Channel-aware protection or regularization, including channel 3848
- CVaR, maximum-risk, learned timestep distribution, early/late hand weighting
- Per-sparsity lambda selection
- Benchmarks beyond held-out DLM loss and GSM8K

These are added only after the fixed `mu + lambda * sigma` pilot passes its
predeclared gates.
