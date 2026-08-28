# DLM-Loss Gradient Aggregation Ablation Design

## Question

Given the official LLaDA DLM-loss gradient for a fixed calibration state, does
signed mean, mean absolute effect, or mean-square effect best rank weights for
50% row-wise pruning?

This experiment does not use, modify, or compare against Sink-Aware pruning.

## Fixed protocol

- Model: `GSAI-ML/LLaDA-8B-Base` at revision
  `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, loaded in BF16.
- Calibration data: WikiText-2 raw train, eight deterministic 256-token spans,
  seed 0.
- Timesteps: `0.05, 0.15, ..., 0.95`; eight samples at every timestep.
- Corruption: `p_mask(t) = (1 - 0.001) * t + 0.001`; state seed is
  `timestep_index * 8 + sample_position`; resample with incremented seed only
  when no token is masked.
- Historical state digest: regenerated `calibration_manifest.json` must have
  canonical SHA-256
  `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`.
  A mismatch stops the experiment before scoring.
- Prunable scope: the seven `nn.Linear` matrices returned by the existing
  Wanda `find_layers` traversal in each of 32 transformer blocks (224 matrices).
- Pruning: exactly `floor(input_width * 0.5)` entries per output row, stable
  ascending score order.
- Evaluation: GSM8K, 5-shot, temperature 0, generation length 256, block length
  256, denoising steps 256, identical evaluator and prompts for all methods.
- Compared methods only: Dense, Wanda 50%, SparseGPT 50%, DLM-SUM 50%,
  DLM-ABS 50%, DLM-SQUARE 50%.

Wanda and SparseGPT use the standard eight clean 256-token calibration spans.
They are reference baselines, not calibration-compute-matched baselines: each
DLM method uses 80 corrupted states from eight spans and ten timesteps. The
primary controlled comparison is among SUM, ABS, and SQUARE, which share the
same states and backward passes. Beating a reference baseline must not be
described as an improvement under matched calibration compute.

## Exact loss and effects

For one batch-one state `s = (sample, timestep)` with sequence length `L`, mask
`m`, clean tokens `x`, and corrupted input `x_t`, use the official LLaDA
pre-training loss:

```text
L_s = (1 / L) * sum_{j: m_j = 1} CE(f_theta(x_t)_j, x_j) / p_mask(t)
```

For every target weight `w_i` and its gradient `g_i,s = dL_s/dw_i`:

```text
d_i,s = -w_i * g_i,s
SUM(i)    = mean_s d_i,s
ABS(i)    = mean_s |d_i,s|
SQUARE(i) = mean_s d_i,s^2
```

All three accumulators are updated from the same backward pass. Experiment 1A
passes state weight `alpha = 1`; the accumulator interface accepts a scalar
state weight so a later experiment can supply `alpha(t)` without changing the
gradient collector. No other weighting or heuristic is present.

## Selected architecture

Use one isolated runner at `experiments/dlm_loss_aggregation/run.py` and one
test module. Reuse only the repository's model loader, WikiText loader,
`find_layers`, Wanda, SparseGPT, LLaDA generation, and lm-eval task formatting.
Do not extend the previous Time-Risk scorer.

The runner processes one transformer block at a time. Only that block's seven
linear weights require gradients. Post-accumulate hooks immediately copy each
gradient to CPU FP32, form `d = -weight * gradient` with a frozen CPU FP32
weight snapshot, update SUM/ABS/SQUARE, clear the parameter gradient, and retain
no per-state gradient. `zero_grad(set_to_none=True)` brackets every state.
Cached dense hidden states advance one block after all 80 gradients for the
current block have been consumed.

After a block is complete, each module is finalized independently. The runner
computes its three row-wise masks, matrix-level pairwise Spearman values, exact
mask XOR fractions, sign consistency, spike ratio, finite/shape/count checks,
and compact distribution summaries. FP32 scores are then discarded.

Before applying exact Spearman to all matrices, the smoke run profiles ranking
the largest prunable matrix and records elapsed time and peak CPU RSS. If that
profile is not comfortable for the current machine, Spearman alone uses a
deterministic seed-0 sample of at most 1,000,000 shared weight indices per
matrix. The CSV records whether each row is exact or sampled and the sample
size. Mask construction and mask XOR remain exact over every weight.

The three masks are bit-packed immediately and retained in CPU RAM. Across
6,979,321,856 prunable weights this costs about 2.44 GiB for all methods,
compared with about 19.5 GiB as CPU boolean tensors. Permanent artifacts keep
only mask SHA-256 values and aggregate metadata; regenerating the pinned
protocol deterministically reconstructs the masks.

## Evaluation flow

Evaluation is sequential in one orchestration process:

1. Collect and retain the three packed DLM masks.
2. Evaluate Dense from the pinned checkpoint and release the model.
3. For SUM, ABS, and SQUARE in order, reload Dense, unpack/apply one module mask
   at a time, evaluate, verify the mask hash/result, release the model, and free
   that method's packed mask.
4. Reload Dense, run existing Wanda 50% on the fixed eight clean calibration
   spans, evaluate, record the derived zero-mask hash, and release the model.
5. Repeat for existing SparseGPT 50%.

`eval_llada.py` gains a small optional in-memory model/tokenizer seam used by
the experiment runner. CLI checkpoint loading remains unchanged. The runner
calls lm-eval directly with the same `LLaDAEvalHarness`; it does not duplicate
the evaluator or prompt code and creates no temporary checkpoint.

Before a full GSM8K evaluation, every method runs the same small timing subset.
The result records examples, elapsed seconds, examples/second, and projected
full duration. A failed timing/evaluator check stops full evaluation.

## Artifacts

Permanent files live below `experiments/dlm_loss_aggregation/`:

```text
config.yaml
calibration_manifest.json
scores/metadata.json
masks/hashes.json
diagnostics/score_spearman.csv
diagnostics/mask_xor.csv
diagnostics/sign_consistency.csv
diagnostics/spike_ratio.csv
logs/
results/gsm8k.csv
report.md
```

`scores/metadata.json` contains score equations, update counts, module shapes,
finite checks, summary quantiles, and hashes of canonical metadata. It contains
no score tensor. `masks/hashes.json` contains method/module hashes, row counts,
and overall hashes, but no mask payload.

Writes are atomic. A completed artifact is read back and validated before the
runner releases the state needed to produce it. Full checkpoints and full score
tensors are never written.

## Diagnostics

Each CSV contains one row per matrix plus module-type and weighted-overall rows.
Spearman and XOR files contain all three pairs: SUM/ABS, SUM/SQUARE, and
ABS/SQUARE. Weighted overall means use matrix element counts. Module-type means
are also element-weighted.

For every weight:

```text
C_i = |SUM(i)| / (ABS(i) + eps)
T_i = sqrt(SQUARE(i)) / (ABS(i) + eps)
```

The diagnostic CSVs store compact distribution summaries for `C` and `T`, not
the tensors. Neither diagnostic affects pruning.

## Gates and failure handling

The runner stops without a full run when any of these occurs:

- historical calibration digest mismatch;
- model revision, mask token, block count, module count, or shape mismatch;
- non-finite loss, gradient, effect, or score;
- missing gradients or unequal update counts;
- score shape differs from its weight;
- ABS/SQUARE contains a negative value;
- dense weights change before explicit mask application;
- any pruned row has the wrong prune count;
- one-state peak allocated or reserved CUDA memory exceeds 30 GiB;
- an artifact/result fails readback validation.

The one-module tiny-calibration smoke test precedes full score collection. The
one-state benchmark records allocated/reserved CUDA peaks and projects full
scoring runtime before 32-block execution. Real-model score correlations,
negative-SUM fraction, and exact mask XOR are observations only. Identical or
nearly identical scores or masks are valid scientific results, including the
mathematically expected single-state ABS/SQUARE ranking identity.
Undefined correlations from constant score vectors are recorded with their
reason rather than treated as a scoring failure.

## Resource budget

- Permanent disk: less than 100 MiB.
- CPU masks: about 2.44 GiB bit-packed.
- Estimated peak CPU working memory: 10--14 GiB, below the observed 25 GiB
  available RAM. Unpacked full masks are forbidden.
- Estimated peak GPU memory: below 18 GiB from the prior 16.2 GiB endpoint plus
  margin; the measured 30 GiB smoke gate is authoritative.

## Testing and milestones

Tests cover the exact official loss, deterministic manifest/digest validation,
a hand-constructed mixture of positive and negative effects proving the exact
SUM/ABS/SQUARE equations, future scalar state weighting, stable row-wise signed
pruning, valid identical ABS/SQUARE single-state masks, compact mask
round-trip/hash, diagnostics, resource-accounting metadata, and in-memory
evaluator injection. Implementation uses red-green cycles and focused commits
for scoring/masks, diagnostics, evaluation orchestration, and experiment
configuration/reporting.
