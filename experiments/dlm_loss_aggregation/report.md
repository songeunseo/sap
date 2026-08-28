# DLM-Loss Gradient Aggregation Ablation

## Question

Why should DLM gradient sensitivity be squared rather than signed-summed or
absolute-summed? Experiment 1A tests whether signed direction, effect magnitude,
or rare large effects provide the most useful pruning ranking. No full scoring
or GSM8K result has been run yet; this report records the required pre-full-run
validation checkpoint.

## Fixed Conditions

- Model: `GSAI-ML/LLaDA-8B-Base`, revision
  `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, BF16.
- DLM calibration: WikiText-2 raw train, seed 0, eight 256-token spans and
  timesteps `0.05, 0.15, ..., 0.95`, yielding 80 deterministic shared states.
- Corruption: `p_mask(t) = (1 - 0.001)t + 0.001`; state seeds are 0 through 79.
  An empty mask is regenerated with the next seed.
- Manifest digest: exactly
  `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`.
- Pruning scope: the same seven Linear matrices per block returned by the Wanda
  traversal, 32 blocks, 50% unstructured pruning within each output row.
- Future GSM8K protocol: 5-shot, deterministic decoding, generation length 256,
  block length 256, and 256 denoising steps for every method.
- Wanda and SparseGPT will use the standard eight clean 256-token spans. They
  are reference baselines, not calibration-compute-matched baselines. The
  controlled comparison is SUM versus ABS versus SQUARE, whose 80 states and
  backward passes are identical.

The exact loss implementation is:

```text
p_mask(t) = (1 - 0.001)t + 0.001
L_s = sum_{j: mask_j} CE(float32(logits_j), clean_token_j)
      / p_mask(t) / sequence_length
```

This is implemented by `official_dlm_loss`: PyTorch cross-entropy with
`reduction="sum"` over masked positions only, divided by `p_mask` and by the
number of sequence tokens. At least one token must be masked.

## Methods

For state `s`, `g_i,s = dL_s/dw_i` and the first-order pruning effect is
`d_i,s = -w_i g_i,s`. One backward pass updates all three uniform accumulators:

```text
SUM(i)    = mean_s d_i,s
ABS(i)    = mean_s |d_i,s|
SQUARE(i) = mean_s d_i,s^2
```

Signed SUM is not absolutized. Each method prunes the smallest half of every
row. Gradients are immediately copied into a block/module CPU FP32 accumulator
and cleared; no per-state gradients or full-model FP32 scores are retained.
Masks are designed to remain bit-packed in CPU RAM and to be consumed in the
order Dense, SUM, ABS, SQUARE, Wanda, SparseGPT. Evaluation reloads the pinned
dense model for every method and accepts the pruned model directly in memory.

## Diagnostics

### Setup

The real-model smoke used block 31 `ff_out`, shape `[4096, 12288]`, and the
first two manifest states. The exact-Spearman profile used the same largest
matrix and the first state. Correlations and identity are observations, not
failure gates.

### Results

- Smoke DLM losses: `1.5755308867` and `0.4088213146`; both finite.
- Update count: 2 for every weight. Dense weights remained bitwise unchanged.
- All score shapes matched the weight, ABS/SQUARE were nonnegative, and every
  row pruned exactly 6,144 of 12,288 entries.
- Negative SUM fraction: `0.4997775`.
- Smoke Spearman (deterministic 1,000,000-weight diagnostic sample): SUM/ABS
  `0.0000706`, SUM/SQUARE `0.0001011`, ABS/SQUARE `0.9971167`.
- Exact full-matrix mask XOR: SUM/ABS `0.5000909`, SUM/SQUARE `0.5001321`,
  ABS/SQUARE `0.0262021`.
- Mean sign consistency: `0.6236244`; median `0.7052408`.
- Mean spike ratio: `0.9467193`; median `0.9855910`.
- Exact tie-aware Spearman on all 50,331,648 weights completed in `10.2852 s`
  with `2,754,024 KiB` (`2.63 GiB`) incremental peak RSS. The one-state
  ABS/SQUARE rank correlation was exactly 1, as mathematically expected.

Exact Spearman is therefore selected for every matrix. The deterministic
one-million-weight fallback remains implemented but is not selected. Mask XOR
is exact regardless of this decision.

## Resource Profile

- Smoke CUDA peak: `16,476,021,760` bytes allocated (`15.34 GiB`) and
  `16,548,626,432` bytes reserved (`15.41 GiB`).
- Profile process peak RSS: `5,497,764 KiB` (`5.24 GiB`).
- One profiled backward state: `0.3260 s`; its simple 32-block x 80-state
  projection is `834.46 s` (`13.91 min`). This projection excludes model load,
  prefix handling, diagnostics, mask packing, pruning, and evaluation, so it is
  a lower-bound planning estimate rather than a measured full-run duration.
- Current host resources after profiling: 91 GiB total RAM, 24 GiB available;
  2.0 GiB free on `/`; 4.6 GiB free on `/dev/shm`.
- Planned retained DLM masks: about 2.44 GiB bit-packed in CPU RAM. Estimated
  full collector peak remains 10--14 GiB CPU RAM and below 18 GiB GPU VRAM.
- Permanent artifact target remains below 100 MiB; the current manifest and
  compact preflight outputs occupy about 1 MiB. No score tensor, mask payload,
  pruned checkpoint, or baseline checkpoint has been written.

## GSM8K

Pending. Dense, DLM-SUM, DLM-ABS, DLM-SQUARE, Wanda, and SparseGPT have not
been evaluated. The full 32-block scoring run must not start until this
preflight checkpoint is reviewed.

## Interpretation

### Hypothesis

Signed aggregation may lose magnitude information through cancellation, while
squaring may either usefully emphasize rare large effects or overweight spikes.

### Observed preflight result

The implementation and resource gates pass. The two-state smoke shows that SUM
and the magnitude-based rankings can differ substantially, while ABS and
SQUARE remain close but are not identical. The one-state ABS/SQUARE identity is
expected and confirms no basis for treating identical rankings as an error.

### Interpretation and decision

These diagnostics do not answer which aggregation improves GSM8K. They only
show that the variants are implemented distinctly where multiple states permit
them to differ and that the run fits the machine. Proceeding to full scoring is
technically supported, but remains paused for explicit review. Later GSM8K
interpretation will use the preregistered rules: ABS over SUM suggests harmful
sign cancellation; SUM over ABS suggests useful directional consistency;
SQUARE over ABS suggests useful emphasis on large rare events; ABS over SQUARE
suggests excessive spike emphasis; similar results suggest aggregation is not
the main limitation.
