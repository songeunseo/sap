# DLM-Loss Gradient Aggregation Ablation

## Question

Why should DLM gradient sensitivity be squared rather than signed-summed or
absolute-summed? Experiment 1A tests whether signed direction, effect magnitude,
or rare large effects provide the most useful pruning ranking. Full 32-block
scoring and diagnostics are complete. No pruning baseline or GSM8K evaluation
has been run yet.

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
diagnostic checkpoint is reviewed.

## Full Scoring Results

### Setup and correctness

Run `20260828T175010-2484545` processed all 32 blocks, seven prunable matrices
per block, and the same 80 manifest states. All 224 matrices received exactly
80 updates (17,920 matrix-state updates total). Every loss, gradient, effect,
and score was finite; score shapes matched the dense weights; ABS and SQUARE
were nonnegative; dense weights remained unchanged; and each output row pruned
exactly half its entries. `evaluation_started` is false.

The 672 packed masks were read back and independently rehashed from their raw
bytes. All hashes matched. They occupy 2,617,245,696 bytes (2.44 GiB) in the
RAM-backed transient directory
`/dev/shm/dlm_loss_aggregation_20260828T175010-2484545`.

### Runtime and memory

- Total scoring and diagnostic runtime: 4,227.28 seconds (70 minutes 27 seconds).
- Peak process RSS: 8.65 GiB.
- Conservative accounted CPU peak, process RSS plus retained tmpfs masks:
  11.09 GiB.
- Peak CUDA allocated: 16.44 GiB; peak CUDA reserved: 18.32 GiB.
- Exact Spearman was used for every matrix. No fallback sampling affected ranks.

### Pairwise diagnostics

Values below are element-count-weighted means across matrices.

| Scope | SUM/ABS rho | SUM/SQUARE rho | ABS/SQUARE rho | SUM/ABS XOR | SUM/SQUARE XOR | ABS/SQUARE XOR |
|---|---:|---:|---:|---:|---:|---:|
| Overall | -0.0000476 | -0.0000454 | 0.992708 | 0.500045 | 0.500046 | 0.028463 |
| attn_out | -0.000198 | -0.000194 | 0.995857 | 0.500131 | 0.500135 | 0.024397 |
| ff_out | -0.0000590 | -0.0000862 | 0.991894 | 0.500039 | 0.500046 | 0.031418 |
| ff_proj | -0.000450 | -0.000474 | 0.990588 | 0.500331 | 0.500329 | 0.030667 |
| k_proj | 0.000890 | 0.000991 | 0.994540 | 0.499419 | 0.499402 | 0.027285 |
| q_proj | 0.000599 | 0.000670 | 0.994832 | 0.499721 | 0.499710 | 0.026559 |
| up_proj | -0.0000610 | -0.0000628 | 0.992297 | 0.500044 | 0.500050 | 0.027178 |
| v_proj | -0.000198 | -0.000189 | 0.995639 | 0.500075 | 0.500071 | 0.023991 |

### Sign consistency and spike ratio

Quantiles use deterministic proportional samples of approximately one million
weights per scope; means, minima, and maxima use all weights.

| Scope | C mean | C p50 | C p90 | C p99 | T mean | T p50 | T p90 | T p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 0.127214 | 0.102779 | 0.273896 | 0.448212 | 1.249317 | 1.293649 | 1.582258 | 2.245786 |
| attn_out | 0.135425 | 0.112447 | 0.283736 | 0.449243 | 1.301372 | 1.318899 | 1.549000 | 1.950500 |
| ff_out | 0.132678 | 0.107938 | 0.281812 | 0.460012 | 1.283523 | 1.299152 | 1.596008 | 2.330767 |
| ff_proj | 0.129989 | 0.104790 | 0.279177 | 0.456577 | 1.284498 | 1.321725 | 1.646829 | 2.381367 |
| k_proj | 0.108415 | 0.084080 | 0.243390 | 0.405714 | 1.114635 | 1.222147 | 1.518465 | 2.130708 |
| q_proj | 0.107329 | 0.083264 | 0.240570 | 0.399491 | 1.104032 | 1.215227 | 1.491865 | 2.071066 |
| up_proj | 0.127483 | 0.103141 | 0.272673 | 0.449597 | 1.246209 | 1.283166 | 1.559857 | 2.257773 |
| v_proj | 0.132157 | 0.108471 | 0.281081 | 0.448868 | 1.278391 | 1.317165 | 1.571347 | 2.034936 |

Overall C ranged from 0 to 0.979992. Overall T ranged from approximately zero
to 8.473275.

### Signed SUM balance

The overall negative-SUM fraction was 0.5000328 (3,489,889,558 of
6,979,321,856 weights). Module-type fractions were: attn_out 0.5001017,
ff_out 0.5000141, ff_proj 0.5002291, k_proj 0.4996103, q_proj 0.4998402,
up_proj 0.5000276, and v_proj 0.5000614. Layer-weighted fractions ranged only
from 0.4995620 (layer 2) to 0.5002591 (layer 26).

### Largest ABS/SQUARE disagreements

The matrix-level ABS/SQUARE XOR median was 0.026771; p95 was 0.037202 and the
maximum was 0.046619. The clearest cluster was late `ff_out`: layers 26--31 had
XOR 0.040212, 0.045240, 0.046619, 0.044115, 0.043005, and 0.041119,
respectively. Layer 28 `ff_out` was the maximum, with Spearman 0.985198. Other
large disagreements included layer 0 `k_proj` (0.039447), layer 0 `v_proj`
(0.037632), layer 28 `ff_proj` (0.037446), and layer 30 `up_proj` (0.037367).

### Diagnostic interpretation

SUM is effectively rank-uncorrelated with both magnitude aggregations and
changes about half of all prune/keep decisions. ABS and SQUARE are strongly
rank-correlated, but their 2.85% overall mask disagreement is nonzero and rises
above 4% in several late `ff_out` matrices. Low overall sign consistency
(mean 0.127) indicates substantial cancellation across the 80 states, while a
spike ratio above one for most sampled weights shows that state effects are not
uniform in magnitude. These are score diagnostics only; they do not establish
which mask improves GSM8K.

## Interpretation

### Hypothesis

Signed aggregation may lose magnitude information through cancellation, while
squaring may either usefully emphasize rare large effects or overweight spikes.

### Observed result

The implementation and resource gates pass. The two-state smoke shows that SUM
and the magnitude-based rankings can differ substantially, while ABS and
SQUARE remain close but are not identical. The one-state ABS/SQUARE identity is
expected and confirms no basis for treating identical rankings as an error.
The full run confirms that SUM produces a materially different ranking, while
ABS and SQUARE are close but still disagree on 2.85% of prune/keep decisions.

### Interpretation and decision

These diagnostics do not answer which aggregation improves GSM8K. They only
show that the variants are implemented distinctly where multiple states permit
them to differ and that the run fits the machine. Full scoring is complete;
downstream pruning and evaluation remain paused for explicit review. Later GSM8K
interpretation will use the preregistered rules: ABS over SUM suggests harmful
sign cancellation; SUM over ABS suggests useful directional consistency;
SQUARE over ABS suggests useful emphasis on large rare events; ABS over SQUARE
suggests excessive spike emphasis; similar results suggest aggregation is not
the main limitation.
