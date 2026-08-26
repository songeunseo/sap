# Stage 0 DLM sensitivity feasibility

## Result: GO — Stage 1 authorized

Command (exit 0), at `7e7bbd3e9e06f48559ce514dcb97b3c13662c2dd`:

```bash
python dlm_gradient_sensitivity.py feasibility --config codex/time_risk_sensitivity/config.json --output codex/time_risk_sensitivity/results/stage0.json
```

The preflight/log is `stage0-run.log`: 2026-08-18T15:19:13Z to
2026-08-18T15:19:40Z (local 2026-08-19T00:19:13+0900 to
2026-08-19T00:19:40+0900). Environment: RTX 5090 (32607 MiB; driver 570.211.01,
CUDA 12.8), 91 GiB RAM / 8 GiB swap, Python 3.10.19, torch 2.8.0+cu128,
transformers 4.49.0, accelerate 0.34.2, datasets 2.21.0.

## Raw measurements and gates

| block | state / finish s | allocated / reserved GiB (cuda:0) | swap delta KiB | finite / elements; nonzero |
| --- | --- | --- | --- | --- |
| 31 | 0.488280099 / 1.776394170 | 15.422616 / 15.525391 | 0 | 218103808 / 218103808; 218103808 |
| 0 | 0.475722279 / 1.809395336 | 16.157689 / 16.201172 | 0 | 218103808 / 218103808; 218103808 |

The state timer encloses the gradient backward/FP32 CPU cast and real
`add_state(..., "a")` split-sum update; the seven replayed fills are outside
that timer. `finish_timestep()` was timed separately. Both allocated and
reserved peaks are below the inclusive 30 GiB limit, both swap deltas are zero,
and all gradients are finite and nonzero.

## Independent projection audit

Raw two-point slope: `(0.475722279400 - 0.488280098885) / 31 =
-0.000405090951 s/block`. The fit is therefore `constrained` with
`max_endpoint_constant`: fixed `0.488280098885 s`, suffix `0 s`, and all 32
per-block costs set to the slower endpoint. Independent recomputation gives:

- scoring: `80 * 32 * 0.488280098885 = 1249.997053146 s`
- finalization: `10 * 32 * 1.809395335615 = 579.006507397 s`
- total: `1829.003560543 s = 0.508056545 GPU h`

This equals `stage0.json`. Every declared gate predicate is true: memory,
swap, finite gradients, nonzero gradients, and total runtime <= 24 GPU hours.
The projection is inside the user's 12-hour result window (`0.5081 h < 12 h`);
this does not alter the predeclared 24-hour scientific gate.

## Next authorized stage

GO: run Stage 1 without enabling checkpointing or changing the Stage 0 bounds.

## Stage 1 execution: BLOCKED (operational)

The committed scorer (`fcfc68e61fd2140703b90cccf20a6c1adc5c7899`) was run with
the predeclared pilot command at 2026-08-18T16:25:00Z. The block-31 repeated
gradient-square check passed (finite, absolute difference `0.0`, relative
difference `0.0`, within `1e-6` / `1e-5` tolerances), but the first scoring
block stopped before artifact publication:

```text
RuntimeError: quantile() input tensor is too large
  lib/dlm_gradient_sensitivity.py:714 in score_block
```

No packed block, `stage1.json`, or reliability decision exists: the manifest
has 0/32 completed blocks and states 224 expected module entries, 10
full/A/B updates, 16 variants, and 112 packed entries per completed block.
The ignored `stage1-run.log`, `masks/manifest.json`, and `masks/states.json`
are preserved for debugging. Preflight recorded RTX 5090 with 32,103 MiB free,
49 GiB available RAM, 34 GiB disk headroom, and an already-full 8 GiB swap.

This is not a scientific NO-GO: the split-Spearman, split-Jaccard, and
all-identical-mask predicates were not evaluated. Stage 1 and Task 8 remain
unauthorized pending a scorer fix and a fresh checksum-verified resume.

## Stage 1 final result: NO-GO

The runtime fix at `592726a` completed the predeclared scorer command on
2026-08-18T16:37:07Z–18:49:45Z (2:12:38; exit 2 is the scorer's declared
NO-GO exit). The block-31 repeat remained finite and exact: absolute and
relative gradient-square maxima `0.0`, within `1e-6` / `1e-5` tolerances.

Independent audit passed: 32 checksum-valid blocks; 224 module entries; 10
full/A/B Welford updates each; 16 full variants and 112 entries per block
(3,584 masks); exact row prune counts and checksums for every packed mask;
one model/config/state binding; and no persisted split masks. Model/config/state
digests are `23431f0f`, `ac29cb07`, and `1e44d30e`, respectively.

The split-Spearman predicate passes: median module rho is `0.884158` (range
`0.747690`–`0.956689`; threshold `>= 0.5`; no undefined modules). The
split-mask predicate fails independently at every decision sparsity:

| sparsity | mean lambda-1 split Jaccard | threshold |
| --- | ---: | ---: |
| 0.50 | 0.792833 | 0.95 |
| 0.60 | 0.812856 | 0.95 |
| 0.70 | 0.838615 | 0.95 |

Useful masks are not all identical to Mean, so that separate stop is false.
Reliability nevertheless fails, therefore Stage 1 is **NO-GO** and Task 8 is
not authorized. The ignored run/audit logs retain stdout, stderr, resource
samples, and full audit evidence; packed blocks remain ignored.
