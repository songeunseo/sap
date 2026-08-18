# Stage 0 DLM sensitivity feasibility

## Result: GO — Stage 1 authorized

Command (exit 0), at `7e7bbd3e9e06f48559ce514dcb97b3c13662c2dd`:

```bash
python dlm_gradient_sensitivity.py feasibility --config experiments/time_risk/pilot.json --output results/time_risk/stage0.json
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
The separate 12-hour result target is also met (`0.5081 h <= 8 h`); this does
not alter the predeclared 24-hour scientific gate.

## Next authorized stage

GO: run Stage 1 without enabling checkpointing or changing the Stage 0 bounds.
