# Projection-Wise Functional Capacity Allocation @ 65%

## Question

Can changing projection-level sparsity allocation improve pruning under an identical global parameter budget?

## Frozen Setup

GSAI-ML/LLaDA-8B-Base, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`; all 224 historical projections. Standard Wanda uses the exact unweighted 80-state DLM activation cache and stable row-wise ranking from the cited uniform sweep. All Uniform-65 mask hashes match that sweep. Grid: 50/55/60/65/70/75%. Allocation uses only the mean state masked-token Dense||Sparse KL. All masks, state digests and corpus intervals were verified; held-out 40 states are disjoint from allocation 80 states. Calibration uses a seven-variant same-path dense sham; held-out evaluation uses batch-one full forwards and exact repeated-dense sham equality.

## Projection Capacity Curves

| Sparsity | Mean damage | Median | p10 | p90 | CV |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.00133141 | 0.00126059 | 0.000770127 | 0.00169476 | 0.688199 |
| 0.55 | 0.00171493 | 0.00155855 | 0.000887328 | 0.00230122 | 0.912924 |
| 0.6 | 0.00226691 | 0.00196383 | 0.00102085 | 0.0030753 | 1.17871 |
| 0.65 | 0.00307862 | 0.00256452 | 0.00122851 | 0.00422865 | 1.43673 |
| 0.7 | 0.00426897 | 0.00339597 | 0.00151193 | 0.00586129 | 1.70229 |
| 0.75 | 0.00601502 | 0.00461571 | 0.00194386 | 0.00825512 | 1.93267 |

Raw curves contain 11 negative increments. No envelope or manual correction was used.
Full raw curves, per-state diagnostics, marginal distributions, layer/type breakdowns and top/bottom 20 increments are in `capacity_curves_raw.json` and `marginal_costs.json`.

## Frozen Allocation

Both models prune exactly 4,536,008,704 / 6,979,321,856 weights (64.9921123798%); nominal target is 65%. The difference from mathematical 65% is historical row flooring. Budget difference between methods: 0.
Grid counts: {'0.5': 20, '0.55': 17, '0.6': 54, '0.65': 64, '0.7': 40, '0.75': 29}. Exact-budget feasibility skips: 28.
Layer-index Spearman: -0.5835120714973884; layer-quartile means: [0.7089285714285715, 0.6348214285714285, 0.6196428571428572, 0.5919642857142857].
These are descriptive patterns from the frozen rule. They did not alter it.

| Projection | Assigned sparsity | Pruned weights |
|---|---:|---:|
| block_00.attn_out | 70% | 11743232 |
| block_00.ff_out | 75% | 37748736 |
| block_00.q_proj | 70% | 11743232 |
| block_00.k_proj | 60% | 10063872 |
| block_00.v_proj | 50% | 8388608 |
| block_00.ff_proj | 75% | 37748736 |
| block_00.up_proj | 75% | 37748736 |
| block_01.attn_out | 75% | 12582912 |
| block_01.ff_out | 75% | 37748736 |
| block_01.q_proj | 75% | 12582912 |
| block_01.k_proj | 75% | 12582912 |
| block_01.v_proj | 75% | 12582912 |
| block_01.ff_proj | 75% | 37748736 |
| block_01.up_proj | 75% | 37748736 |
| block_02.attn_out | 75% | 12582912 |
| block_02.ff_out | 75% | 37748736 |
| block_02.q_proj | 75% | 12582912 |
| block_02.k_proj | 75% | 12582912 |
| block_02.v_proj | 75% | 12582912 |
| block_02.ff_proj | 75% | 37748736 |
| block_02.up_proj | 75% | 37748736 |
| block_03.attn_out | 75% | 12582912 |
| block_03.ff_out | 75% | 37748736 |
| block_03.q_proj | 75% | 12582912 |
| block_03.k_proj | 75% | 12582912 |
| block_03.v_proj | 75% | 12582912 |
| block_03.ff_proj | 70% | 35229696 |
| block_03.up_proj | 70% | 35229696 |
| block_04.attn_out | 75% | 12582912 |
| block_04.ff_out | 75% | 37748736 |
| block_04.q_proj | 75% | 12582912 |
| block_04.k_proj | 75% | 12582912 |
| block_04.v_proj | 65% | 10903552 |
| block_04.ff_proj | 70% | 35229696 |
| block_04.up_proj | 70% | 35229696 |
| block_05.attn_out | 65% | 10903552 |
| block_05.ff_out | 75% | 37748736 |
| block_05.q_proj | 75% | 12582912 |
| block_05.k_proj | 70% | 11743232 |
| block_05.v_proj | 65% | 10903552 |
| block_05.ff_proj | 70% | 35229696 |
| block_05.up_proj | 70% | 35229696 |
| block_06.attn_out | 60% | 10063872 |
| block_06.ff_out | 70% | 35229696 |
| block_06.q_proj | 75% | 12582912 |
| block_06.k_proj | 70% | 11743232 |
| block_06.v_proj | 60% | 10063872 |
| block_06.ff_proj | 65% | 32710656 |
| block_06.up_proj | 70% | 35229696 |
| block_07.attn_out | 65% | 10903552 |
| block_07.ff_out | 70% | 35229696 |
| block_07.q_proj | 70% | 11743232 |
| block_07.k_proj | 70% | 11743232 |
| block_07.v_proj | 60% | 10063872 |
| block_07.ff_proj | 65% | 32710656 |
| block_07.up_proj | 65% | 32710656 |
| block_08.attn_out | 65% | 10903552 |
| block_08.ff_out | 70% | 35229696 |
| block_08.q_proj | 65% | 10903552 |
| block_08.k_proj | 70% | 11743232 |
| block_08.v_proj | 60% | 10063872 |
| block_08.ff_proj | 65% | 32710656 |
| block_08.up_proj | 65% | 32710656 |
| block_09.attn_out | 60% | 10063872 |
| block_09.ff_out | 65% | 32714752 |
| block_09.q_proj | 70% | 11743232 |
| block_09.k_proj | 70% | 11743232 |
| block_09.v_proj | 60% | 10063872 |
| block_09.ff_proj | 65% | 32710656 |
| block_09.up_proj | 65% | 32710656 |
| block_10.attn_out | 60% | 10063872 |
| block_10.ff_out | 70% | 35229696 |
| block_10.q_proj | 60% | 10063872 |
| block_10.k_proj | 65% | 10903552 |
| block_10.v_proj | 60% | 10063872 |
| block_10.ff_proj | 65% | 32710656 |
| block_10.up_proj | 65% | 32710656 |
| block_11.attn_out | 60% | 10063872 |
| block_11.ff_out | 70% | 35229696 |
| block_11.q_proj | 65% | 10903552 |
| block_11.k_proj | 70% | 11743232 |
| block_11.v_proj | 65% | 10903552 |
| block_11.ff_proj | 60% | 30191616 |
| block_11.up_proj | 65% | 32710656 |
| block_12.attn_out | 65% | 10903552 |
| block_12.ff_out | 65% | 32714752 |
| block_12.q_proj | 65% | 10903552 |
| block_12.k_proj | 65% | 10903552 |
| block_12.v_proj | 60% | 10063872 |
| block_12.ff_proj | 60% | 30191616 |
| block_12.up_proj | 65% | 32710656 |
| block_13.attn_out | 55% | 9224192 |
| block_13.ff_out | 70% | 35229696 |
| block_13.q_proj | 55% | 9224192 |
| block_13.k_proj | 55% | 9224192 |
| block_13.v_proj | 60% | 10063872 |
| block_13.ff_proj | 60% | 30191616 |
| block_13.up_proj | 65% | 32710656 |
| block_14.attn_out | 60% | 10063872 |
| block_14.ff_out | 70% | 35229696 |
| block_14.q_proj | 60% | 10063872 |
| block_14.k_proj | 65% | 10903552 |
| block_14.v_proj | 60% | 10063872 |
| block_14.ff_proj | 60% | 30191616 |
| block_14.up_proj | 65% | 32710656 |
| block_15.attn_out | 55% | 9224192 |
| block_15.ff_out | 70% | 35229696 |
| block_15.q_proj | 60% | 10063872 |
| block_15.k_proj | 65% | 10903552 |
| block_15.v_proj | 60% | 10063872 |
| block_15.ff_proj | 60% | 30191616 |
| block_15.up_proj | 65% | 32710656 |
| block_16.attn_out | 55% | 9224192 |
| block_16.ff_out | 70% | 35229696 |
| block_16.q_proj | 60% | 10063872 |
| block_16.k_proj | 60% | 10063872 |
| block_16.v_proj | 60% | 10063872 |
| block_16.ff_proj | 60% | 30191616 |
| block_16.up_proj | 65% | 32710656 |
| block_17.attn_out | 55% | 9224192 |
| block_17.ff_out | 70% | 35229696 |
| block_17.q_proj | 60% | 10063872 |
| block_17.k_proj | 60% | 10063872 |
| block_17.v_proj | 60% | 10063872 |
| block_17.ff_proj | 65% | 32710656 |
| block_17.up_proj | 65% | 32710656 |
| block_18.attn_out | 55% | 9224192 |
| block_18.ff_out | 70% | 35229696 |
| block_18.q_proj | 65% | 10903552 |
| block_18.k_proj | 65% | 10903552 |
| block_18.v_proj | 60% | 10063872 |
| block_18.ff_proj | 65% | 32710656 |
| block_18.up_proj | 65% | 32710656 |
| block_19.attn_out | 55% | 9224192 |
| block_19.ff_out | 70% | 35229696 |
| block_19.q_proj | 60% | 10063872 |
| block_19.k_proj | 60% | 10063872 |
| block_19.v_proj | 60% | 10063872 |
| block_19.ff_proj | 65% | 32710656 |
| block_19.up_proj | 65% | 32710656 |
| block_20.attn_out | 50% | 8388608 |
| block_20.ff_out | 70% | 35229696 |
| block_20.q_proj | 65% | 10903552 |
| block_20.k_proj | 65% | 10903552 |
| block_20.v_proj | 55% | 9224192 |
| block_20.ff_proj | 65% | 32710656 |
| block_20.up_proj | 70% | 35229696 |
| block_21.attn_out | 50% | 8388608 |
| block_21.ff_out | 70% | 35229696 |
| block_21.q_proj | 60% | 10063872 |
| block_21.k_proj | 60% | 10063872 |
| block_21.v_proj | 55% | 9224192 |
| block_21.ff_proj | 65% | 32710656 |
| block_21.up_proj | 70% | 35229696 |
| block_22.attn_out | 55% | 9224192 |
| block_22.ff_out | 70% | 35229696 |
| block_22.q_proj | 60% | 10063872 |
| block_22.k_proj | 60% | 10063872 |
| block_22.v_proj | 55% | 9224192 |
| block_22.ff_proj | 65% | 32710656 |
| block_22.up_proj | 65% | 32710656 |
| block_23.attn_out | 50% | 8388608 |
| block_23.ff_out | 70% | 35229696 |
| block_23.q_proj | 60% | 10063872 |
| block_23.k_proj | 60% | 10063872 |
| block_23.v_proj | 55% | 9224192 |
| block_23.ff_proj | 65% | 32710656 |
| block_23.up_proj | 65% | 32710656 |
| block_24.attn_out | 50% | 8388608 |
| block_24.ff_out | 65% | 32714752 |
| block_24.q_proj | 65% | 10903552 |
| block_24.k_proj | 60% | 10063872 |
| block_24.v_proj | 50% | 8388608 |
| block_24.ff_proj | 65% | 32710656 |
| block_24.up_proj | 65% | 32710656 |
| block_25.attn_out | 50% | 8388608 |
| block_25.ff_out | 65% | 32714752 |
| block_25.q_proj | 65% | 10903552 |
| block_25.k_proj | 65% | 10903552 |
| block_25.v_proj | 50% | 8388608 |
| block_25.ff_proj | 60% | 30191616 |
| block_25.up_proj | 65% | 32710656 |
| block_26.attn_out | 50% | 8388608 |
| block_26.ff_out | 70% | 35229696 |
| block_26.q_proj | 60% | 10063872 |
| block_26.k_proj | 60% | 10063872 |
| block_26.v_proj | 50% | 8388608 |
| block_26.ff_proj | 60% | 30191616 |
| block_26.up_proj | 60% | 30191616 |
| block_27.attn_out | 50% | 8388608 |
| block_27.ff_out | 65% | 32714752 |
| block_27.q_proj | 65% | 10903552 |
| block_27.k_proj | 65% | 10903552 |
| block_27.v_proj | 50% | 8388608 |
| block_27.ff_proj | 60% | 30191616 |
| block_27.up_proj | 65% | 32710656 |
| block_28.attn_out | 55% | 9224192 |
| block_28.ff_out | 70% | 35229696 |
| block_28.q_proj | 70% | 11743232 |
| block_28.k_proj | 65% | 10903552 |
| block_28.v_proj | 50% | 8388608 |
| block_28.ff_proj | 60% | 30191616 |
| block_28.up_proj | 60% | 30191616 |
| block_29.attn_out | 60% | 10063872 |
| block_29.ff_out | 70% | 35229696 |
| block_29.q_proj | 65% | 10903552 |
| block_29.k_proj | 65% | 10903552 |
| block_29.v_proj | 55% | 9224192 |
| block_29.ff_proj | 60% | 30191616 |
| block_29.up_proj | 60% | 30191616 |
| block_30.attn_out | 50% | 8388608 |
| block_30.ff_out | 65% | 32714752 |
| block_30.q_proj | 70% | 11743232 |
| block_30.k_proj | 60% | 10063872 |
| block_30.v_proj | 50% | 8388608 |
| block_30.ff_proj | 55% | 27672576 |
| block_30.up_proj | 55% | 27672576 |
| block_31.attn_out | 50% | 8388608 |
| block_31.ff_out | 50% | 25165824 |
| block_31.q_proj | 65% | 10903552 |
| block_31.k_proj | 60% | 10063872 |
| block_31.v_proj | 50% | 8388608 |
| block_31.ff_proj | 50% | 25165824 |
| block_31.up_proj | 50% | 25165824 |

## Full-Model Held-Out DLM Evaluation

| Method | Global sparsity | Mean KL | Median KL | Top-1 agreement |
|--------|----------------:|--------:|----------:|----------------:|
| Uniform Wanda | 64.99211238% | 0.561242 | 0.3977715 | 0.6445247 |
| Capacity Wanda | 64.99211238% | 0.4553846 | 0.2940221 | 0.66143 |

Mean KL averages state-level masked-token means; median KL pools masked tokens.
Paired capacity-minus-uniform mean: -0.1058574; median: -0.09888586; 95% bootstrap CI: [-0.1264912104140967, -0.08549042319878937]. 20,000 paired state resamples, seed 0.
States improved/worsened/tied: 37/3/0. Sequence means improved: 8/8. Timestep means improved: 5/5.

| Group | Uniform mean KL | Capacity mean KL | Difference |
|---|---:|---:|---:|
| per_sequence: 8 | 0.5541683 | 0.4702172 | -0.08395107 |
| per_sequence: 9 | 0.5153897 | 0.4088267 | -0.106563 |
| per_sequence: 10 | 0.4738441 | 0.3904711 | -0.08337297 |
| per_sequence: 11 | 0.7629048 | 0.583245 | -0.1796598 |
| per_sequence: 12 | 0.7467949 | 0.603767 | -0.1430279 |
| per_sequence: 13 | 0.5659071 | 0.4886409 | -0.07726627 |
| per_sequence: 14 | 0.4613195 | 0.3769832 | -0.08433628 |
| per_sequence: 15 | 0.4096075 | 0.3209258 | -0.08868172 |
| per_timestep: 0.1 | 0.4352376 | 0.3775614 | -0.05767619 |
| per_timestep: 0.3 | 0.5281721 | 0.4320685 | -0.09610356 |
| per_timestep: 0.5 | 0.7097596 | 0.6038606 | -0.105899 |
| per_timestep: 0.7 | 0.7135677 | 0.5432859 | -0.1702818 |
| per_timestep: 0.9 | 0.419473 | 0.3201467 | -0.09932628 |

## Kill-Gate Decision

### SUPPORTED

Projection-wise functional capacity allocation improves held-out full-model DLM fidelity under the same global sparsity budget. Allocation has real decision-level leverage; the next task is to approximate this oracle with a cheap DLM-specific signal.

## Downstream

```json
{
  "evaluations": [
    {
      "capacity": {
        "correct": 20,
        "limit": 100,
        "metrics": {
          "accuracy": 0.2,
          "eval_seconds": 2390.733841650188,
          "examples_per_second": 0.041828160984652175,
          "full_num_examples": 1319,
          "num_examples": 100
        },
        "predictions": "/home/tmluser1/sap/experiments/projection_capacity_allocation_65/gsm8k/capacity_100_predictions.jsonl",
        "sha256": "f7c04baf8c3922067d9778e8656fef66afaadb1de80b231dfc929285e906101b"
      },
      "uniform": {
        "correct": 12,
        "limit": 100,
        "metrics": {
          "accuracy": 0.12,
          "eval_seconds": 3787.4957195350435,
          "examples_per_second": 0.02640267010315621,
          "full_num_examples": 1319,
          "num_examples": 100
        },
        "predictions": "/home/tmluser1/sap/experiments/projection_capacity_allocation_65/gsm8k/uniform_100_predictions.jsonl",
        "sha256": "2517601bf8e6aafd3caf11af1d21f33faed4ce4f8ce1473a68d1f074deb2a706"
      }
    },
    {
      "capacity": {
        "correct": 250,
        "limit": 1319,
        "metrics": {
          "accuracy": 0.18953752843062927,
          "eval_seconds": 27934.030428440077,
          "examples_per_second": 0.04721839203902009,
          "full_num_examples": 1319,
          "num_examples": 1319
        },
        "predictions": "/home/tmluser1/sap/experiments/projection_capacity_allocation_65/gsm8k/capacity_1319_predictions.jsonl",
        "sha256": "b36f46d0524b1faa9435090535a5f08b06bfc912d8123fb92ae6071cb92e5d4a"
      },
      "uniform": {
        "correct": 139,
        "limit": 1319,
        "metrics": {
          "accuracy": 0.10538286580742987,
          "eval_seconds": 38156.98698827019,
          "examples_per_second": 0.03456771889262307,
          "full_num_examples": 1319,
          "num_examples": 1319
        },
        "predictions": "/home/tmluser1/sap/experiments/projection_capacity_allocation_65/gsm8k/uniform_1319_predictions.jsonl",
        "sha256": "9f9fee4f951173f118db6e191c7b793d40402d5c880cf01b3e2daea0bb656897"
      }
    }
  ],
  "mini_directionally_better": true,
  "protocol": {
    "evaluation": {
      "block_length": 256,
      "bootstrap_iters": 0,
      "denoising_steps": 256,
      "fewshot_seed": 1234,
      "generation_length": 256,
      "max_cuda_gib": 30,
      "num_fewshot": 5,
      "numpy_seed": 1234,
      "order": [
        "Dense",
        "DLM-SUM",
        "DLM-ABS",
        "DLM-SQUARE",
        "Wanda",
        "SparseGPT"
      ],
      "pathological_slowdown_factor": 3.0,
      "primary_filter": "strict-match",
      "primary_metric": "exact_match,strict-match",
      "random_seed": 0,
      "task": "gsm8k",
      "temperature": 0,
      "timing_examples": 32,
      "torch_seed": 1234
    },
    "gsm8k_task_sha256": "442b817bc83138ec89bd30ea9d6b4acac977dfc4653c177ad778f354c8543d29",
    "lm_eval_version": "0.4.8",
    "model": {
      "dtype": "bfloat16",
      "id": "GSAI-ML/LLaDA-8B-Base",
      "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
    }
  },
  "protocol_hash": "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add"
}
```
