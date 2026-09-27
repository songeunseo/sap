# Full experiment closeout

CPU verification passed: canonical checkpoint names, exact coverage, fingerprints, official regrading, paired statistics, mask file hashes, fresh diagnostics and historical A/Multi200 reproduction.



### 5.1 Prior development and separate100 comparisons

| Method | Development100 correct | Separate100 correct |
|---|---:|---:|
| Multi | 63/100 | 59/100 |
| A | 60/100 | 58/100 |
| Uniform | 54/100 | 52/100 |

| Contrast | Gain/loss | Net | Exact p | Holm p |
|---|---:|---:|---:|---:|
| Multi−A | 6/5 | +1 | 1 | 1 |
| Multi−Uniform | 13/6 | +7 | 0.16707 | 0.50121 |
| A−Uniform | 13/7 | +6 | 0.26318 | 0.52635 |

The separate100 sample was fixed before the current method selection and is disjoint from development100. It is not certified unseen across the entire project. The Multi−A paired difference is small and does not establish an incremental C effect. Failure to reject a difference also does not establish equivalence. These observations motivate the larger fixed comparison without determining its outcome.

### 5.2 Full comparison

| Arm | Full1319 | Previously examined200 | Primary1119 |
|---|---:|---:|---:|
| A | 764/1319 | 118/200 | 646/1119 |
| Multi | 758/1319 | 122/200 | 636/1119 |
| Cross | 757/1319 | 120/200 | 637/1119 |
| CrossMatched | 755/1319 | 121/200 | 634/1119 |

| Primary contrast | Gain/loss | Difference (pp) | Exact p | Holm p | Unadjusted 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| Multi-A | 64/74 | -0.894 | 0.44372 | 1 | [-2.949, 1.162] |
| Multi-Cross | 74/75 | -0.089 | 1 | 1 | [-2.234, 1.966] |
| Multi-CrossMatched | 67/65 | +0.179 | 0.93068 | 1 | [-1.787, 2.145] |

Multi-A shows a negative difference (-10 answers) and does not pass the Holm-adjusted 0.05 threshold. Multi-Cross shows a negative difference (-1 answers) and does not pass the Holm-adjusted 0.05 threshold. Multi-CrossMatched shows a positive difference (+2 answers) and does not pass the Holm-adjusted 0.05 threshold. These pairwise findings do not establish equivalence, universal necessity, or generalization. The contribution claim must be reviewed jointly with the control results and the uncertainty intervals.

### 5.3 Fidelity and cost

| Arm | A | Natural C | Cross C | Query CE | Response sign flip rate |
|---|---:|---:|---:|---:|---:|
| A | 0.732798 | 0.868318 | 1.01725 | 3.00819 | 0.0758929 |
| Multi | 0.750689 | 0.867225 | 1.00993 | 3.02124 | 0.0744978 |
| Cross | 0.724621 | 0.835103 | 0.975545 | 3.01986 | 0.0742188 |
| CrossMatched | 0.724543 | 0.83233 | 0.975925 | 3.02052 | 0.0739397 |

Diagnostic fidelity and task capability measure different properties. A lower response loss is not itself evidence of improved answer accuracy or a causal explanation for any observed gain.

The current run took 8.236 wall-clock hours. Current plus preserved first-attempt records contain 1,351,936 model forward calls. This count is complete for the recorded current and preserved first attempts. It excludes earlier construction of reused probes, calibration artifacts and masks, so it is not the end-to-end method cost.


## Error categories and cost gaps

```json
{
  "errors": {
    "full_1319": {
      "A": {
        "correct": 764,
        "valid_wrong": 487,
        "strict_invalid": 68
      },
      "Multi": {
        "correct": 758,
        "valid_wrong": 485,
        "strict_invalid": 76
      },
      "Cross": {
        "correct": 757,
        "valid_wrong": 483,
        "strict_invalid": 79
      },
      "CrossMatched": {
        "correct": 755,
        "valid_wrong": 481,
        "strict_invalid": 83
      }
    },
    "previously_seen_200": {
      "A": {
        "correct": 118,
        "valid_wrong": 70,
        "strict_invalid": 12
      },
      "Multi": {
        "correct": 122,
        "valid_wrong": 70,
        "strict_invalid": 8
      },
      "Cross": {
        "correct": 120,
        "valid_wrong": 71,
        "strict_invalid": 9
      },
      "CrossMatched": {
        "correct": 121,
        "valid_wrong": 69,
        "strict_invalid": 10
      }
    },
    "primary_remaining_1119": {
      "A": {
        "valid_wrong": 417,
        "correct": 646,
        "strict_invalid": 56
      },
      "Multi": {
        "valid_wrong": 415,
        "correct": 636,
        "strict_invalid": 68
      },
      "Cross": {
        "strict_invalid": 70,
        "correct": 637,
        "valid_wrong": 412
      },
      "CrossMatched": {
        "valid_wrong": 412,
        "correct": 634,
        "strict_invalid": 73
      }
    }
  },
  "costs": {
    "runs": {
      "current": {
        "attempt_count": 49,
        "cost_count": 49,
        "recorded_forwards": 1351808,
        "sum_worker_wall_seconds": 115048.9721346465,
        "missing_costs": [],
        "orphan_costs": [],
        "statuses": {
          "complete": 49
        }
      },
      "preserved_first_attempt": {
        "attempt_count": 1,
        "cost_count": 1,
        "recorded_forwards": 128,
        "sum_worker_wall_seconds": 77.60248645301908,
        "missing_costs": [],
        "orphan_costs": [],
        "statuses": {
          "complete": 1
        }
      }
    },
    "recorded_forwards_total": 1351936,
    "complete_accounting": true,
    "counts_are_lower_bounds": false,
    "scope": "Current full evaluation plus preserved scheduler-failure attempt; excludes earlier reused probe/bank/model construction"
  }
}
```

Scientific interpretation and Obsidian synchronization remain pending agent review.
