# Exact-budget Role Trade-off DP @65%

- Supported frontier points: 33; search capped: False
- Global sparsity: 0.6499211238
- Current Role risks: [11.31084233506979, 12.0861817783668]
- Supported minimax relative risks: [0.994642586172025, 0.9946026292578339]
- Supported minimax changed projections: 30
- Exact local-Max relative risks: [0.9988665346831626, 0.9936653300924132]
- Exact local-Max changed projections: 26

## Crossfit
```json
{
  "unmasked": {
    "mean_relative_change": [
      0.004051712240266431,
      -0.00794836602847275
    ],
    "median_relative_change": [
      0.004732705287638472,
      -0.007421481571032951
    ],
    "both_improve_sequences": 0,
    "improve_sequences_by_role": [
      0,
      8
    ]
  },
  "policy": {
    "mean_relative_change": [
      -0.00542737492793352,
      -0.005520516777851578
    ],
    "median_relative_change": [
      -0.005142958947433052,
      -0.004990097527421855
    ],
    "both_improve_sequences": 8,
    "improve_sequences_by_role": [
      8,
      8
    ]
  },
  "masked": {
    "mean_relative_change": [
      -0.008385950786225421,
      0.003053451828174028
    ],
    "median_relative_change": [
      -0.008100962438193138,
      0.0038260215649829066
    ],
    "both_improve_sequences": 1,
    "improve_sequences_by_role": [
      8,
      1
    ]
  }
}
```

## Limits
- Frontier contains all discovered supported points, not unsupported discrete Pareto points.
- Exact optimality applies to every scalar DP solve and the exact local-Max solve.
- Crossfit is descriptive and proxy-only; no downstream evaluation was run.
