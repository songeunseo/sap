# Role–Aggregate exact-budget bundle mini100

## Setup

Frozen six bundles, two backgrounds, 12 hybrids. Same mini100/5shot/256 steps/strict EM, original Wanda masks, equal exact global budget. Baselines Aggregate19, Role24. No downstream-driven allocation selection.

## Results

Benefit sign: add=hybrid−Aggregate; revert=Role−hybrid. Positive means Role-side bundle helps in that background.

| Hybrid | Correct /100 | Role-side benefit (questions) | raw p | Holm p (12) |
|---|---:|---:|---:|---:|
| add_b0 | 21 | +2 | 0.72656 | 1.00000 |
| add_b1 | 20 | +1 | 1.00000 | 1.00000 |
| add_b2 | 20 | +1 | 1.00000 | 1.00000 |
| add_b3 | 20 | +1 | 1.00000 | 1.00000 |
| add_b4 | 18 | -1 | 1.00000 | 1.00000 |
| add_b5 | 17 | -2 | 0.72656 | 1.00000 |
| revert_b0 | 20 | +4 | 0.34375 | 1.00000 |
| revert_b1 | 19 | +5 | 0.17969 | 1.00000 |
| revert_b2 | 22 | +2 | 0.68750 | 1.00000 |
| revert_b3 | 24 | +0 | 1.00000 | 1.00000 |
| revert_b4 | 20 | +4 | 0.28906 | 1.00000 |
| revert_b5 | 24 | +0 | 1.00000 | 1.00000 |

## Background interaction

Role-side benefit in Role background minus benefit in Aggregate background. CI is unadjusted and exploratory, not a multiplicity-controlled claim.

- B0: {'role_background_benefit_minus_aggregate_background_benefit': 2, 'accuracy_ci95_unadjusted_exploratory': [-0.06, 0.1]}
- B1: {'role_background_benefit_minus_aggregate_background_benefit': 4, 'accuracy_ci95_unadjusted_exploratory': [-0.06, 0.13]}
- B2: {'role_background_benefit_minus_aggregate_background_benefit': 1, 'accuracy_ci95_unadjusted_exploratory': [-0.07, 0.09]}
- B3: {'role_background_benefit_minus_aggregate_background_benefit': -1, 'accuracy_ci95_unadjusted_exploratory': [-0.08, 0.05]}
- B4: {'role_background_benefit_minus_aggregate_background_benefit': 5, 'accuracy_ci95_unadjusted_exploratory': [-0.04, 0.14]}
- B5: {'role_background_benefit_minus_aggregate_background_benefit': 2, 'accuracy_ci95_unadjusted_exploratory': [-0.06, 0.11]}

## Interpretation / Decision

Conditional bundle effects are measured, not individual-module/semantic-role mechanisms. Mini100 cannot establish full1319 attribution. No automatic best-hybrid selection or full launch. Inspect paired records before interpreting; retain role-separation premise.
