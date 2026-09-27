# Projection Capacity Follow-up @ 65%

## Existing-data diagnostic

- Projections/states: 224/80
- Reconstruction allocation exact pruned count: 4,536,008,704
- EIS+type control exact pruned count: 4,536,008,704
- Single-anchor beats reconstruction in both cross-fit directions: True

## Allocation comparison

| Allocation | Additive calibration damage | Projections changed vs oracle |
|---|---:|---:|
| Oracle | 0.2623835 | 0 |
| Reconstruction | 0.52533286 | 162 |
| EIS+type | 0.2828367 | 68 |

These additive single-projection costs are diagnostics, not jointly sparse model results.
