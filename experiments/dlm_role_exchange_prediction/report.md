# Role Exchange Prediction Validation

## Results

Rows: dev 30800, final 30800; bundle observations 5120.

| Model | MSE | Spearman | Sign accuracy |
|---|---:|---:|---:|
| P0_structure | 3.7098624e-05 | 0.2652 | 0.5751 |
| P1_pooled_dense | 3.4625705e-05 | 0.2681 | 0.5709 |
| P2_role_dense | 3.4661578e-05 | 0.2685 | 0.5711 |
| P3_role_dense_sparse | 3.7752283e-05 | 0.2613 | 0.5677 |
| P4_role_energy | 7.3253246e-05 | 0.2273 | 0.5637 |

## Primary comparisons

- H1_P2_minus_P1: error difference 3.5872796e-08, simultaneous CI [-1.302809830771009e-07, 2.151634330518559e-07], relative reduction -0.10%.
- H2_P3_minus_P2: error difference 3.0907053e-06, simultaneous CI [-1.3538388128655724e-07, 1.0408267304790857e-05], relative reduction -8.92%.
- role_minus_random: error difference 3.4727886e-08, simultaneous CI [-1.5243306833327944e-07, 2.2328120156694638e-07], relative reduction -0.10%.

## Gate

{"H1_pass": false, "H2_pass": false, "role_control_pass": false}

Both role reconstruction improved but KL worsened: 7275/16960.

## Bundle generalization

| Model | Bundle MSE | Spearman | Sign accuracy |
|---|---:|---:|---:|
| P0 structure | 1.13296e-4 | 0.0810 | 0.5363 |
| P1 pooled dense | **1.05380e-4** | 0.2416 | 0.5705 |
| P2 role dense | 1.06062e-4 | 0.2344 | 0.5713 |
| P3 role dense+sparse | 1.17178e-4 | 0.2580 | 0.5801 |
| P4 role energy | 2.24198e-4 | 0.2107 | 0.5754 |

Mean bundle nonadditivity was `+0.000238`. P2 did not improve bundle MSE over P1.

## Interpretation

- Pooled dense reconstruction contains limited predictive information: P1 reduced individual-exchange MSE by 6.67% relative to structure-only P0. This was exploratory rather than a preregistered primary comparison.
- Separating the same reconstruction statistic into masked/unmasked inputs added no out-of-document, out-of-layer predictive value. P2 was 0.10% worse than P1 and indistinguishable within the simultaneous interval.
- Actual masked/unmasked roles did not beat cardinality-matched random partitions. The observed role-specific stability therefore does not translate into general functional-damage prediction through this proxy.
- Sparse-background reconstruction did not repair the problem. P3 was 8.92% worse than P2, with an interval spanning zero; energy normalization degraded further.
- Role reconstruction cannot serve as a safety constraint: even when both measured role errors decreased, held-out KL worsened in 42.90% of cases.
- These results reject the tested reconstruction-based use of role separation. They do not prove that masked/unmasked roles are functionally identical or that the prior Role allocation's downstream result was chance.

## Decision

- Do not derive another aggregation rule from masked/unmasked reconstruction.
- Do not use role reconstruction as an allocation objective or safety constraint.
- Keep masked/unmasked only as a diagnostic partition until a different statistic demonstrates incremental prediction beyond pooled and random controls.
- The next method-development branch must target downstream propagation or sparse interaction directly; another local reconstruction transformation is not warranted by these results.
