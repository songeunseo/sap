# Role-Wanda one-shot context refresh — mini-100

GPU0 sparse_context, GPU1 dense_target. Same frozen80 calibration states,
historical Standard Wanda masks/grid and exact65% row-floor weight budget.
Both use the historical dense role-energy denominator and max-level marginal greedy.
Initial model is frozen Role-Wanda65; exactly one reallocation, no iteration tuning.

Candidate1 compares W_s x_sparse with W_dense x_sparse.
Candidate2 compares W_s x_sparse with W_dense x_dense.
BF16 batch1 outputs, FP32 squared error, pooled role numerator/denominator ratios.
No learned KL predictor, gradient scoring, compensation or new ranking.

Before downstream, batch1 dense reconstruction must reproduce the historical Role allocation.
Failure is a measurement-control blocker, not a negative method result.
All candidate and dense-control curves remain raw; no monotonic envelope or clipping.

```bash
python3 experiments/dlm_role_context_mini100/status.py
```

Logs: `logs/sparse_context.log`, `logs/dense_target.log`, `logs/freeze.log`.
Each method saves per-state stats, curves, dense control, exact allocation trace,
mask manifest, predictions and paired results vs frozen Role24/100.
Completed collection and results use config hashes; original experiments are not overwritten.
If candidate mask exactly equals historical Role, reuse its validated predictions.
Mini-100 is a repeatedly used development set, not independent confirmation.
