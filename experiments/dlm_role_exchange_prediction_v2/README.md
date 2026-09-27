# Role exchange: corrective rerun

This is a corrective remeasurement of previously seen data, not a new confirmatory test.
The V1 outputs are preserved. Its broad recommendation to abandon role reconstruction
is on hold because measurement, regression and split-validation errors were found.

## Fixed corrections

- Physical mask changes and batch-one full model forward for every single and bundle.
  The same Role-A baseline is used for all deltas within a state.
- Independent smoke tests span seven projection types and multiple depths, comparing
  hooks to physical changes, separate role composition, restoration, and reused local features.
- Ridge minimizes equal-document/layer weighted **mean** squared error plus coefficient
  squared norm, alpha 1, with an unpenalized intercept and training-only scaling.
- The two overlapping development spans (sequence 8 and 10) are excluded after resolving
  actual parent articles. Development has 14 documents/70 states, final 16/80.
- Cluster bootstrap resamples documents and 4-layer blocks, with equal layer weights.
  Intervals are conditional on the fitted out-of-fold models (they do not include refitting).
- Reconstruction-versus-KL consistency is evaluated on exact-budget bundles. Individual
  weight restoration is not used to claim safety or unsafety of budget-constrained allocation.
- Retain masked/unmasked inputs. Additional nested pooled+contrast and role×time/type
  predictors are explicitly exploratory, not a new selection of a pruning method.
- Report stratum results, full OOF predictions, coefficient fits, and V1/V2 target changes.
  Inconclusive comparisons never imply that role separation is universally useless.

## Execution

From the repository root:

```bash
python3 experiments/dlm_role_exchange_prediction_v2/status.py
tmux attach -t role_exchange_v2
```

The queue runs development on GPU 0 and final on GPU 1. Each worker must pass its
smoke tests before collecting states. Analysis starts only after both workers exit
successfully and completion receipts exist. Per-state checkpoints are resumed only
when the frozen config identity matches. Code and source hashes are checked at startup.

Outputs: `analysis.json`, `decision.json`, `report.md`, `smoke_*.json`, and `logs/` here.
Large per-state artifacts and OOF predictions live at
`/DATA/tmluser1/sap-dlm-role-exchange-prediction-v2/`.

No GSM8K generation or new allocation is performed by this experiment.
