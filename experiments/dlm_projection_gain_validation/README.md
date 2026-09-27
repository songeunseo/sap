Projection-wise Standard Wanda square curve with signed CE50 gain calibration.

Frozen calibration and historical row masks are preserved. Six projection curves,
six joint blocks, and three adjacent pairs are physically measured in the dense
model; results must pass dense CE reproduction and exact weight restoration.

Run inside tmux: `bash experiments/dlm_projection_gain_validation/run.sh collect`.
After completion: use the same PYTHONPATH in run.sh and run the verify module.
Read `progress.json`, `report.md`, and `verification.json` for status/results.
No allocation or downstream benchmark is launched by this diagnostic.
