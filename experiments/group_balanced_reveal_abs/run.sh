#!/usr/bin/env bash
set -euo pipefail

cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
root=experiments/group_balanced_reveal_abs

python3 -m experiments.group_balanced_reveal_abs.run preflight > "$root/logs/preflight.stdout.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.group_balanced_reveal_abs.run score-shard --block-start 0 --block-end 16 > "$root/logs/score_gpu0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python3 -m experiments.group_balanced_reveal_abs.run score-shard --block-start 16 --block-end 32 > "$root/logs/score_gpu1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
python3 -m experiments.group_balanced_reveal_abs.run merge > "$root/logs/merge.stdout.log" 2>&1
python3 -m experiments.group_balanced_reveal_abs.run freeze-diagnostics > "$root/logs/freeze_diagnostics.stdout.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.group_balanced_reveal_abs.run evaluate > "$root/logs/evaluate.stdout.log" 2>&1
python3 -m experiments.group_balanced_reveal_abs.run report > "$root/logs/report.stdout.log" 2>&1
