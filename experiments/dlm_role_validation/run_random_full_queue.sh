#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
while tmux has-session -t role_target75 2>/dev/null; do
  sleep 30
done
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p experiments/dlm_role_validation/logs
exec /usr/bin/python3 -u -m experiments.dlm_role_validation.run full --suite random65 \
  >> experiments/dlm_role_validation/logs/random65_full.log 2>&1
