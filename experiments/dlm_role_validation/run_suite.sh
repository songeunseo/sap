#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 {random65|target50|target75} GPU" >&2
  exit 2
fi
cd /home/tmluser1/sap
suite="$1"
gpu="$2"
mkdir -p experiments/dlm_role_validation/locks
exec 9>"experiments/dlm_role_validation/locks/$suite.lock"
flock 9
if [[ -f "experiments/dlm_role_validation/mini_$suite.json" ]]; then
  echo "suite $suite already complete"
  exit 0
fi
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p experiments/dlm_role_validation/logs
exec /usr/bin/python3 -u -m experiments.dlm_role_validation.run screen --suite "$suite" \
  >> "experiments/dlm_role_validation/logs/$suite.log" 2>&1
