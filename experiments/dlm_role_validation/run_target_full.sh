#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 {target50|target75} GPU" >&2
  exit 2
fi
cd /home/tmluser1/sap
suite="$1"
gpu="$2"
if [[ "$suite" != "target50" && "$suite" != "target75" ]]; then
  echo "full target must be target50 or target75" >&2
  exit 2
fi
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p experiments/dlm_role_validation/logs
exec /usr/bin/python3 -u -m experiments.dlm_role_validation.run full \
  --suite "$suite" --force-full --role-only \
  >> "experiments/dlm_role_validation/logs/${suite}_full.log" 2>&1
