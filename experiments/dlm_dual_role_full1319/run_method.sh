#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 METHOD GPU" >&2
  exit 2
fi
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES="$2"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p experiments/dlm_dual_role_full1319/logs
exec /usr/bin/python3 -u -m experiments.dlm_dual_role_full1319.run evaluate --method "$1" \
  >> "experiments/dlm_dual_role_full1319/logs/$1.log" 2>&1
