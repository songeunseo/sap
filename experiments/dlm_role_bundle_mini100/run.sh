#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ "$1" == worker ]]; then
  export CUDA_VISIBLE_DEVICES="${2:?GPU index required}"
  exec /usr/bin/python3 -u -m experiments.dlm_role_bundle_mini100.run worker --gpu "$2"
fi
exec /usr/bin/python3 -u -m experiments.dlm_role_bundle_mini100.run "$@"
