#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=3
exec /usr/bin/python3 -u -m experiments.dlm_reveal_kl_mini100.run "$@"
