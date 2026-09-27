#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p experiments/dlm_dsa_search65/logs
/usr/bin/python3 -u -m experiments.dlm_dsa_search65.run run >> experiments/dlm_dsa_search65/logs/search.log 2>&1
