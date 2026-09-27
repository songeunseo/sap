#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap

export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p experiments/dlm_role_wanda25_full/logs
log=experiments/dlm_role_wanda25_full/logs/pipeline.log

/usr/bin/python3 -u -m experiments.dlm_role_wanda25_full.run prepare >> "$log" 2>&1
/usr/bin/python3 -u -m experiments.dlm_role_wanda25_full.run evaluate --method role >> "$log" 2>&1
/usr/bin/python3 -u -m experiments.dlm_role_wanda25_full.run evaluate --method uniform >> "$log" 2>&1
/usr/bin/python3 -u -m experiments.dlm_role_wanda25_full.run finalize >> "$log" 2>&1
