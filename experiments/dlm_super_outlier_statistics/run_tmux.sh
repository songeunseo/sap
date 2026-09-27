#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p experiments/dlm_super_outlier_statistics/logs
exec /usr/bin/python3 -u -m experiments.dlm_super_outlier_statistics.run \
  >> experiments/dlm_super_outlier_statistics/logs/run.log 2>&1
