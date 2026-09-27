#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
/usr/bin/python3 -m experiments.dlm_scale_shape50.run run >> experiments/dlm_scale_shape50/run.log 2>&1
