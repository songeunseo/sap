#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /home/tmluser1/sap
exec /usr/bin/python3 -B -u writing/dlm_pruning/closeout.py "$@"
