#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec /usr/bin/python3 -u -m experiments.dlm_context_response50.full_run "$@"
