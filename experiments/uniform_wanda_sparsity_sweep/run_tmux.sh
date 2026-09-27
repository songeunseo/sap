#!/usr/bin/env bash
set -euo pipefail

cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
/usr/bin/python3 -u experiments/uniform_wanda_sparsity_sweep/run.py all \
  > experiments/uniform_wanda_sparsity_sweep/run.log 2>&1
