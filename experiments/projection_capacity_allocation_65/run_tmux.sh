#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
/usr/bin/python3 -u experiments/projection_capacity_allocation_65/run.py all \
  > experiments/projection_capacity_allocation_65/logs/run.log 2>&1
