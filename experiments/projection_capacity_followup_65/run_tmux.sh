#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
mkdir -p experiments/projection_capacity_followup_65/logs
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
/usr/bin/python3 experiments/projection_capacity_followup_65/run_heldout.py \
  > experiments/projection_capacity_followup_65/logs/heldout.log 2>&1
