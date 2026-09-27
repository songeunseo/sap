#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
mkdir -p experiments/dlm_role_global_minimax_mini100/logs
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
exec /usr/bin/python3 -u experiments/dlm_role_global_minimax_mini100/run.py all
