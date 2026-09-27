#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
mkdir -p experiments/dlm_role_proxy_aggregation_65/logs
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
/usr/bin/python3 -u experiments/dlm_role_proxy_aggregation_65/run.py all 2>&1 | tee -a experiments/dlm_role_proxy_aggregation_65/logs/run.log
