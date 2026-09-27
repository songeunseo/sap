#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
mkdir -p experiments/dlm_role_causal_decomposition/logs
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
/usr/bin/python3 -u experiments/dlm_role_causal_decomposition/run.py collect 2>&1 | tee -a experiments/dlm_role_causal_decomposition/logs/collect.log
/usr/bin/python3 -u experiments/dlm_role_causal_decomposition/analyze.py 2>&1 | tee -a experiments/dlm_role_causal_decomposition/logs/analyze.log
