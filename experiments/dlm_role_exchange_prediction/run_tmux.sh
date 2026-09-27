#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
mkdir -p experiments/dlm_role_exchange_prediction/logs /DATA/tmluser1/sap-dlm-role-exchange-prediction
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
case "${1:?development|final|analyze}" in
  development) export CUDA_VISIBLE_DEVICES=0; exec /usr/bin/python3 -u experiments/dlm_role_exchange_prediction/collect.py collect --split development ;;
  final) export CUDA_VISIBLE_DEVICES=1; exec /usr/bin/python3 -u experiments/dlm_role_exchange_prediction/collect.py collect --split final ;;
  analyze) exec /usr/bin/python3 -u experiments/dlm_role_exchange_prediction/analyze.py ;;
esac
