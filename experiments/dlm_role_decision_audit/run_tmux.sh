#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
mkdir -p experiments/dlm_role_decision_audit/logs /DATA/tmluser1/sap-dlm-role-decision-audit
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
case "${1:?dense|role_sparse|analyze}" in
  dense) export CUDA_VISIBLE_DEVICES=0; exec /usr/bin/python3 -u experiments/dlm_role_decision_audit/collect.py --background dense ;;
  role_sparse) export CUDA_VISIBLE_DEVICES=1; exec /usr/bin/python3 -u experiments/dlm_role_decision_audit/collect.py --background role_sparse ;;
  analyze) exec /usr/bin/python3 -u experiments/dlm_role_decision_audit/analyze.py ;;
esac
