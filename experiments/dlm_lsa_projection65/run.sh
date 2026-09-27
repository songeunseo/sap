#!/usr/bin/env bash
set -uo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export TOKENIZERS_PARALLELISM=false
mkdir -p experiments/dlm_lsa_projection65/logs
/usr/bin/python3 -m experiments.dlm_lsa_projection65.run run > experiments/dlm_lsa_projection65/logs/run.log 2>&1
result=$?
echo "exit=$result" >> experiments/dlm_lsa_projection65/logs/run.log
exit "$result"
