#!/usr/bin/env bash
set -euo pipefail

cd /home/tmluser1/sap
while tmux has-session -t oracle_allocation75 2>/dev/null; do
  sleep 30
done

export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=0
/usr/bin/python3 -u experiments/oracle_allocation50/run.py all \
  > experiments/oracle_allocation50/run.log 2>&1
