#!/usr/bin/env bash
set -uo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
/usr/bin/python3 -m experiments.dlm_allocation_backtrace.analyze > experiments/dlm_allocation_backtrace/run.log 2>&1
result=$?
echo "exit=$result" >> experiments/dlm_allocation_backtrace/run.log
exit "$result"
