#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
root=experiments/dlm_role_exchange_prediction_v2
case "$1" in
  prepare) exec /usr/bin/python3 -u -m experiments.dlm_role_exchange_prediction_v2.prepare ;;
  development|final)
    export CUDA_VISIBLE_DEVICES="${2:?GPU index required}"
    exec /usr/bin/python3 -u -m experiments.dlm_role_exchange_prediction_v2.collect --split "$1" ;;
  analyze) exec /usr/bin/python3 -u -m experiments.dlm_role_exchange_prediction_v2.analyze ;;
  queue)
    trap 'touch experiments/dlm_role_exchange_prediction_v2/logs/queue_failed' ERR
    bash "$root/run.sh" development 0 > "$root/logs/development.log" 2>&1 & dev_pid=$!
    bash "$root/run.sh" final 1 > "$root/logs/final.log" 2>&1 & final_pid=$!
    wait "$dev_pid"
    wait "$final_pid"
    bash "$root/run.sh" analyze > "$root/logs/analyze.log" 2>&1
    ;;
  *) exit 2 ;;
esac
