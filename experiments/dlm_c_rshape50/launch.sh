#!/usr/bin/env bash
set -euo pipefail
D=/home/tmluser1/sap/experiments/dlm_c_rshape50
mkdir -p $D/output/logs
for arm in AplusR RShape; do
  CUDA_VISIBLE_DEVICES=$1 $D/run.sh worker --arm $arm > $D/output/logs/$arm.log 2>&1
done
