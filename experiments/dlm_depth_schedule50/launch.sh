#!/usr/bin/env bash
# Sequential arms on one GPU, then CPU report. Stops on first failure.
set -euo pipefail
D=/home/tmluser1/sap/experiments/dlm_depth_schedule50
mkdir -p $D/output/logs
for arm in DIS EIS AmShape AmPerm1 AmPerm2 AmPerm3; do
  CUDA_VISIBLE_DEVICES=$1 $D/run.sh worker --arm $arm > $D/output/logs/$arm.log 2>&1
done
$D/run.sh report > $D/output/logs/report.log 2>&1
