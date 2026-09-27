#!/usr/bin/env bash
# prune on GPU1 (includes EXP-002 mask reproduction gate), then two eval shards on GPU1/GPU3, then report
set -euo pipefail
D=/home/tmluser1/sap/experiments/dlm_wanda_sapcalib50_full
mkdir -p $D/output/logs
CUDA_VISIBLE_DEVICES=1 $D/run.sh prune 2>&1 | tee -a $D/output/logs/prune.log
CUDA_VISIBLE_DEVICES=1 $D/run.sh eval --shard 0 > $D/output/logs/shard0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=3 $D/run.sh eval --shard 1 > $D/output/logs/shard1.log 2>&1 &
P1=$!
wait $P0; wait $P1
CUDA_VISIBLE_DEVICES= $D/run.sh report 2>&1 | tee -a $D/output/logs/report.log
