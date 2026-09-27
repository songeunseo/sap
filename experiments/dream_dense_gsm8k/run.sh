#!/usr/bin/env bash
set -euo pipefail

cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export HF_HOME=/DATA/tmluser1/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p experiments/dream_dense_gsm8k/output/logs

export CUDA_VISIBLE_DEVICES=0
/usr/bin/python3 -u -m experiments.dream_dense_gsm8k.run evaluate --phase mini 2>&1 | tee experiments/dream_dense_gsm8k/output/logs/mini.log
/usr/bin/python3 -u -m experiments.dream_dense_gsm8k.run validate-mini 2>&1 | tee -a experiments/dream_dense_gsm8k/output/logs/mini.log

export CUDA_VISIBLE_DEVICES=0,1,2,3
/usr/bin/python3 -u -m accelerate.commands.launch --num_processes 4 --multi_gpu \
  -m experiments.dream_dense_gsm8k.run evaluate --phase full \
  2>&1 | tee experiments/dream_dense_gsm8k/output/logs/full.log
/usr/bin/python3 -u -m experiments.dream_dense_gsm8k.run finalize 2>&1 | tee experiments/dream_dense_gsm8k/output/logs/finalize.log
