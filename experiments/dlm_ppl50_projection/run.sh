#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
method="$1"
gpu="$2"
case "$method" in owl_projection|dsa_projection) ;; *) exit 2;; esac
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export TOKENIZERS_PARALLELISM=false
mkdir -p experiments/dlm_ppl50_projection/logs
log="experiments/dlm_ppl50_projection/logs/${method}.log"
echo "Waiting for GPU ${gpu} to become idle" >> "$log"
while nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")"; do
    sleep 30
done
echo "Starting on GPU ${gpu} at $(date -Is)" >> "$log"
export CUDA_VISIBLE_DEVICES="$gpu"
/usr/bin/python3 -m experiments.dlm_ppl50_projection.run run --method "$method" >> "$log" 2>&1
