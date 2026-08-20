#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

evaluate() {
    local percent="$1"
    script -q -e -c \
        "accelerate launch eval_llada.py --tasks gsm8k --limit 256 --log_samples --model llada_dist --model_args model_path=/dev/shm/mean-dlm-${percent},gen_length=1024,steps=1024,block_length=1024 --output_path results/mean_dlm/gsm8k-mean${percent}-256" \
        "results/mean_dlm/gsm8k-mean${percent}-256.log"
}

evaluate 50
rm -rf /dev/shm/mean-dlm-50

script -q -e -c \
    "python dlm_gradient_sensitivity.py materialize-direct --config experiments/time_risk/pilot.json --sparsity 0.25 --output-dir /dev/shm/mean-dlm-25 --output results/mean_dlm/pruning-25.json" \
    results/mean_dlm/materialize-25.log

evaluate 25
rm -rf /dev/shm/mean-dlm-25
