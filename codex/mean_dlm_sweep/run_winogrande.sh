#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

materialize() {
    local percent="$1" sparsity="$2" command="${3:-materialize}"
    [[ -d "/dev/shm/mean-dlm-${percent}" ]] && return
    local artifact_args=()
    [[ "$command" == materialize ]] && artifact_args=(--artifact-dir codex/time_risk_sensitivity/results/masks)
    script -q -e -c \
        "python dlm_gradient_sensitivity.py ${command} --config codex/time_risk_sensitivity/config.json ${artifact_args[*]} --sparsity ${sparsity} --output-dir /dev/shm/mean-dlm-${percent} --output codex/mean_dlm_sweep/results/pruning-${percent}.json" \
        "codex/mean_dlm_sweep/results/materialize-winogrande-${percent}.log"
}

evaluate() {
    local percent="$1" suffix="$2" limit="${3:-}"
    local output="codex/mean_dlm_sweep/results/winogrande-mean${percent}-${suffix}"
    [[ ! -e "$output" && ! -e "${output}.log" ]]
    script -q -e -c \
        "accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 ${limit} --log_samples --model llada_dist --batch_size 8 --model_args model_path=/dev/shm/mean-dlm-${percent},cfg=0.0,is_check_greedy=False,mc_num=128 --output_path ${output}" \
        "${output}.log"
}

case "${1:-}" in
    pilot)
        materialize 50 0.50
        evaluate 50 pilot "--limit 16"
        ;;
    full)
        materialize 50 0.50
        evaluate 50 full
        rm -rf /dev/shm/mean-dlm-50

        materialize 25 0.25 materialize-direct
        evaluate 25 full
        rm -rf /dev/shm/mean-dlm-25

        materialize 75 0.75
        evaluate 75 full
        rm -rf /dev/shm/mean-dlm-75
        ;;
    *)
        echo "usage: $0 pilot|full" >&2
        exit 2
        ;;
esac
