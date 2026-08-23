#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

model="GSAI-ML/LLaDA-8B-Base"
config="experiments/time_risk/calibration_16x512.json"
root="results/calibration_16x512"
mkdir -p "$root"

completed() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -name 'results_*.json' -print -quit)" ]]
}

checkpoint_complete() {
    [[ -f "$1/model.safetensors.index.json" ]]
}

evaluate() {
    local name="$1" checkpoint="$2" output="$root/$1"
    completed "$output" && return
    rm -rf "$output"
    script -q -e -c \
        "accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --log_samples --model llada_dist --batch_size 8 --model_args model_path=${checkpoint},cfg=0.0,is_check_greedy=False,mc_num=128 --output_path ${output}" \
        "$root/eval-${name}.log"
}

for method in wanda sparsegpt; do
    for percent in 25 50; do
        name="${method}-${percent}"
        checkpoint="/dev/shm/calibration-16x512-${name}"
        if ! completed "$root/$name"; then
            if ! checkpoint_complete "$checkpoint"; then
                rm -rf "$checkpoint"
                script -q -e -c \
                    "python main_llada.py --model ${model} --prune_method ${method} --sparsity_ratio 0.${percent} --sparsity_type unstructured --calib_dataset wikitext2 --nsamples 16 --seqlen 512 --seed 0 --skip_ppl --save_model ${checkpoint}" \
                    "$root/prune-${name}.log"
            fi
            evaluate "$name" "$checkpoint"
        fi
        rm -rf "$checkpoint"
    done
done

for percent in 25 50; do
    name="mean-${percent}"
    checkpoint="/dev/shm/calibration-16x512-${name}"
    if ! completed "$root/$name"; then
        if ! checkpoint_complete "$checkpoint"; then
            rm -rf "$checkpoint"
            script -q -e -c \
                "python dlm_gradient_sensitivity.py materialize-direct --config ${config} --sparsity 0.${percent} --output-dir ${checkpoint} --output ${root}/pruning-${name}.json" \
                "$root/prune-${name}.log"
        fi
        evaluate "$name" "$checkpoint"
    fi
    rm -rf "$checkpoint"
done
