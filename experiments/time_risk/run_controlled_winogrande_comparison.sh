#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

model="GSAI-ML/LLaDA-8B-Base"
root="results/controlled_winogrande"
mkdir -p "$root"

completed() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -name 'results_*.json' -print -quit)" ]]
}

evaluate() {
    local name="$1" model_path="$2" output="$root/$1"
    completed "$output" && return
    script -q -e -c \
        "accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --log_samples --model llada_dist --batch_size 8 --model_args model_path=${model_path},cfg=0.0,is_check_greedy=False,mc_num=128 --output_path ${output}" \
        "$root/eval-${name}.log"
}

evaluate dense "$model"

for method in wanda sink sparsegpt sink_sgpt; do
    for percent in 25 50 75; do
        name="${method}-${percent}"
        output="$root/$name"
        checkpoint="/dev/shm/controlled-${name}"
        if ! completed "$output"; then
            if [[ ! -f "$checkpoint/model.safetensors.index.json" ]]; then
                rm -rf "$checkpoint"
                script -q -e -c \
                    "python main_llada.py --model ${model} --prune_method ${method} --sparsity_ratio 0.${percent} --sparsity_type unstructured --calib_dataset wikitext2 --nsamples 8 --seqlen 256 --seed 0 --skip_ppl --save_model ${checkpoint}" \
                    "$root/prune-${name}.log"
            fi
            evaluate "$name" "$checkpoint"
        fi
        rm -rf "$checkpoint"
    done
done
