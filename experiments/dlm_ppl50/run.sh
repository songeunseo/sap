#!/usr/bin/env bash
set -uo pipefail
cd /home/tmluser1/sap
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export TOKENIZERS_PARALLELISM=false
mkdir -p experiments/dlm_ppl50/logs
for method in uniform owl dlp alpha lsa_layer lsa_projection dsa evopress; do
    /usr/bin/python3 -m experiments.dlm_ppl50.run run --method "$method" > "experiments/dlm_ppl50/logs/$method.log" 2>&1
    result=$?
    /usr/bin/python3 -m experiments.dlm_ppl50.status  >> experiments/dlm_ppl50/logs/queue.log 2>&1
    if [ "$result" -ne 0 ]; then
        echo "$method failed with exit $result; continuing remaining candidates" >> experiments/dlm_ppl50/logs/queue.log
    fi
done
/usr/bin/python3 -m experiments.dlm_ppl50.status >> experiments/dlm_ppl50/logs/queue.log
