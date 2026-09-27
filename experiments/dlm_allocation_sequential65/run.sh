#!/usr/bin/env bash
set -uo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export TOKENIZERS_PARALLELISM=false
mkdir -p experiments/dlm_allocation_sequential65/logs
for method in "$@"; do
    /usr/bin/python3 -m experiments.dlm_allocation_sequential65.run run --method "$method" > "experiments/dlm_allocation_sequential65/logs/$method.log" 2>&1
    result=$?
    if [ "$result" -ne 0 ]; then
        echo "$method failed with exit $result; log preserved" >&2
    fi
done
/usr/bin/python3 -m experiments.dlm_allocation_sequential65.run summarize
