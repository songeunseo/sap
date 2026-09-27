#!/usr/bin/env bash
set -uo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="$1"
shift
for method in "$@"; do
    /usr/bin/python3 -u -m experiments.dlm_allocation_baselines65.run "$method" > "experiments/dlm_allocation_baselines65/logs/$method.log" 2>&1
    result=$?
    if [ "$result" -ne 0 ]; then
        echo "$method failed (exit $result); see its log. Continuing the independent queue."
    fi
done
