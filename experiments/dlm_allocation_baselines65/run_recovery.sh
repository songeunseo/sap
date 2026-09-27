#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
export PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/DATA/tmluser1/sap_storage_recovery_20260914/tmp
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES="$1"
exec /usr/bin/python3 -u -m experiments.dlm_allocation_baselines65.recovery supervise "$2"
