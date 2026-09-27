#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
coverage_root=experiments/dlm_support_coverage50
exec 9>"$coverage_root/pipeline.lock"
flock -n 9
trap 'code=$?; echo "$code" > "$coverage_root/exit_code.txt"' EXIT
bash "$coverage_root/run.sh" prepare
bash "$coverage_root/run.sh" baseline
for coverage_method in pooled coverage; do
    bash "$coverage_root/run.sh" evaluate --method "$coverage_method"
done
bash "$coverage_root/run.sh" summarize
