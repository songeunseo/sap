#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
root=experiments/dlm_context_response50
exec 9>"$root/pipeline.lock"
flock -n 9
trap 'code=$?; echo "$code" > "$root/exit_code.txt"' EXIT
bash "$root/run.sh" collect
for method in uniform A AC; do
    bash "$root/run.sh" evaluate --method "$method"
done
bash "$root/run.sh" summarize
