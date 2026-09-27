#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
exec 9>experiments/dlm_reveal_kl_mini100/pipeline.lock
flock -n 9
trap 'rc=$?; echo "$rc" > experiments/dlm_reveal_kl_mini100/exit_code.txt' EXIT
bash experiments/dlm_reveal_kl_mini100/run.sh collect
bash experiments/dlm_reveal_kl_mini100/run.sh evaluate --method reveal
bash experiments/dlm_reveal_kl_mini100/run.sh evaluate --method all_masked
bash experiments/dlm_reveal_kl_mini100/run.sh summarize
