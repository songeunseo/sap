#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
mkdir -p experiments/dlm_role_validation/logs
while tmux has-session -t role_random65 2>/dev/null; do
  sleep 30
done
/home/tmluser1/sap/experiments/dlm_role_validation/run_suite.sh target50 0
/home/tmluser1/sap/experiments/dlm_role_validation/run_suite.sh target75 0
