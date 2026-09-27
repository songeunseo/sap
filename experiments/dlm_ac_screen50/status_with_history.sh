#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh status
exec /usr/bin/python3 -B /home/tmluser1/sap/experiments/dlm_ac_screen50/status_history.py
