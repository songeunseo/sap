#!/usr/bin/env bash
set -euo pipefail
cd /home/tmluser1/sap
while /usr/bin/python3 experiments/projection_capacity_followup_65/queue_control.py projection_capacity_followup65; do
  sleep 30
done
/usr/bin/python3 -c 'import json; from pathlib import Path; p=Path("experiments/projection_capacity_followup_65/heldout_dlm_results.json"); assert p.exists() and json.loads(p.read_text())["status"] == "complete"'
exec bash /home/tmluser1/sap/experiments/projection_capacity_followup_65/run_downstream_tmux.sh
