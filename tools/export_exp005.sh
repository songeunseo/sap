#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != *:* ]]; then
  echo "usage: $0 user@server:/destination/sap" >&2
  exit 2
fi

destination="$1"
remote="${destination%%:*}"
remote_dir="${destination#*:}"
root="$(git rev-parse --show-toplevel)"
cd "$root"

ssh "$remote" "mkdir -p '$remote_dir'"
git archive HEAD | ssh "$remote" "tar -xf - -C '$remote_dir'"

rsync -aP \
  experiments/dlm_loss_aggregation/exp005/ \
  "$destination/experiments/dlm_loss_aggregation/exp005/"
rsync -aP \
  experiments/dlm_loss_aggregation/calibration_manifest.json \
  experiments/dlm_loss_aggregation/scores \
  experiments/dlm_loss_aggregation/masks \
  "$destination/experiments/dlm_loss_aggregation/"
rsync -aP \
  experiments/dlm_loss_aggregation/exp002/results/predictions/dlm_abs.jsonl \
  "$destination/experiments/dlm_loss_aggregation/exp002/results/predictions/"

echo "EXP-005 export complete: $destination"
