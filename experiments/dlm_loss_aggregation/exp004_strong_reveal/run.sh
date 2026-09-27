#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

export PYTHONPATH="/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:$repo_root${PYTHONPATH:+:$PYTHONPATH}"

runner=(python3 -m experiments.dlm_loss_aggregation.exp004_strong_reveal.run)
"${runner[@]}" preflight
"${runner[@]}" score
"${runner[@]}" evaluate
"${runner[@]}" report
