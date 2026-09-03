#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

runner=(python -m experiments.dlm_loss_aggregation.exp005.run)
"${runner[@]}" preflight
"${runner[@]}" partition
"${runner[@]}" score
"${runner[@]}" evaluate
"${runner[@]}" analyze
