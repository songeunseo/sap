#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

session=reveal_50x_finalize
log=experiments/reveal_50x_stress_test/logs/finalize.log
tmux new-session -d -s "$session" "python3 experiments/reveal_50x_stress_test/finalize.py > '$log' 2>&1"
echo "$session"
