#!/usr/bin/env python3
import json
from pathlib import Path

import torch


ROOT = Path(__file__).parent
RESULTS = Path("/DATA/tmluser1/sap-dlm-role-causal-decomposition/results.pt")


def main():
    if (ROOT / "analysis.json").exists():
        row = json.loads((ROOT / "analysis.json").read_text())
        print("complete: collection 80/80 and analysis complete")
        print("actual-random contrast:", row["role_vs_random"]["actual_minus_random_mean"])
        return
    completed = 0
    if RESULTS.exists():
        completed = int(torch.load(RESULTS, map_location="cpu", weights_only=False)["completed_states"])
    print(f"causal collection: {completed}/80 states")
    log = ROOT / "logs" / "collect.log"
    if log.exists():
        events = []
        for line in log.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") == "state_complete":
                events.append(row)
        if events:
            latest = events[-1]
            done_this_run = max(1, latest["state"] - completed + len(events))
            seconds_per_state = latest["elapsed_seconds"] / max(1, len(events))
            print(f"observed {seconds_per_state:.1f}s/state; ETA about {(80-completed)*seconds_per_state/3600:.2f}h")


if __name__ == "__main__":
    main()
