#!/usr/bin/env python3
"""Concise progress and whole-run ETA for the role proxy experiment."""
from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).parent


def main():
    receipt = ROOT / "collection_receipt.json"
    preparation = ROOT / "preparation.json"
    result = ROOT / "mini100_results.json"
    if result.exists():
        row = json.loads(result.read_text())
        print("complete")
        for item in row["ranking"]:
            print(f"{item['method']}: {item['correct']}/100 (source={item['source']})")
        return
    if preparation.exists():
        doc = json.loads(preparation.read_text())
        methods = doc["novel_representatives"]
        completed = []
        for method in methods:
            path = ROOT / "gsm8k" / f"{method}_100_predictions.receipt.json"
            if path.exists():
                completed.append(method)
        pending = [x for x in methods if x not in completed]
        print(f"mini evaluation: {len(completed)}/{len(methods)} methods complete")
        print("remaining:", ", ".join(pending) or "none")
        log = ROOT / "logs" / "run.log"
        progress = []
        if log.exists():
            normalized = log.read_text(errors="replace").replace("\r", "\n")
            # Only inspect output after the most recent completed method.  Without
            # this boundary, a newly started 0/100 method inherits the preceding
            # method's terminal 100/100 progress line.
            boundary = normalized.rfind('"event": "mini_method_complete"')
            current_log = normalized[boundary:] if boundary >= 0 else normalized
            progress = re.findall(r"Generating\.\.\.:\s+\d+%.*?(\d+)/100.*?([0-9.]+)s/it", current_log)
        if progress and pending:
            examples, seconds_per_example = int(progress[-1][0]), float(progress[-1][1])
            seconds = (100 - examples) * seconds_per_example + (len(pending) - 1) * 100 * seconds_per_example
            print(f"current: {pending[0]} {examples}/100, {seconds_per_example:.2f}s/example")
            print(f"whole-run ETA: about {seconds / 3600:.2f} hours (model reload overhead excluded)")
        else:
            print("whole-run ETA: about 35 minutes per remaining mini model, plus reload overhead")
        return
    if receipt.exists():
        print("collection complete; allocation preparation pending")
        return
    log = ROOT / "logs" / "run.log"
    if log.exists():
        states = 0
        for line in log.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") == "moment_state":
                states = max(states, int(row["state"]))
        print(f"role moment collection: {states}/80 states")
    else:
        print("not started")


if __name__ == "__main__":
    main()
