#!/usr/bin/env python3
"""Dependency-free live status for the frozen DLM PPL50 experiment."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1] / "experiments/dlm_ppl50"
METHODS = ("uniform", "owl", "dlp", "alpha", "lsa_layer", "lsa_projection", "dsa", "evopress")
LABELS = {"uniform": "Uniform-Wanda", "owl": "OWL", "dlp": "DLP", "alpha": "AlphaPruning", "lsa_layer": "LSA Layer", "lsa_projection": "LSA Projection", "dsa": "DSA", "evopress": "EvoPress"}


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def show():
    print("DLM PPL50 | " + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"{'Method':<16} {'Stage':<25} {'Progress':<24} {'ETA':<12} {'NELBO':>9} {'PPL bound':>10}")
    print("-" * 102)
    for method in METHODS:
        row = read(ROOT / method / "progress.json")
        result = read(ROOT / method / "validation/results.json")
        stage = row.get("stage", "queued")
        progress = "-"
        eta = "-"
        summary = result.get("summary", row.get("summary", {}))
        if result:
            stage = "complete"
            progress = f"{summary['blocks']}/{summary['blocks']} chunks"
        elif row:
            pid = row.get("pid")
            if pid and stage not in ("complete", "failed"):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    stage += " (stopped)"
                except PermissionError:
                    pass
            if stage.startswith("evaluating_validation"):
                done, total = row.get("completed", 0), row.get("total", 1)
                progress = f"{done}/{total} ({100*done/total:.1f}%)"
            elif "block" in row:
                progress = f"block {row['block']+1}/32"
                if "total" in row:
                    progress += f"; {row.get('completed',0)}/{row['total']}"
            elif "generation" in row:
                progress = f"gen {row['generation']}/{row.get('generations','?')}"
                if "selection_stage" in row:
                    progress += f"; stage {row['selection_stage']}"
            elif "total" in row:
                progress = f"{row.get('completed',0)}/{row['total']}"
            if "remaining_seconds" in row:
                eta = duration(row['remaining_seconds'])
        nelbo = f"{summary['token_nelbo']:.6f}" if summary else "-"
        ppl = f"{summary['ppl_upper_bound_estimate']:.6f}" if summary else "-"
        print(f"{LABELS[method]:<16} {stage:<25} {progress:<24} {eta:<12} {nelbo:>9} {ppl:>10}")
        if row.get("error"):
            print("  Error: " + row['error'])
    print("\nETA is for the current stage. NELBO/PPL appear after full validation completes.")
    print("Logs: " + str(ROOT / "logs"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Refresh every 5 seconds; Ctrl+C to exit")
    args = parser.parse_args()
    try:
        while True:
            if args.watch:
                print("\033[2J\033[H", end="", flush=True)
            show()
            if not args.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print()
