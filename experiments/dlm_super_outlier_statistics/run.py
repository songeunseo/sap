#!/usr/bin/env python3
"""Freeze inputs, collect dense statistics, and run the exploratory analysis."""
import hashlib
import json
from pathlib import Path

from experiments.dlm_super_outlier_statistics.collect import ROOT, CAL, CANDIDATES, collect, sha

CAPACITY = Path("experiments/projection_capacity_allocation_65/capacity_curves_raw.json")


def freeze():
    path = ROOT / "config.json"
    config = {
        "status": "frozen_before_collection",
        "model": {"id": "GSAI-ML/LLaDA-8B-Base",
                  "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                  "dtype": "bfloat16"},
        "states": "historical frozen 8 sequences x 10 corruption timesteps",
        "super_channel": 3848,
        "scope": "dense statistics and exploratory association; no pruning allocation",
        "primary_targets": ["D_g(65)", "marginal D_g(65->70)/(0.05*N_g)"],
        "statistics": "general concentration; channel 3848 included/excluded; Linear read/write contribution",
        "bootstrap": {"seed": 20260910, "resamples": 2000,
                      "unit": "sequence and layer cluster"},
        "sources": {str(source): sha(source) for source in (CAL, CANDIDATES, CAPACITY)},
    }
    text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError("frozen config changed")
    if not path.exists():
        path.write_text(text)
    return config


def main():
    freeze()
    collect()
    from experiments.dlm_super_outlier_statistics.analyze import analyze
    analyze()
    print(json.dumps({"event": "experiment_complete", "report": str(ROOT / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
