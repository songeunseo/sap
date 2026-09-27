"""Phased CLI for the dual-role reconstruction allocation diagnostic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_dual_role_allocation.analyze import run_analysis
from experiments.dlm_dual_role_allocation.collect import collect_all, event
from experiments.dlm_dual_role_allocation.io import (
    atomic_write_json,
    load_frozen_inputs,
)

ROOT = Path("experiments/dlm_dual_role_allocation")


def _frozen_config(inputs) -> dict:
    return {
        "status": "frozen_before_role_collection",
        "model": inputs.config["model"],
        "dense_model_sha256": inputs.metadata["dense_model_sha256"],
        "state_digest": inputs.metadata["state_digest"],
        "source_receipt_sha256": inputs.receipt["receipt_sha256"],
        "source_files": inputs.receipt["files"],
        "projection_count": len(inputs.metadata["module_names"]),
        "state_count": inputs.metadata["state_count"],
        "grid": inputs.config["grid"],
        "uniform65_pruned": inputs.metadata["uniform65_pruned"],
        "total_prunable_weights": inputs.metadata["total_weights"],
        "selector": "existing 80-state Standard Wanda candidate mask family",
        "role_curve": "max(E_masked,E_unmasked)",
        "aggregate_curve": "(num_masked+num_unmasked)/(den_masked+den_unmasked)",
        "contrast": "abs(E_masked-E_unmasked)/(E_masked+E_unmasked); zero/zero=0",
        "monotone_envelope": None,
        "crossfit": {
            "directions": [
                {"construction_sequences": [0, 1, 2, 3],
                 "validation_sequences": [4, 5, 6, 7]},
                {"construction_sequences": [4, 5, 6, 7],
                 "validation_sequences": [0, 1, 2, 3]},
            ],
            "qualification": "fixed-mask-family conditional cross-validation",
        },
        "bootstrap": {"unit": "held-out sequence", "resamples": 20000,
                      "seed": 0, "interval": "percentile 95%"},
        "mask_xor_gate": 0.01,
        "reconstruction_limits": {"max_absolute": 1e-7, "max_relative": 1e-5},
        "nested_ols": {
            "baseline": "intercept + layer dummies + type dummies + aggregate marginal/parameter",
            "extended_addition": "role-minus-aggregate marginal/parameter",
            "reference_layer": 0,
            "reference_type": "attn_out",
            "standardization": "construction rows only",
        },
        "full_model_evaluation": False,
        "downstream_evaluation": False,
        "gpu_policy": "physical GPU 0 only; physical GPU 1 untouched",
    }


def freeze() -> dict:
    inputs = load_frozen_inputs()
    config = _frozen_config(inputs)
    config_path = ROOT / "config.json"
    verification_path = ROOT / "state_verification.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise RuntimeError("frozen experiment config differs from current inputs")
    else:
        atomic_write_json(config_path, config)
    verification = {
        "status": inputs.verification["status"],
        "disjoint": inputs.verification["disjoint"],
        "source_receipt_sha256": inputs.receipt["receipt_sha256"],
        "state_digest": inputs.metadata["state_digest"],
        "state_count": inputs.metadata["state_count"],
        "sequence_indices": inputs.metadata.get("sequence_indices"),
        "timesteps": inputs.metadata.get("timesteps"),
        "source_verification": inputs.verification,
    }
    if verification_path.exists():
        existing = json.loads(verification_path.read_text())
        if existing != verification:
            raise RuntimeError("frozen state verification differs from current inputs")
    else:
        atomic_write_json(verification_path, verification)
    event("freeze_complete", projections=config["projection_count"],
          states=config["state_count"], uniform65_pruned=config["uniform65_pruned"],
          source_receipt_sha256=config["source_receipt_sha256"])
    return config


def _require_physical_gpu_zero() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError("collection requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly physical GPU 0 must be visible for collection")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "collect", "analyze", "all"))
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--smoke-output", type=Path)
    args = parser.parse_args(argv)
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if args.smoke_output is not None and args.stop_after is None:
        parser.error("--smoke-output requires --stop-after")
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    freeze()
    if args.phase == "freeze":
        return
    if args.phase in ("collect", "all"):
        _require_physical_gpu_zero()
        destination = args.smoke_output if args.smoke_output is not None else ROOT
        collect_all(destination, stop_after=args.stop_after)
        if args.stop_after is not None:
            event("smoke_collection_complete", destination=str(destination),
                  projections=args.stop_after)
            return
    if args.phase in ("analyze", "all"):
        run_analysis(ROOT)


if __name__ == "__main__":
    main()
