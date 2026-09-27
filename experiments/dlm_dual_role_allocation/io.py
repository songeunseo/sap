"""Frozen input and checkpoint validation for dual-role reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.projection_capacity_allocation_65.core import GRID

MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
EXPECTED_PROJECTIONS = 224
EXPECTED_STATES = 80
EXPECTED_SEQUENCES = set(range(8))
EXPECTED_TIMESTEPS = {round(.05 + .10 * index, 2) for index in range(10)}
EXPECTED_PRUNED = 4_536_008_704
EXPECTED_WEIGHTS = 6_979_321_856


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_receipt(paths: list[str | Path]) -> dict:
    files = {str(Path(path)): file_sha256(path) for path in paths}
    receipt = {"files": files}
    receipt["receipt_sha256"] = json_sha256(receipt)
    return receipt


def validate_frozen_payloads(config: dict, verification: dict, candidate: dict,
                             curves: dict, states: dict, *,
                             expected_projection_count: int = EXPECTED_PROJECTIONS,
                             expected_state_count: int = EXPECTED_STATES,
                             expected_sequences: set[int] = EXPECTED_SEQUENCES,
                             expected_timesteps: set[float] = EXPECTED_TIMESTEPS) -> dict:
    """Validate cross-artifact identities without accessing the filesystem."""
    model = config.get("model", {})
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise RuntimeError("frozen model identity mismatch")
    if verification.get("status") != "verified" or verification.get("disjoint") is not True:
        raise RuntimeError("allocation and held-out state disjointness is not verified")
    grid = [float(value) for value in config.get("grid", [])]
    if grid != list(GRID) or [float(value) for value in curves.get("grid", [])] != list(GRID):
        raise RuntimeError("frozen sparsity grid mismatch")

    state_rows = states.get("states", [])
    if len(state_rows) != expected_state_count:
        raise RuntimeError("frozen state count mismatch")
    sequences = {int(row["sequence_index"]) for row in state_rows}
    timesteps = {round(float(row["timestep"]), 8) for row in state_rows}
    if sequences != set(expected_sequences) or timesteps != {round(float(x), 8) for x in expected_timesteps}:
        raise RuntimeError("frozen sequence or timestep levels mismatch")
    if len({(int(row["sequence_index"]), round(float(row["timestep"]), 8))
            for row in state_rows}) != expected_state_count:
        raise RuntimeError("duplicate sequence/timestep state")

    entries = candidate.get("entries", [])
    projections = curves.get("projections", [])
    if (len(entries) != expected_projection_count or
            len(projections) != expected_projection_count or
            int(config.get("projection_count", expected_projection_count)) != expected_projection_count):
        raise RuntimeError("projection count mismatch")
    module_names = []
    for index, (entry, projection) in enumerate(zip(entries, projections)):
        left = (entry.get("module_index"), entry.get("name"), entry.get("shape"))
        right = (projection.get("module_index", index), projection.get("name"), projection.get("shape"))
        if left != (index, entry.get("name"), entry.get("shape")) or left != right:
            raise RuntimeError("module ordering or shape mismatch")
        shape = [int(value) for value in entry["shape"]]
        if len(shape) != 2 or int(entry["weights"]) != shape[0] * shape[1]:
            raise RuntimeError("module weight count mismatch")
        masks = entry.get("masks", [])
        projection_curves = projection.get("curves", [])
        if len(masks) != 6 or len(projection_curves) != 6:
            raise RuntimeError("six candidate levels required")
        for level, (mask, curve, expected_level) in enumerate(zip(masks, projection_curves, GRID)):
            if (float(mask.get("nominal_sparsity", -1)) != expected_level or
                    float(curve.get("sparsity", -1)) != expected_level):
                raise RuntimeError("candidate level ordering mismatch")
            expected_count = shape[0] * int(shape[1] * expected_level)
            if int(mask.get("pruned", -1)) != expected_count:
                raise RuntimeError("candidate row-floor count mismatch")
            for field in ("mask_sha256", "path", "file_sha256"):
                if not mask.get(field):
                    raise RuntimeError(f"candidate mask missing {field}")
        module_names.append(entry["name"])

    total_weights = sum(int(row["weights"]) for row in entries)
    uniform65_pruned = sum(int(row["masks"][3]["pruned"]) for row in entries)
    budget = config.get("budget", {})
    if (total_weights != int(budget.get("total_weights", -1)) or
            uniform65_pruned != int(budget.get("total_pruned", -1))):
        raise RuntimeError("uniform row-floor budget mismatch")
    return {
        "module_names": module_names,
        "state_count": len(state_rows),
        "sequence_indices": sorted(sequences),
        "timesteps": sorted(timesteps),
        "uniform65_pruned": uniform65_pruned,
        "total_weights": total_weights,
        "state_digest": states.get("historical_state_sha256"),
        "dense_model_sha256": config.get("dense_model_sha256"),
    }


@dataclass(frozen=True)
class FrozenInputs:
    config: dict
    verification: dict
    candidate: dict
    curves: dict
    states: dict
    metadata: dict
    receipt: dict


def load_frozen_inputs(source_root: str | Path = "experiments/projection_capacity_allocation_65") -> FrozenInputs:
    source_root = Path(source_root)
    config_path = source_root / "config.json"
    verification_path = source_root / "state_verification.json"
    candidate_path = source_root / "candidate_mask_manifest.json"
    curves_path = source_root / "capacity_curves_raw.json"
    config = json.loads(config_path.read_text())
    verification = json.loads(verification_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    curves = json.loads(curves_path.read_text())

    calibration_splits = [row for row in verification["splits"] if int(row.get("states", 0)) == 80]
    if len(calibration_splits) != 1:
        raise RuntimeError("cannot identify unique allocation calibration split")
    state_path = Path(calibration_splits[0]["path"])
    if file_sha256(state_path) != calibration_splits[0]["file_sha256"]:
        raise RuntimeError("calibration state file hash mismatch")
    states = json.loads(state_path.read_text())
    if states.get("historical_state_sha256") != calibration_splits[0].get("state_sha256"):
        raise RuntimeError("calibration state digest mismatch")
    metadata = validate_frozen_payloads(config, verification, candidate, curves, states)

    paths = [config_path, verification_path, candidate_path, curves_path, state_path]
    for entry in candidate["entries"]:
        for mask in entry["masks"]:
            path = Path(mask["path"])
            if file_sha256(path) != mask["file_sha256"]:
                raise RuntimeError(f"candidate mask file hash mismatch: {path}")
    receipt = source_receipt(paths)
    return FrozenInputs(config, verification, candidate, curves, states, metadata, receipt)


def validate_checkpoint(payload: dict, expected: dict) -> None:
    labels = {
        "source_receipt_sha256": "source receipt",
        "dense_model_sha256": "dense model",
        "state_digest": "state digest",
        "module_index": "module index",
        "name": "module name",
        "shape": "module shape",
        "grid": "sparsity grid",
        "mask_sha256": "candidate mask hashes",
    }
    for field, label in labels.items():
        if payload.get(field) != expected.get(field):
            raise RuntimeError(f"checkpoint {label} mismatch")
    states = payload.get("states", [])
    if len(states) != int(expected["state_count"]):
        raise RuntimeError("checkpoint state count mismatch")
    if any(len(row.get("levels", [])) != int(expected["level_count"]) for row in states):
        raise RuntimeError("checkpoint level count mismatch")
