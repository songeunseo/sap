"""Bounded dense-forward activation statistics for the frozen 80-state grid.

The saved records are sufficient statistics, not captured activations.  In
particular, ``token_sum`` exists only while one sequence is being collected so
that temporal residuals retain the identity of each token position.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_capacity_predictor import core
from experiments.dlm_loss_aggregation.run import historical_state_digest
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, load_dense
from experiments.wanda_failure_characterization.run_failure_map import model_sha, state_tensors


ROOT = Path("experiments/dlm_capacity_predictor")
CONFIG = ROOT / "config.json"
RUNTIME = ROOT / "runtime"
FINAL = ROOT / "dense_statistics.pt"
CANDIDATES = Path("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json")
SKETCH_SIZE = 64
SKETCH_SEED = 20260910


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _moments(samples):
    x = np.asarray(samples, dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    return {"sum": x.sum(0), "outer": x.T @ x, "count": int(len(x))}


def sequence_record(inputs, sketch):
    """Build one module/sequence record from [time, token, feature] inputs."""
    x = _array(inputs).astype(np.float64, copy=False)
    projection = _array(sketch).astype(np.float64, copy=False)
    if x.ndim != 3 or projection.shape[0] != x.shape[2]:
        raise ValueError("inputs/sketch shapes must be [time,token,feature]/[feature,sketch]")
    if x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty inputs required")
    token_mean = x.mean(0)
    residual = x - token_mean
    return {
        "timesteps": int(x.shape[0]),
        "feature_width": int(x.shape[2]),
        "energy_sum": float(np.square(x).sum()),
        "energy_count": int(x.shape[0] * x.shape[1]),
        "variation_energy_sum": float(np.square(residual).sum()),
        "feature_sq_sum": np.square(x).sum(1),
        "feature_count": np.full(x.shape[0], x.shape[1], dtype=np.int64),
        "mean_sketch": _moments(token_mean @ projection),
        "variation_sketch": _moments(residual @ projection),
    }


def _pool_moments(records, key):
    count = sum(int(r[key]["count"]) for r in records)
    total = sum((_array(r[key]["sum"]).astype(np.float64) for r in records))
    outer = sum((_array(r[key]["outer"]).astype(np.float64) for r in records))
    covariance = outer / count - np.outer(total / count, total / count)
    trace = float(np.trace(covariance))
    denominator = float(np.square(covariance).sum())
    return trace * trace / denominator if trace > 0 and denominator > 0 else None


def aggregate_sequences(sequence_records):
    """Pool raw sequence records without averaging per-sequence summaries."""
    records = list(sequence_records)
    if not records:
        raise ValueError("at least one sequence record required")
    timestep_count = int(records[0]["timesteps"])
    width = int(records[0]["feature_width"])
    if any(int(r["timesteps"]) != timestep_count or int(r["feature_width"]) != width
           for r in records):
        raise ValueError("records must have matching timestep and feature dimensions")
    energy = sum(float(r["energy_sum"]) for r in records)
    count = sum(int(r["energy_count"]) for r in records)
    if energy <= 0 or count <= 0:
        raise ValueError("zero input energy")
    variation = sum(float(r["variation_energy_sum"]) for r in records)
    feature_sum = sum((_array(r["feature_sq_sum"]).astype(np.float64) for r in records))
    feature_count = sum((_array(r["feature_count"]).astype(np.float64) for r in records))
    second = feature_sum / feature_count[:, None]
    totals = second.sum(1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("zero timestep feature energy")
    normalized = second / totals
    average = normalized.mean(0)
    denominator = float(np.square(average).sum())
    variability = float(np.square(normalized - average).sum(1).mean() / denominator)
    pr_mean = _pool_moments(records, "mean_sketch")
    pr_variation = _pool_moments(records, "variation_sketch")
    overall_second = feature_sum.sum(0) / feature_count.sum()
    outlier = float(np.mean(overall_second > 7 * overall_second.mean()))
    result = {
        "variation_ratio": variation / energy,
        "feature_use_variability": variability,
        "pr_mean": pr_mean,
        "pr_variation": pr_variation,
        "log_pr_ratio": (float(math.log(pr_variation / pr_mean))
                         if pr_mean and pr_variation else None),
        "activation_energy": energy / count,
        "activation_outlier_ratio": outlier,
    }
    return result


def _config_calibration_sha(config):
    if "calibration_sha256" in config:
        return config["calibration_sha256"]
    sources = config.get("sources", {})
    if str(core.CALIBRATION) in sources:
        return sources[str(core.CALIBRATION)]
    raise RuntimeError("frozen config must contain calibration_sha256")


def _fingerprint(config, manifest, names):
    expected_file = _config_calibration_sha(config)
    actual_file = core.sha(core.CALIBRATION)
    if expected_file != actual_file:
        raise RuntimeError("frozen calibration file hash mismatch")
    actual_states = historical_state_digest(manifest)
    if manifest.get("historical_state_sha256") != actual_states:
        raise RuntimeError("calibration historical-state digest mismatch")
    configured_states = config.get("historical_state_sha256", actual_states)
    if configured_states != actual_states:
        raise RuntimeError("frozen config historical-state digest mismatch")
    return {
        "config_sha256": core.sha(CONFIG),
        "calibration_sha256": actual_file,
        "historical_state_sha256": actual_states,
        "dense_model_sha256": DENSE_SHA,
        "module_names": list(names),
        "sketch_seed": SKETCH_SEED,
        "sketch_size": SKETCH_SIZE,
        "source_sha256": {
            str(core.CALIBRATION): actual_file,
            str(Path(__file__)): core.sha(Path(__file__)),
            str(ROOT / "core.py"): core.sha(ROOT / "core.py"),
            "experiments/projection_capacity_allocation_65/run.py": core.sha(
                "experiments/projection_capacity_allocation_65/run.py"),
            "experiments/wanda_failure_characterization/run_failure_map.py": core.sha(
                "experiments/wanda_failure_characterization/run_failure_map.py"),
        },
    }


def _validate_record(record, timesteps, sketch_size):
    required = {"timesteps", "feature_width", "energy_sum", "energy_count",
                "variation_energy_sum", "feature_sq_sum", "feature_count",
                "mean_sketch", "variation_sketch"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise RuntimeError("truncated dense sequence record")
    width = int(record["feature_width"])
    if int(record["timesteps"]) != timesteps or width <= 0:
        raise RuntimeError("invalid dense sequence dimensions")
    if _array(record["feature_sq_sum"]).shape != (timesteps, width):
        raise RuntimeError("invalid feature second-moment shape")
    if _array(record["feature_count"]).shape != (timesteps,):
        raise RuntimeError("invalid feature count shape")
    for key in ("mean_sketch", "variation_sketch"):
        moment = record[key]
        if (_array(moment.get("sum")).shape != (sketch_size,) or
                _array(moment.get("outer")).shape != (sketch_size, sketch_size) or
                int(moment.get("count", 0)) <= 0):
            raise RuntimeError("invalid sketch moment shape")


def _validate_sequence_payload(payload, fingerprint, sequence):
    names = list(fingerprint["module_names"])
    if payload.get("sequence_index") != sequence or payload.get("names") != names:
        raise RuntimeError("wrong sequence identity or module order")
    order = payload.get("state_order")
    if (not isinstance(order, list) or len(order) != 10 or
            [int(row.get("timestep_index", -1)) for row in order] != list(range(10))):
        raise RuntimeError("incomplete sequence timestep order")
    records = payload.get("records")
    # Forward hooks fire in computation order (q/k/v before attn_out), which is
    # intentionally different from the repository's pruning-target order.
    # Records are name-keyed; completeness, not dict insertion order, is the contract.
    if not isinstance(records, dict) or len(records) != len(names) or set(records) != set(names):
        raise RuntimeError("incomplete sequence module records")
    for record in records.values():
        _validate_record(record, 10, int(fingerprint["sketch_size"]))


def _validate_final_payload(payload, fingerprint):
    names = list(fingerprint["module_names"])
    if payload.get("config_sha256") != fingerprint["config_sha256"]:
        raise RuntimeError("final artifact config hash missing or changed")
    if payload.get("names") != names or payload.get("sequences") != list(range(8)):
        raise RuntimeError("incomplete final names or sequences")
    per_sequence = payload.get("per_sequence")
    if not isinstance(per_sequence, dict) or list(per_sequence) != [str(i) for i in range(8)]:
        raise RuntimeError("incomplete final per-sequence mapping")
    for records in per_sequence.values():
        if not isinstance(records, dict) or len(records) != len(names) or set(records) != set(names):
            raise RuntimeError("incomplete final module records")
        for record in records.values():
            _validate_record(record, 10, int(fingerprint["sketch_size"]))
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict) or list(aggregate) != names:
        raise RuntimeError("incomplete final aggregate mapping")
    aggregate_fields = set(core.FEATURES) | {
        "pr_mean", "pr_variation", "activation_energy", "activation_outlier_ratio"}
    for row in aggregate.values():
        if not isinstance(row, dict) or set(row) != aggregate_fields:
            raise RuntimeError("invalid final aggregate schema")
        for key, value in row.items():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise RuntimeError(f"invalid final aggregate value: {key}")


def _valid_payload(path, fingerprint, sequence=None):
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(f"existing artifact has a different fingerprint: {path}")
    if sequence is None:
        _validate_final_payload(payload, fingerprint)
    else:
        _validate_sequence_payload(payload, fingerprint, sequence)
    return payload


def _require_candidate_order(mapping, config):
    if not CANDIDATES.exists():
        raise RuntimeError("frozen candidate manifest is required")
    digest = core.sha(CANDIDATES)
    frozen_sources = config.get("source_sha256", config.get("sources", {}))
    configured = frozen_sources.get(str(CANDIDATES))
    if configured is None:
        raise RuntimeError("frozen config lacks candidate manifest hash")
    if configured != digest:
        raise RuntimeError("frozen candidate manifest hash mismatch")
    entries = json.loads(CANDIDATES.read_text()).get("entries", [])
    names = list(mapping)
    expected = [row.get("name") for row in entries]
    if len(entries) != 224 or expected != names:
        raise RuntimeError("dense module order differs from frozen candidate manifest")
    for index, (row, (name, module)) in enumerate(zip(entries, mapping.items())):
        if row.get("module_index") != index or row.get("shape") != list(module.weight.shape):
            raise RuntimeError(f"dense module shape/index differs from frozen candidate: {name}")
    return digest


def load_validated_statistics(path=FINAL):
    """Load a complete dense artifact and verify it against current frozen inputs.

    This intentionally does not load the model, so CPU analysis can validate the
    collector/analysis boundary without reserving GPU memory.
    """
    if not CONFIG.exists():
        raise RuntimeError("frozen config.json is required")
    config = json.loads(CONFIG.read_text())
    manifest = json.loads(core.CALIBRATION.read_text())
    if not CANDIDATES.exists():
        raise RuntimeError("frozen candidate manifest is required")
    candidate_digest = core.sha(CANDIDATES)
    frozen_sources = config.get("source_sha256", config.get("sources", {}))
    if frozen_sources.get(str(CANDIDATES)) != candidate_digest:
        raise RuntimeError("frozen candidate manifest hash mismatch")
    entries = json.loads(CANDIDATES.read_text()).get("entries", [])
    names = [row.get("name") for row in entries]
    if (len(names) != 224 or len(set(names)) != 224 or
            [row.get("module_index") for row in entries] != list(range(224))):
        raise RuntimeError("invalid frozen candidate names/order")
    fingerprint = _fingerprint(config, manifest, names)
    fingerprint["candidate_manifest_sha256"] = candidate_digest
    payload = _valid_payload(Path(path), fingerprint)
    if payload is None:
        raise FileNotFoundError(path)
    return payload


def _validation():
    rng = np.random.default_rng(20260910)
    fixtures = {
        "rank1": rng.normal(size=(4, 9, 1)) @ np.arange(1, 7)[None, None, :],
        "lowrank": rng.normal(size=(4, 9, 3)) @ rng.normal(size=(3, 12)),
    }
    result = {}
    for name, x in fixtures.items():
        sketch = core.sketch_matrix(x.shape[-1], SKETCH_SIZE, SKETCH_SEED)
        approximate = aggregate_sequences([sequence_record(x, sketch)])
        exact = core.temporal_statistics(x, np.eye(x.shape[-1]))
        result[name] = {
            "full_pr_mean": exact["pr_mean"], "sketch_pr_mean": approximate["pr_mean"],
            "pr_mean_absolute_error": abs(exact["pr_mean"] - approximate["pr_mean"]),
            "full_pr_variation": exact["pr_variation"],
            "sketch_pr_variation": approximate["pr_variation"],
            "pr_variation_absolute_error": abs(exact["pr_variation"] - approximate["pr_variation"]),
        }
    return result


@torch.inference_mode()
def _collect_one(model, mapping, states):
    names = list(mapping)
    live = {}
    sketches = {module.weight.shape[1]: core.sketch_matrix(
        module.weight.shape[1], SKETCH_SIZE, SKETCH_SEED).astype(np.float64)
        for module in mapping.values()}
    timestep_rows = []
    dev = next(model.parameters()).device
    for ti, state in enumerate(states):
        seen = set()
        handles = []
        for name, module in mapping.items():
            def hook(_, inp, __, module_name=name):
                if module_name in seen:
                    raise RuntimeError(f"module invoked more than once in state: {module_name}")
                seen.add(module_name)
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu().numpy().astype(np.float64)
                slot = live.setdefault(module_name, {
                    "sum_x": np.zeros_like(x), "energy_sum": 0.0,
                    "feature_sq": [], "counts": [], "sketch_sum": None,
                    "sketch_outer": None,
                })
                if slot["sum_x"].shape != x.shape:
                    raise RuntimeError("token positions or feature width changed within sequence")
                z = x @ sketches[x.shape[1]]
                slot["sum_x"] += x
                slot["energy_sum"] += float(np.square(x).sum())
                slot["feature_sq"].append(np.square(x).sum(0))
                slot["counts"].append(len(x))
                slot["sketch_sum"] = z.sum(0) if slot["sketch_sum"] is None else slot["sketch_sum"] + z.sum(0)
                zo = z.T @ z
                slot["sketch_outer"] = zo if slot["sketch_outer"] is None else slot["sketch_outer"] + zo
            handles.append(module.register_forward_hook(hook))
        try:
            noisy, _, _ = state_tensors(state, dev)
            model(noisy)
        finally:
            for handle in handles:
                handle.remove()
        if seen != set(names):
            raise RuntimeError("dense forward did not invoke every frozen Linear exactly once")
        timestep_rows.append({"timestep_index": int(state["timestep_index"]),
                              "timestep": float(state["timestep"])})

    total_t = len(states)
    records = {}
    for name, slot in live.items():
        token_mean = slot["sum_x"] / total_t
        sketch = sketches[token_mean.shape[1]]
        mean_z = token_mean @ sketch
        mean_outer = mean_z.T @ mean_z
        variation_outer = slot["sketch_outer"] - total_t * mean_outer
        variation_energy = slot["energy_sum"] - total_t * float(np.square(token_mean).sum())
        records[name] = {
            "timesteps": total_t, "feature_width": token_mean.shape[1],
            "energy_sum": slot["energy_sum"], "energy_count": total_t * len(token_mean),
            "variation_energy_sum": max(0.0, variation_energy),
            "feature_sq_sum": np.stack(slot["feature_sq"]),
            "feature_count": np.asarray(slot["counts"], dtype=np.int64),
            "mean_sketch": _moments(mean_z),
            "variation_sketch": {"sum": np.zeros(SKETCH_SIZE),
                                  "outer": variation_outer,
                                  "count": total_t * len(token_mean)},
        }
    return records, timestep_rows


def collect():
    if not CONFIG.exists():
        raise RuntimeError("frozen config.json is required before dense collection")
    config = json.loads(CONFIG.read_text())
    manifest = json.loads(core.CALIBRATION.read_text())
    states = manifest["states"]
    if len(states) != 80:
        raise RuntimeError("expected frozen 80-state calibration")
    keys = [(int(s["sequence_index"]), int(s["timestep_index"])) for s in states]
    if sorted(keys) != [(s, t) for s in range(8) for t in range(10)]:
        raise RuntimeError("expected exactly 8 sequences x 10 timesteps")

    load_started = time.monotonic()
    model, mapping = load_dense()
    load_seconds = time.monotonic() - load_started
    names = list(mapping)
    if len(names) != 224:
        raise RuntimeError("expected frozen 224-module order")
    candidate_sha256 = _require_candidate_order(mapping, config)
    fingerprint = _fingerprint(config, manifest, names)
    fingerprint["candidate_manifest_sha256"] = candidate_sha256
    existing = _valid_payload(FINAL, fingerprint)
    if existing is not None:
        if model_sha(model) != DENSE_SHA:
            raise RuntimeError("dense weights changed while validating final artifact")
        return existing

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    sequences = []
    per_sequence = {}
    for sequence in range(8):
        path = RUNTIME / f"dense_sequence_{sequence}.pt"
        cached = _valid_payload(path, fingerprint, sequence)
        if cached is None:
            selected = sorted((s for s in states if int(s["sequence_index"]) == sequence),
                              key=lambda s: int(s["timestep_index"]))
            records, timestep_rows = _collect_one(model, mapping, selected)
            cached = {"fingerprint": fingerprint, "sequence_index": sequence,
                      "names": names, "state_order": timestep_rows, "records": records}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(cached, temporary); temporary.replace(path)
        sequences.append(sequence)
        per_sequence[str(sequence)] = cached["records"]
        print(json.dumps({"event": "dense_sequence", "sequence": sequence}), flush=True)

    aggregate = {name: aggregate_sequences([per_sequence[str(s)][name] for s in sequences])
                 for name in names}
    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("dense weights changed during collection")
    payload = {
        "config_sha256": fingerprint["config_sha256"],
        "fingerprint": fingerprint, "names": names, "sequences": sequences,
        "per_sequence": per_sequence, "aggregate": aggregate,
        "timing": {"model_load_seconds": load_seconds,
                   "collection_seconds": time.monotonic() - started},
        "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated()
                                   if torch.cuda.is_available() else 0),
        "sketch_validation": _validation(),
        "cost_scope": {
            "dense_collection": "measured above",
            "historical_capacity_curves": "offline reuse; not a deployment collection cost",
            "oracle_training": "offline oracle cost; separate from deployment",
            "deployment_cost": "unmeasured",
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    temporary = FINAL.with_suffix(FINAL.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(FINAL)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = collect()
    print(json.dumps({"status": "complete", "path": str(FINAL),
                      "modules": len(result["names"]), "sequences": result["sequences"]}))


if __name__ == "__main__":
    main()
