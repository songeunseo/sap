import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from experiments.cgq_wanda_diagnostic.run import (
    CONFIG,
    EXPECTED_DIGEST,
    EXPECTED_RUN,
    _collect_confidence,
    _model_sha256,
    _module_map,
    _token_metadata,
    _validate_sources,
)
from experiments.cgq_wanda_structured_diagnostic.core import (
    aggregate_by_timestep,
    attention_head_scores,
    mlp_neuron_scores,
    ratio_deviation_attribution,
    select_components,
    structured_statistics,
)
from experiments.dlm_loss_aggregation.run import _load_model, _module_specs


ROOT = Path(__file__).parent
STATISTICS_PATH = ROOT / "sufficient_statistics.pt"
MANIFEST_PATH = ROOT / "artifact_manifest.json"


class StateStatisticAccumulator:
    def __init__(self, columns):
        self.columns = columns
        self.uniform = []
        self.cgq = []

    def add(self, inp, factors):
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        if inp.ndim != 3 or inp.shape[0] != 1 or factors.shape != inp.shape[:2]:
            raise ValueError("structured diagnostic expects one [1, token, feature] state")
        flat = inp.reshape(-1, inp.shape[-1]).float()
        factor = factors.reshape(-1).to(device=flat.device, dtype=torch.float32)
        square = flat.square()
        self.uniform.append(square.sum(dim=0).cpu())
        self.cgq.append((square * factor.square().unsqueeze(1)).sum(dim=0).cpu())

    def finalize(self, timestep_indices):
        uniform = torch.stack(self.uniform)
        cgq = torch.stack(self.cgq)
        return {
            "overall_uniform": uniform.mean(dim=0),
            "overall_cgq": cgq.mean(dim=0),
            "timestep_uniform": aggregate_by_timestep(uniform, timestep_indices) / 8,
            "timestep_cgq": aggregate_by_timestep(cgq, timestep_indices) / 8,
        }


@torch.no_grad()
def collect_sufficient_statistics(model, manifest, factors):
    modules = _module_map(model)
    accumulators = {
        name: StateStatisticAccumulator(layer.weight.shape[1]) for name, layer in modules.items()
    }
    current = {"factors": None}
    handles = []
    for name, layer in modules.items():
        def hook(_, inp, __, module_name=name):
            accumulators[module_name].add(inp[0].data, current["factors"])
        handles.append(layer.register_forward_hook(hook))
    device = model.model.transformer.wte.weight.device
    try:
        for index, state in enumerate(manifest["states"]):
            current["factors"] = factors[index:index + 1]
            ids = torch.tensor(state["noisy_ids"], dtype=torch.long, device=device)
            model(ids)
            print(f"statistics {index + 1}/80", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    timestep_indices = torch.tensor([state["timestep_index"] for state in manifest["states"]])
    return modules, {
        name: accumulator.finalize(timestep_indices) for name, accumulator in accumulators.items()
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_modules(mapping, layer):
    prefix = f"block_{layer:02d}."
    return {name.rsplit(".", 1)[-1]: module for name, module in mapping.items() if name.startswith(prefix)}


def _condition_statistics(statistics, layer, condition):
    prefix = f"block_{layer:02d}."
    key = f"overall_{condition}"
    return {name.rsplit(".", 1)[-1]: values[key] for name, values in statistics.items() if name.startswith(prefix)}


def _attribution(groups):
    components = [name for name in groups["uniform"] if name != "total"]
    uniform_total = groups["uniform"]["total"]
    cgq_total = groups["cgq"]["total"]
    delta_total = cgq_total - uniform_total
    result = {}
    for name in components:
        delta = groups["cgq"][name] - groups["uniform"][name]
        result[name] = {
            "uniform_fraction": groups["uniform"][name] / uniform_total,
            "cgq_fraction": groups["cgq"][name] / cgq_total,
            "delta_fraction": delta / delta_total,
            "component_ratio": groups["cgq"][name] / groups["uniform"][name],
        }
    return result


def _summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    q = torch.quantile(values, torch.tensor([.1, .5, .9], dtype=torch.float64))
    return {"mean": float(values.mean()), "p10": float(q[0]), "p50": float(q[1]),
            "p90": float(q[2]), "min": float(values.min()), "max": float(values.max())}


def _attribution_summary(attribution):
    return {component: {key: _summary(value) for key, value in metrics.items()}
            for component, metrics in attribution.items()}


def _extreme_attributions(groups, attribution, count=10):
    ratio = groups["cgq"]["total"] / groups["uniform"]["total"]
    deviation = (ratio - ratio.mean()).abs()
    selected = torch.topk(deviation, min(count, ratio.numel())).indices
    deviation_attribution = ratio_deviation_attribution(groups)
    rows = []
    for unit in selected.tolist():
        row = {"unit": unit, "group_ratio": float(ratio[unit]), "deviation_from_layer_mean": float(ratio[unit] - ratio.mean())}
        row["components"] = {
            component: {**{key: float(value[unit]) for key, value in metrics.items()},
                        "ratio_deviation_contribution": float(deviation_attribution[component][unit])}
            for component, metrics in attribution.items()
        }
        rows.append(row)
    return rows


def _deviation_attribution_summary(groups):
    values = ratio_deviation_attribution(groups)
    magnitudes = {name: value.abs().mean() for name, value in values.items()}
    total = sum(magnitudes.values())
    return {name: {"mean_absolute_contribution": float(value),
                   "fraction_of_component_absolute_contributions": float(value / total)}
            for name, value in magnitudes.items()}


def _previous_feature_rows():
    document = json.loads(Path("experiments/cgq_wanda_diagnostic/diagnostics.json").read_text())
    return {(row["layer"], row["module_type"]): row for row in document["modules"]}


def analyze(model, statistics):
    internal = model.model.config
    previous = _previous_feature_rows()
    mlp_layers, attention_layers = [], []
    extreme = {"mlp": [], "attention": []}
    for layer in range(32):
        modules = _layer_modules(_module_map(model), layer)
        uniform = _condition_statistics(statistics, layer, "uniform")
        cgq = _condition_statistics(statistics, layer, "cgq")
        mlp_names = ("ff_proj", "up_proj", "ff_out")
        attention_names = ("q_proj", "k_proj", "v_proj", "attn_out")
        mlp = mlp_neuron_scores(
            {name: modules[name].weight for name in mlp_names},
            select_components(uniform, mlp_names), select_components(cgq, mlp_names),
        )
        attention = attention_head_scores(
            {name: modules[name].weight for name in attention_names},
            select_components(uniform, attention_names), select_components(cgq, attention_names),
            internal.n_heads, internal.d_model // internal.n_heads,
        )
        mlp_attribution = _attribution(mlp)
        attention_attribution = _attribution(attention)
        mlp_stats = structured_statistics(mlp["uniform"]["total"], mlp["cgq"]["total"], fractions=(.05, .10, .20))
        attention_stats = structured_statistics(attention["uniform"]["total"], attention["cgq"]["total"], counts=(4, 8))
        ff_out_cv = previous[(layer, "ff_out")]["ratio"]["cv"]
        attn_out_cv = previous[(layer, "attn_out")]["ratio"]["cv"]
        qkv_cv = float(np.mean([previous[(layer, name)]["ratio"]["cv"] for name in ("q_proj", "k_proj", "v_proj")]))
        mlp_layers.append({"layer": layer, **mlp_stats,
                           "feature_reference": {"ff_out_cv": ff_out_cv, "structured_to_ff_out_cv": mlp_stats["ratio"]["cv"] / ff_out_cv},
                           "component_attribution": _attribution_summary(mlp_attribution),
                           "ratio_deviation_attribution": _deviation_attribution_summary(mlp)})
        attention_layers.append({"layer": layer, **attention_stats,
                                 "feature_reference": {"attn_out_cv": attn_out_cv, "qkv_mean_cv": qkv_cv,
                                                       "structured_to_attn_out_cv": attention_stats["ratio"]["cv"] / attn_out_cv,
                                                       "structured_to_qkv_cv": attention_stats["ratio"]["cv"] / qkv_cv},
                                 "component_attribution": _attribution_summary(attention_attribution),
                                 "ratio_deviation_attribution": _deviation_attribution_summary(attention)})
        extreme["mlp"].extend({"layer": layer, **row} for row in _extreme_attributions(mlp, mlp_attribution))
        extreme["attention"].extend({"layer": layer, **row} for row in _extreme_attributions(attention, attention_attribution))
    extreme["mlp"] = sorted(extreme["mlp"], key=lambda row: abs(row["deviation_from_layer_mean"]), reverse=True)[:20]
    extreme["attention"] = sorted(extreme["attention"], key=lambda row: abs(row["deviation_from_layer_mean"]), reverse=True)[:20]
    return mlp_layers, attention_layers, extreme


def _family_summary(rows):
    keys = {
        "spearman": [row["spearman"] for row in rows],
        "kendall_tau": [row["kendall_tau"] for row in rows],
        "ratio_cv": [row["ratio"]["cv"] for row in rows],
        "normalized_mean_rank_displacement": [row["rank_displacement"]["mean_absolute_normalized"] for row in rows],
    }
    return {name: _summary(values) for name, values in keys.items()}


def main(reuse_statistics=False):
    started = time.time()
    config, manifest, partition = _validate_sources()
    model, devices = _load_model(config)
    internal = model.model.config
    specs = _module_specs(model)
    if len(specs) != 224 or internal.n_heads != internal.effective_n_kv_heads:
        raise RuntimeError("expected 224 matrices and an unambiguous equal-head attention mapping")
    weight_before = _model_sha256(model)
    ROOT.mkdir(parents=True, exist_ok=True)
    if reuse_statistics:
        if not STATISTICS_PATH.exists():
            raise RuntimeError("cannot reuse missing sufficient statistics")
        confidence_error = 0.0
        weight_after_collection = weight_before
    else:
        confidence, confidence_error = _collect_confidence(model, manifest, partition)
        _, _, _, factors = _token_metadata(manifest, partition, confidence)
        _, statistics = collect_sufficient_statistics(model, manifest, factors)
        weight_after_collection = _model_sha256(model)
        if weight_before != weight_after_collection:
            raise RuntimeError("model weights changed during activation collection")
        payload = {
            "version": 1,
            "source_state_sha256": EXPECTED_DIGEST,
            "normalization": {"overall": "mean over 80 batch examples", "timestep": "mean over 8 batch examples"},
            "statistics": statistics,
        }
        torch.save(payload, STATISTICS_PATH)
    loaded = torch.load(STATISTICS_PATH, map_location="cpu", weights_only=True)
    if loaded["source_state_sha256"] != EXPECTED_DIGEST or len(loaded["statistics"]) != 224:
        raise RuntimeError("saved sufficient-statistics readback failed")
    mlp, attention, extremes = analyze(model, loaded["statistics"])
    weight_after_analysis = _model_sha256(model)
    if weight_before != weight_after_analysis:
        raise RuntimeError("model weights changed during structured analysis")
    architecture = {
        "block_type": type(model.model.transformer.blocks[0]).__name__, "blocks": len(model.model.transformer.blocks),
        "d_model": internal.d_model, "mlp_hidden_size": internal.mlp_hidden_size,
        "q_heads": internal.n_heads, "k_heads": internal.effective_n_kv_heads,
        "v_heads": internal.effective_n_kv_heads, "head_dim": internal.d_model // internal.n_heads,
        "parameter_shapes": {spec["module_type"]: spec["shape"] for spec in specs[:7]},
        "mlp_forward": "ff_out(silu(ff_proj(ff_norm(x))) * up_proj(ff_norm(x)))",
        "attention_layout": "q/k/v outputs reshape [B,T,32,128] -> [B,32,T,128]; concatenated head outputs [B,T,4096] feed attn_out",
    }
    report = {
        "status": "complete", "analysis_only": True, "pruning_performed": False,
        "step10_timestep_analysis_performed": False,
        "source": {"run_id": EXPECTED_RUN, "state_sha256": EXPECTED_DIGEST, "state_count": 80,
                   "model": config["model"], "previous_diagnostic": "experiments/cgq_wanda_diagnostic/diagnostics.json"},
        "architecture": architecture,
        "group_score": {"primary": "raw additive sum of abs(W_ij)*sqrt(A_j) over all parameter slices in the functional unit",
                        "mlp": "ff_proj[k,:] union up_proj[k,:] union ff_out[:,k]",
                        "attention": "q_proj[h*128:(h+1)*128,:] union k_proj[...] union v_proj[...] union attn_out[:,h*128:(h+1)*128]"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda_runtime": torch.version.cuda, "devices": list(devices), "gpu": torch.cuda.get_device_name(devices[0])},
        "integrity": {"confidence_reproduction_max_abs_error": confidence_error,
                      "weight_sha256_before": weight_before, "weight_sha256_after_collection": weight_after_collection,
                      "weight_sha256_after_analysis": weight_after_analysis,
                      "statistics_path": str(STATISTICS_PATH), "statistics_sha256": _sha256(STATISTICS_PATH)},
        "mlp": {"summary": _family_summary(mlp), "layers": mlp},
        "attention": {"summary": _family_summary(attention), "layers": attention},
        "extreme_unit_attribution": extremes,
        "runtime_seconds": time.time() - started,
    }
    MANIFEST_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "manifest": str(MANIFEST_PATH),
                      "statistics": str(STATISTICS_PATH), "seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-statistics", action="store_true")
    arguments = parser.parse_args()
    main(reuse_statistics=arguments.reuse_statistics)
