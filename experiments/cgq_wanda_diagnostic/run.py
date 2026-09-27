import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from experiments.cgq_wanda_diagnostic.core import (
    ActivationAccumulator,
    cgq_factors,
    feature_comparison,
    masked_confidence_deciles,
)
from experiments.dlm_loss_aggregation.run import (
    _load_model,
    _module_specs,
    historical_state_digest,
    load_config,
    require_historical_digest,
    validate_config,
)
from experiments.dlm_loss_aggregation.exp004.run import validate_partition_artifact
from lib.prune_llada import find_layers


ROOT = Path(__file__).parent
MANIFEST = Path("experiments/dlm_loss_aggregation/calibration_manifest.json")
PARTITION = Path("experiments/dlm_loss_aggregation/exp004/token_partition_summary.json")
CONFIG = Path("experiments/dlm_loss_aggregation/config.yaml")
EXPECTED_RUN = "20260828T175010-2484545"
EXPECTED_DIGEST = "1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df"


def _json(path):
    return json.loads(Path(path).read_text())


def _model_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _quantiles(values, probabilities=(0.01, 0.1, 0.5, 0.9, 0.99)):
    values = np.asarray(values, dtype=np.float64)
    return {f"p{int(p * 100):02d}": float(np.quantile(values, p)) for p in probabilities}


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()),
        **_quantiles(values), "max": float(values.max()),
    }


def _correlation(left, right, method="spearman"):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    result = spearmanr(left, right) if method == "spearman" else pearsonr(left, right)
    return float(result.statistic)


def _validate_sources():
    config = validate_config(load_config(CONFIG))
    manifest = _json(MANIFEST)
    digest = require_historical_digest(manifest, EXPECTED_DIGEST)
    if digest != historical_state_digest(manifest) or len(manifest["states"]) != 80:
        raise RuntimeError("frozen EXP-001 states failed validation")
    partition = _json(PARTITION)
    validate_partition_artifact(partition, manifest, expected_count=80)
    if config["model"] != manifest["model"] or manifest["mask_id"] != 126336:
        raise RuntimeError("model or mask token differs from frozen protocol")
    return config, manifest, partition


@torch.no_grad()
def _collect_confidence(model, manifest, partition):
    device = model.model.transformer.wte.weight.device
    rows = []
    maximum_error = 0.0
    for index, (state, saved) in enumerate(zip(manifest["states"], partition["states"])):
        ids = torch.tensor(state["noisy_ids"], dtype=torch.long, device=device)
        logits = model(ids).logits
        confidence = F.softmax(logits, dim=-1).amax(dim=-1).float().cpu()
        masked_positions = [row["position"] for row in saved["masked_predictions"]]
        expected = torch.tensor([row["confidence"] for row in saved["masked_predictions"]])
        actual = confidence[0, masked_positions]
        maximum_error = max(maximum_error, float((actual - expected).abs().max()))
        rows.append(confidence)
        print(f"confidence {index + 1}/80", flush=True)
        del logits, ids
    confidence = torch.cat(rows, dim=0)
    if maximum_error > 1e-6:
        raise RuntimeError(f"recomputed masked confidence differs from EXP-004: {maximum_error}")
    return confidence, maximum_error


def _module_map(model):
    result = {}
    for block_index, block in enumerate(model.model.transformer.blocks):
        for name, layer in find_layers(block).items():
            result[f"block_{block_index:02d}.{name}"] = layer
    return result


@torch.no_grad()
def _collect_activations(model, manifest, factors):
    modules = _module_map(model)
    accumulators = {
        name: ActivationAccumulator(layer.weight.shape[1]) for name, layer in modules.items()
    }
    current = {"factors": None}
    handles = []
    for name, layer in modules.items():
        def hook(_, inp, __, module_name=name):
            accumulators[module_name].add_batch(inp[0].data, current["factors"])
        handles.append(layer.register_forward_hook(hook))
    device = model.model.transformer.wte.weight.device
    try:
        for index, state in enumerate(manifest["states"]):
            current["factors"] = factors[index:index + 1]
            ids = torch.tensor(state["noisy_ids"], dtype=torch.long, device=device)
            model(ids)
            print(f"activation {index + 1}/80", flush=True)
            del ids
    finally:
        for handle in handles:
            handle.remove()
    return modules, accumulators


def _token_metadata(manifest, partition, confidence):
    masked = torch.tensor(np.concatenate([np.asarray(s["mask"], dtype=bool) for s in manifest["states"]]))
    reveal = torch.zeros_like(masked)
    for index, state in enumerate(partition["states"]):
        reveal[index, state["reveal_indices"]] = True
    timestep = torch.tensor([s["timestep"] for s in manifest["states"]], dtype=torch.float64)
    return masked, reveal, timestep, cgq_factors(confidence, masked)


def _corr_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
            "count": int(values.size)}


def _token_analysis(accumulators, confidence, factors, masked, reveal, timesteps):
    confidence_flat = confidence.flatten().numpy()
    factors_flat = factors.flatten().numpy()
    masked_flat = masked.flatten().numpy()
    reveal_flat = reveal.flatten().numpy()
    module_norms = {name: acc.token_norms for name, acc in accumulators.items()}
    all_norms = torch.stack(list(module_norms.values())).numpy()
    pooled_conf = _correlation(np.tile(confidence_flat, len(module_norms)), all_norms.reshape(-1))
    pooled_factor = _correlation(np.tile(factors_flat, len(module_norms)), all_norms.reshape(-1))
    state_conf, state_factor = [], []
    for norms in all_norms:
        for state_index in range(80):
            state_conf.append(_correlation(confidence[state_index], norms[state_index]))
            state_factor.append(_correlation(factors[state_index], norms[state_index]))
    timestep_rows = []
    for timestep in sorted(set(timesteps.tolist())):
        selected = timesteps.eq(timestep).numpy()
        norms = all_norms[:, selected, :].reshape(-1)
        timestep_rows.append({
            "timestep": timestep,
            "confidence_spearman": _correlation(np.tile(confidence[selected].reshape(-1), len(module_norms)), norms),
            "factor_spearman": _correlation(np.tile(factors[selected].reshape(-1), len(module_norms)), norms),
        })
    flat_norms = all_norms.reshape(-1)
    tiled_mask = np.tile(masked_flat, len(module_norms))
    tiled_reveal = np.tile(reveal_flat, len(module_norms))
    distributions = {
        "masked": _distribution(flat_norms[tiled_mask]),
        "unmasked": _distribution(flat_norms[~tiled_mask]),
        "reveal_masked": _distribution(flat_norms[tiled_reveal]),
        "remain_masked": _distribution(flat_norms[tiled_mask & ~tiled_reveal]),
    }
    masked_conf = confidence_flat[masked_flat]
    q20, q80 = np.quantile(masked_conf, [.2, .8])
    tiled_conf = np.tile(confidence_flat, len(module_norms))
    distributions["low_confidence_masked_q20"] = _distribution(flat_norms[tiled_mask & (tiled_conf <= q20)])
    distributions["high_confidence_masked_q80"] = _distribution(flat_norms[tiled_mask & (tiled_conf >= q80)])
    return {
        "pooled": {"confidence_spearman": pooled_conf, "factor_spearman": pooled_factor},
        "state_wise": {"confidence_spearman": _corr_summary(state_conf), "factor_spearman": _corr_summary(state_factor)},
        "timestep_wise": timestep_rows,
        "activation_norm_distributions": distributions,
        "confidence_bins": {"low_max_q20": float(q20), "high_min_q80": float(q80)},
    }, all_norms


def _energy_analysis(all_norms, confidence, factors, masked):
    energy_by_token = torch.from_numpy(all_norms).double().square().sum(dim=0)
    weighted = energy_by_token * factors.double().square()
    masked_energy = energy_by_token[masked]
    masked_weighted = weighted[masked]
    mean_norm_by_token = torch.from_numpy(all_norms).double().mean(dim=0)
    deciles = masked_confidence_deciles(
        confidence[masked], masked_energy, masked_weighted, mean_norm_by_token[masked]
    )
    rows = []
    for label, selected in (("masked", masked), ("unmasked", ~masked)):
        rows.append({
            "group": label, "token_count": int(selected.sum()),
            "uniform_energy_fraction": float(energy_by_token[selected].sum() / energy_by_token.sum()),
            "cgq_energy_fraction": float(weighted[selected].sum() / weighted.sum()),
            "mean_confidence": float(confidence[selected].mean()),
            "mean_factor": float(factors[selected].mean()),
        })
    return deciles, rows


def _score_rank_diagnostic(weight, uniform, cgq, row_count=128):
    selected = torch.linspace(0, weight.shape[0] - 1, min(row_count, weight.shape[0])).round().long()
    sampled_weight = weight.detach()[selected.to(weight.device)].abs().float().cpu()
    left = sampled_weight * uniform.sqrt().unsqueeze(0)
    right = sampled_weight * cgq.sqrt().unsqueeze(0)
    columns = left.shape[1]
    ordinal = torch.arange(columns).expand(left.shape[0], -1)
    left_rank = torch.empty_like(ordinal)
    right_rank = torch.empty_like(ordinal)
    left_rank.scatter_(1, torch.argsort(left, dim=1, stable=True), ordinal)
    right_rank.scatter_(1, torch.argsort(right, dim=1, stable=True), ordinal)
    centered_left = left_rank.double() - left_rank.double().mean(dim=1, keepdim=True)
    centered_right = right_rank.double() - right_rank.double().mean(dim=1, keepdim=True)
    correlations = (centered_left * centered_right).sum(dim=1) / (
        centered_left.norm(dim=1) * centered_right.norm(dim=1)
    )
    displacement = (left_rank - right_rank).abs().float().numpy()
    return {"sampled_rows": selected.tolist(), "row_count": selected.numel(),
            "row_spearman": _distribution(correlations.numpy()),
            "absolute_rank_displacement": _distribution(displacement)}


def _module_analysis(modules, accumulators):
    rows, score_rows = [], []
    representative = {"block_00", "block_15", "block_31"}
    for name, accumulator in accumulators.items():
        block_text, module = name.split(".", 1)
        layer = int(block_text.split("_")[1])
        module_type = module.rsplit(".", 1)[-1]
        compared = feature_comparison(accumulator.uniform, accumulator.cgq)
        rows.append({"layer": layer, "module": module, "module_type": module_type,
                     "features": accumulator.uniform.numel(), **compared})
        if block_text in representative and module_type == "attn_out":
            score_rows.append({"layer": layer, "module": module, "module_type": module_type,
                               **_score_rank_diagnostic(modules[name].weight, accumulator.uniform, accumulator.cgq)})
    return rows, score_rows


def _aggregate_modules(rows):
    groups = {}
    for row in rows:
        band = "early" if row["layer"] <= 10 else "mid" if row["layer"] <= 21 else "late"
        for key in (("module_type", row["module_type"]), ("layer", str(row["layer"])), ("layer_band", band)):
            groups.setdefault(key, []).append(row)
    result = []
    for (scope, value), group in sorted(groups.items()):
        result.append({"scope": scope, "value": value, "matrix_count": len(group),
                       "mean_spearman": float(np.mean([r["spearman"] for r in group])),
                       "mean_ratio_cv": float(np.mean([r["ratio"]["cv"] for r in group])),
                       "mean_relative_l2_difference": float(np.mean([r["relative_l2_difference"] for r in group])),
                       "max_ratio_cv": float(max(r["ratio"]["cv"] for r in group))})
    return result


def main():
    started = time.time()
    config, manifest, partition = _validate_sources()
    model, devices = _load_model(config)
    specs = _module_specs(model)
    if len(specs) != 224:
        raise RuntimeError(f"expected 224 Linear matrices, found {len(specs)}")
    before = _model_sha256(model)
    confidence, masked_error = _collect_confidence(model, manifest, partition)
    masked, reveal, timesteps, factors = _token_metadata(manifest, partition, confidence)
    modules, accumulators = _collect_activations(model, manifest, factors)
    after = _model_sha256(model)
    if before != after:
        raise RuntimeError("dense model weights changed during diagnostic")
    token_analysis, all_norms = _token_analysis(accumulators, confidence, factors, masked, reveal, timesteps)
    deciles, masked_unmasked = _energy_analysis(all_norms, confidence, factors, masked)
    module_rows, score_rows = _module_analysis(modules, accumulators)
    result = {
        "status": "complete", "analysis_only": True, "pruning_performed": False,
        "source": {"run_id": EXPECTED_RUN, "state_sha256": EXPECTED_DIGEST,
                   "state_count": 80, "sequence_length": 256, "matrix_count": 224,
                   "model": config["model"], "partition_sha256": partition["sha256"]},
        "confidence_coverage": {"exp004": "masked positions only", "diagnostic": "all positions recomputed",
                                "position_count": confidence.numel(), "masked_reproduction_max_abs_error": masked_error},
        "wanda_semantics": {"input_shape_per_hook": "[1, 256, in_features]", "tmp": "batch dimension = 1",
                            "nsamples": 80, "dtype_conversion": "input to float32 before squared norm",
                            "scaler_row": "(1/80) * sum_state sum_token X[state,token,j]^2",
                            "score": "abs(W[i,j]) * sqrt(scaler_row[j])"},
        "cgq": {"masked": "1 + sqrt(confidence)", "unmasked": "0.7 + sqrt(confidence)",
                "statistic": "(1/80) * sum_state sum_token r[state,token]^2 * X[state,token,j]^2",
                "factor_distribution": _distribution(factors.numpy()), "confidence_distribution": _distribution(confidence.numpy())},
        "token_analysis": token_analysis, "confidence_deciles_masked": deciles,
        "masked_unmasked_energy": masked_unmasked, "modules": module_rows,
        "localization": _aggregate_modules(module_rows),
        "representative_score_rank_diagnostics": score_rows,
        "runtime": {"seconds": time.time() - started, "cuda_devices": list(devices), "weight_sha256_before": before, "weight_sha256_after": after},
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(ROOT / "diagnostics.json"), "seconds": result["runtime"]["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
