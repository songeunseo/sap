import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

from experiments.dlm_loss_aggregation.run import _load_model, _suffix_logits, load_config, require_historical_digest, validate_config
from experiments.mlp_neuron_redundancy.core import (
    ablate_variant_rows,
    distribution_summary,
    fixed_candidates,
    masked_losses_and_kl,
    pairwise_stability,
    rank_predictability,
    sham_corrected_losses,
)


ROOT = Path(__file__).parent
CONFIG_PATH = Path("experiments/dlm_loss_aggregation/config.yaml")
STATE_MANIFEST_PATH = Path("experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json")
GRADIENT_PATH = Path("experiments/mlp_token_contrast_diagnostic/state_gate_derivatives.pt")
CANDIDATE_PATH = ROOT / "sampled_neurons.json"
STATE_RESULTS_PATH = ROOT / "state_ablation_results.pt"
MANIFEST_PATH = ROOT / "artifact_manifest.json"
LAYERS = (0, 4, 8, 12, 16, 20, 26, 31)
SEED = 20260905
STATE_SHA = "1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df"
BATCH_CANDIDATES = (1, 4, 8, 16)


def _atomic_json(path, document):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _official_loss(logits, clean, mask, p_mask):
    return F.cross_entropy(logits[mask].float(), clean[mask], reduction="sum") / p_mask / clean.numel()


def _capture_dense(model, noisy, candidate_map):
    prefixes, energies = {}, {}
    handles = []
    for layer in LAYERS:
        block = model.model.transformer.blocks[layer]
        def prefix_hook(_, inputs, layer_index=layer):
            prefixes[layer_index] = inputs[0].detach()
        handles.append(block.register_forward_pre_hook(prefix_hook))
        def energy_hook(_, inputs, __, layer_index=layer):
            selected = inputs[0][0, :, candidate_map[layer_index]].float()
            energies[layer_index] = selected.square().sum(dim=0).cpu()
        handles.append(block.ff_out.register_forward_hook(energy_hook))
    try:
        logits = model(noisy).logits
    finally:
        for handle in handles:
            handle.remove()
    return logits, prefixes, energies


def _variant_logits(model, prefix, layer, neurons):
    block = model.model.transformer.blocks[layer]
    current = [None] + list(neurons)
    repeated = prefix.expand(len(current), -1, -1).contiguous()
    handle = block.ff_out.register_forward_pre_hook(
        lambda _, inputs: (ablate_variant_rows(inputs[0], current),)
    )
    try:
        return _suffix_logits(model, repeated, start_block=layer)
    finally:
        handle.remove()


def _state_tensors(state, device):
    return (
        torch.tensor(state["noisy_ids"], dtype=torch.long, device=device),
        torch.tensor(state["clean_ids"], dtype=torch.long, device=device),
        torch.tensor(state["mask"], dtype=torch.bool, device=device),
    )


@torch.inference_mode()
def validate_and_benchmark(model, state, candidate_map):
    device = model.model.transformer.wte.weight.device
    noisy, clean, mask = _state_tensors(state, device)
    dense, prefixes, _ = _capture_dense(model, noisy, candidate_map)
    dense_loss = _official_loss(dense, clean, mask, state["p_mask"])
    repeated_dense = model(noisy).logits
    repeat_loss = _official_loss(repeated_dense, clean, mask, state["p_mask"])
    no_op = {"logit_max_abs": (dense.float() - repeated_dense.float()).abs().max().item(),
             "loss_abs": abs(dense_loss.item() - repeat_loss.item())}
    layer, neuron = LAYERS[0], candidate_map[LAYERS[0]][0]
    block = model.model.transformer.blocks[layer]
    captured = {}
    def verify_hook(_, inputs):
        original = inputs[0]
        changed = ablate_variant_rows(original, [neuron])
        actual = (
            F.linear(changed.float(), block.ff_out.weight.float(), None)
            - F.linear(original.float(), block.ff_out.weight.float(), None)
        )
        expected = -original[:, :, neuron].unsqueeze(-1).float() * block.ff_out.weight[:, neuron].float()
        captured["max_abs"] = (actual.float() - expected).abs().max().item()
        captured["relative"] = captured["max_abs"] / expected.abs().max().clamp_min(1e-30).item()
        captured["ablated_channel_max_abs"] = changed[:, :, neuron].abs().max().item()
        preserved = torch.ones(original.shape[-1], dtype=torch.bool, device=original.device)
        preserved[neuron] = False
        captured["other_channels_bitwise_equal"] = torch.equal(changed[:, :, preserved], original[:, :, preserved])
        return (changed,)
    handle = block.ff_out.register_forward_pre_hook(verify_hook)
    try:
        _suffix_logits(model, prefixes[layer].clone(), start_block=layer)
    finally:
        handle.remove()
    exact_logits = _variant_logits(model, prefixes[layer], layer, [neuron])
    exact_loss, exact_kl = masked_losses_and_kl(exact_logits, dense, clean, mask, state["p_mask"])
    _, exact_sham_kl = masked_losses_and_kl(exact_logits, exact_logits[:1], clean, mask, state["p_mask"])
    reference_delta = sham_corrected_losses(exact_loss)[0].item()
    reference_kl = exact_sham_kl[1].item()
    repeated_logits = _variant_logits(model, prefixes[layer], layer, [neuron])
    repeated_loss, repeated_kl = masked_losses_and_kl(repeated_logits, dense, clean, mask, state["p_mask"])
    deterministic = {"loss_max_abs": (exact_loss - repeated_loss).abs().max().item(),
                     "kl_max_abs": (exact_kl - repeated_kl).abs().max().item()}
    benchmark = []
    for count in BATCH_CANDIDATES:
        neurons = candidate_map[layer][:count]
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        logits = _variant_logits(model, prefixes[layer], layer, neurons)
        losses, kl = masked_losses_and_kl(logits, dense, clean, mask, state["p_mask"])
        _, sham_relative_kl = masked_losses_and_kl(logits, logits[:1], clean, mask, state["p_mask"])
        corrected = sham_corrected_losses(losses)
        elapsed = time.perf_counter() - started
        benchmark.append({"candidates_per_batch": count, "variants_including_sham": count + 1,
                          "seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "sham_delta_loss": losses[0].item() - dense_loss.item(), "sham_kl": kl[0].item(),
                          "first_candidate_raw_delta_loss": losses[1].item() - dense_loss.item(),
                          "first_candidate_sham_corrected_delta_loss": corrected[0].item(),
                          "first_candidate_corrected_delta_difference_vs_batch1": abs(corrected[0].item() - reference_delta),
                          "first_candidate_sham_relative_kl": sham_relative_kl[1].item(),
                          "first_candidate_corrected_kl_difference_vs_batch1": abs(sham_relative_kl[1].item() - reference_kl)})
        del logits, losses, kl
    max_loss_drift = max(row["first_candidate_corrected_delta_difference_vs_batch1"] for row in benchmark)
    safe_batches = [row["candidates_per_batch"] for row in benchmark
                    if row["first_candidate_corrected_delta_difference_vs_batch1"] <= 5e-5]
    if (no_op["loss_abs"] > 1e-7 or deterministic["loss_max_abs"] > 1e-7
            or captured["relative"] > 0.02 or captured["ablated_channel_max_abs"] != 0
            or not captured["other_channels_bitwise_equal"] or not safe_batches):
        evidence = {"dense_repeat": no_op, "intervention_identity": captured,
                    "repeated_ablation": deterministic, "batch_benchmark": benchmark,
                    "limits": {"no_op_loss": 1e-7, "repeat_loss": 1e-7,
                               "intervention_relative": 0.02, "batch_loss": 5e-5}}
        raise RuntimeError("intervention or batching sanity gate failed: " + json.dumps(evidence, sort_keys=True))
    return {"dense_repeat": no_op, "intervention_identity": captured,
            "repeated_ablation": deterministic, "batch_benchmark": benchmark,
            "selected_candidates_per_batch": max(safe_batches),
            "batch_loss_agreement_limit": 5e-5, "intervention_relative_limit": 0.02}


@torch.inference_mode()
def collect(model, manifest, candidate_map, batch_size):
    n_states, n_layers, n_candidates = 80, len(LAYERS), 32
    delta = torch.full((n_states, n_layers, n_candidates), torch.nan, dtype=torch.float32)
    raw_delta = torch.full_like(delta, torch.nan)
    kl = torch.full_like(delta, torch.nan)
    raw_kl = torch.full_like(delta, torch.nan)
    activation = torch.full_like(delta, torch.nan)
    sham_delta = torch.full((n_states, n_layers, math.ceil(n_candidates / batch_size)), torch.nan)
    sham_kl = torch.full_like(sham_delta, torch.nan)
    dense_losses = torch.empty(n_states)
    device = model.model.transformer.wte.weight.device
    for state_index, state in enumerate(manifest["states"]):
        noisy, clean, mask = _state_tensors(state, device)
        dense, prefixes, energies = _capture_dense(model, noisy, candidate_map)
        dense_loss = _official_loss(dense, clean, mask, state["p_mask"])
        dense_losses[state_index] = dense_loss.cpu()
        for layer_slot, layer in enumerate(LAYERS):
            activation[state_index, layer_slot] = energies[layer]
            for batch_slot, start in enumerate(range(0, n_candidates, batch_size)):
                neurons = candidate_map[layer][start:start + batch_size]
                logits = _variant_logits(model, prefixes[layer], layer, neurons)
                losses, divergences = masked_losses_and_kl(logits, dense, clean, mask, state["p_mask"])
                _, sham_divergences = masked_losses_and_kl(logits, logits[:1], clean, mask, state["p_mask"])
                sham_delta[state_index, layer_slot, batch_slot] = losses[0].cpu() - dense_losses[state_index]
                sham_kl[state_index, layer_slot, batch_slot] = divergences[0].cpu()
                stop = start + len(neurons)
                raw_delta[state_index, layer_slot, start:stop] = losses[1:].cpu() - dense_losses[state_index]
                delta[state_index, layer_slot, start:stop] = sham_corrected_losses(losses).cpu()
                raw_kl[state_index, layer_slot, start:stop] = divergences[1:].cpu()
                kl[state_index, layer_slot, start:stop] = sham_divergences[1:].cpu()
                del logits, losses, divergences
        print(f"state {state_index + 1}/80", flush=True)
    if not all(torch.isfinite(value).all().item() for value in (delta, kl, activation, sham_delta, sham_kl, dense_losses)):
        raise RuntimeError("collected results contain non-finite values")
    return {"delta_loss": delta, "raw_delta_loss_vs_dense_batch1": raw_delta,
            "masked_kl": kl, "raw_masked_kl_vs_dense_batch1": raw_kl, "activation_energy": activation,
            "sham_delta_loss": sham_delta, "sham_masked_kl": sham_kl, "dense_loss": dense_losses}


def _sensitivity(delta):
    return {"mean": delta.mean(dim=0), "positive": delta.clamp_min(0).mean(dim=0),
            "absolute": delta.abs().mean(dim=0), "median": delta.median(dim=0).values,
            "p90": torch.quantile(delta, .9, dim=0), "maximum": delta.max(dim=0).values,
            "positive_fraction": (delta > 0).float().mean(dim=0),
            "negative_fraction": (delta < 0).float().mean(dim=0)}


def _safe_distribution(values):
    return distribution_summary(values.clamp_min(0))


def _set_compare(left, right, fraction):
    count = max(1, math.ceil(left.numel() * fraction))
    a = set(torch.argsort(left, stable=True)[:count].tolist())
    b = set(torch.argsort(right, stable=True)[:count].tolist())
    inter = len(a & b)
    return {"count": count, "overlap": inter / count, "jaccard": inter / len(a | b),
            "crossed_boundary": 2 * (count - inter)}


def _stability(delta, group_index, group_count):
    result = {}
    for metric, transform in (("positive", lambda x: x.clamp_min(0).mean(0)),
                              ("absolute", lambda x: x.abs().mean(0))):
        grouped = torch.stack([transform(delta[group_index == group]) for group in range(group_count)])
        result[metric] = pairwise_stability(grouped)
        half_a, half_b = transform(delta[group_index < group_count // 2]), transform(delta[group_index >= group_count // 2])
        result[metric]["split_half"] = {"spearman": float(spearmanr(half_a, half_b).statistic),
                                                    "bottom_10": _set_compare(half_a, half_b, .1),
                                                    "bottom_20": _set_compare(half_a, half_b, .2)}
    return result


def _within_layer_ranks(scores):
    return torch.stack([
        torch.cat([torch.from_numpy(rankdata(scores[group, layer].numpy())).float() for layer in range(scores.shape[1])])
        for group in range(scores.shape[0])
    ])


def _pooled_stability(delta, group_index, group_count):
    result = {}
    for metric, transform in (("positive", lambda x: x.clamp_min(0).mean(0)),
                              ("absolute", lambda x: x.abs().mean(0))):
        grouped = torch.stack([transform(delta[group_index == group]) for group in range(group_count)])
        result[metric] = pairwise_stability(_within_layer_ranks(grouped))
    return result


def _trajectory_groups(delta, timestep_index):
    groups = {"early": (0, 1, 2), "middle": (3, 4, 5, 6), "late": (7, 8, 9)}
    result = {}
    for metric, transform in (("positive", lambda x: x.clamp_min(0).mean(0)),
                              ("absolute", lambda x: x.abs().mean(0))):
        scores = {name: transform(delta[torch.isin(timestep_index, torch.tensor(indices))])
                  for name, indices in groups.items()}
        result[metric] = {}
        for left, right in (("early", "middle"), ("middle", "late"), ("early", "late")):
            result[metric][f"{left}_vs_{right}"] = {
                "spearman": float(spearmanr(scores[left], scores[right]).statistic),
                "bottom_10": _set_compare(scores[left], scores[right], .1),
                "bottom_20": _set_compare(scores[left], scores[right], .2),
            }
    return result


def _metric_summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    q = torch.quantile(values, torch.tensor([.1, .5, .9], dtype=torch.float64))
    return {"mean": values.mean().item(), "median": q[1].item(), "p10": q[0].item(),
            "p90": q[2].item(), "min": values.min().item(), "max": values.max().item()}


def _baseline_comparisons(results, model, candidate_map, gradient_payload, manifest):
    delta = results["delta_loss"]
    sequence = torch.tensor([row["sequence_index"] for row in manifest["states"]])
    layer_vector = torch.arange(len(LAYERS)).repeat_interleave(32)
    magnitude = torch.stack([model.model.transformer.blocks[layer].ff_out.weight[:, candidate_map[layer]].float().norm(dim=0).cpu() for layer in LAYERS])
    gradient = gradient_payload["derivatives"]["uniform"][:, list(LAYERS)]
    candidate_indices = torch.tensor([candidate_map[layer] for layer in LAYERS])
    gradient = torch.gather(gradient, 2, candidate_indices.unsqueeze(0).expand(80, -1, -1))
    baselines_used = {"magnitude": magnitude, "activation_energy": results["activation_energy"], "uniform_gate_gradient": gradient}
    comparisons = {}
    for direction, calibration_sequences in (("A_to_B", range(4)), ("B_to_A", range(4, 8))):
        calibration = torch.isin(sequence, torch.tensor(list(calibration_sequences)))
        heldout = ~calibration
        baseline_scores = {
            "magnitude": magnitude,
            "structured_wanda": magnitude * results["activation_energy"][calibration].mean(0).sqrt(),
            "uniform_gradient_abs": gradient[calibration].abs().mean(0),
        }
        exact = {"positive": delta[heldout].clamp_min(0).mean(0), "absolute": delta[heldout].abs().mean(0)}
        comparisons[direction] = {}
        for baseline_name, baseline in baseline_scores.items():
            comparisons[direction][baseline_name] = {}
            for exact_name, target in exact.items():
                pooled = rank_predictability(baseline.flatten(), target.flatten(), layer_vector)
                per_layer = [rank_predictability(baseline[i], target[i], torch.zeros(32, dtype=torch.long)) for i in range(8)]
                comparisons[direction][baseline_name][exact_name] = {"pooled_within_layer": pooled, "per_layer": per_layer}
    return baselines_used, comparisons


def _first_order(results, gradient_payload, candidate_map):
    gradient = gradient_payload["derivatives"]["uniform"][:, list(LAYERS)]
    indices = torch.tensor([candidate_map[layer] for layer in LAYERS])
    predicted = torch.gather(gradient, 2, indices.unsqueeze(0).expand(80, -1, -1)).float()
    exact = results["delta_loss"].float()
    safe = exact.abs() > torch.quantile(exact.abs(), .01)
    return predicted, {
        "state_level": {"spearman": float(spearmanr(predicted.flatten(), exact.flatten()).statistic),
                        "pearson": float(pearsonr(predicted.flatten(), exact.flatten()).statistic),
                        "sign_agreement": (predicted.sign() == exact.sign()).float().mean().item(),
                        "mae": (predicted - exact).abs().mean().item(),
                        "relative_error_median_safe": ((predicted - exact).abs()[safe] / exact.abs()[safe]).median().item(),
                        "safe_floor": torch.quantile(exact.abs(), .01).item()},
        "aggregated_abs": rank_predictability(predicted.abs().mean(0).flatten(), exact.abs().mean(0).flatten(),
                                               torch.arange(8).repeat_interleave(32)),
    }


def analyze(results, model, candidate_map, gradient_payload, manifest):
    delta, kl = results["delta_loss"], results["masked_kl"]
    sensitivity = _sensitivity(delta)
    sequence = torch.tensor([row["sequence_index"] for row in manifest["states"]])
    timestep = torch.tensor([row["timestep_index"] for row in manifest["states"]])
    layers = []
    for slot, layer in enumerate(LAYERS):
        layers.append({"layer": layer,
                       "D_pos": _safe_distribution(sensitivity["positive"][slot]),
                       "D_abs": _safe_distribution(sensitivity["absolute"][slot]),
                       "per_neuron": [{name: float(value[slot, neuron]) for name, value in sensitivity.items()} for neuron in range(32)],
                       "sequence_stability": _stability(delta[:, slot], sequence, 8),
                       "timestep_stability": _stability(delta[:, slot], timestep, 10),
                       "trajectory_groups": _trajectory_groups(delta[:, slot], timestep),
                       "kl_mean": _safe_distribution(kl[:, slot].mean(0)),
                       "Dabs_KL_spearman": float(spearmanr(sensitivity["absolute"][slot], kl[:, slot].mean(0)).statistic)})
    pooled = {"D_pos": _safe_distribution(sensitivity["positive"].flatten()),
              "D_abs": _safe_distribution(sensitivity["absolute"].flatten()),
              "masked_kl": _safe_distribution(kl.mean(0).flatten()),
              "Dabs_KL_spearman": float(spearmanr(sensitivity["absolute"].flatten(), kl.mean(0).flatten()).statistic)}
    sham_abs = results["sham_delta_loss"].abs().flatten()
    effect_abs = delta.abs().flatten()
    sham_range = results["sham_delta_loss"].max(dim=2).values - results["sham_delta_loss"].min(dim=2).values
    sham_kl_range = results["sham_masked_kl"].max(dim=2).values - results["sham_masked_kl"].min(dim=2).values
    noise = {"delta_loss_abs": _safe_distribution(sham_abs), "masked_kl": _safe_distribution(results["sham_masked_kl"].abs().flatten()),
             "same_shape_sham_repeatability": {"delta_loss_max_range": sham_range.abs().max().item(),
                                                "delta_loss_nonzero_ranges": int((sham_range != 0).sum().item()),
                                                "kl_max_range": sham_kl_range.abs().max().item(),
                                                "kl_nonzero_ranges": int((sham_kl_range != 0).sum().item())},
             "effect_abs": _safe_distribution(effect_abs),
             "fraction_effects_at_or_below_sham_p99": (effect_abs <= torch.quantile(sham_abs, .99)).float().mean().item(),
             "median_effect_over_sham_p99": effect_abs.median().item() / torch.quantile(sham_abs, .99).clamp_min(1e-30).item()}
    baselines_used, predictability = _baseline_comparisons(results, model, candidate_map, gradient_payload, manifest)
    predicted, first_order = _first_order(results, gradient_payload, candidate_map)
    stability = {"sequence_pooled_within_layer": _pooled_stability(delta, sequence, 8),
                 "timestep_pooled_within_layer": _pooled_stability(delta, timestep, 10)}
    return layers, pooled, noise, stability, baselines_used, predictability, predicted, first_order


def main(reuse=False):
    started = time.time()
    config = load_config(CONFIG_PATH)
    validate_config(config)
    manifest = json.loads(STATE_MANIFEST_PATH.read_text())
    require_historical_digest(manifest, STATE_SHA)
    ROOT.mkdir(parents=True, exist_ok=True)
    candidates = fixed_candidates(LAYERS, 12288, 32, SEED)
    candidate_document = {"seed": SEED, "sampling": "numpy.default_rng uniform without replacement independently per layer",
                          "layers": list(LAYERS), "hidden_size": 12288, "per_layer": 32,
                          "total": 256, "neurons": candidates, "frozen_before_measurement": True}
    _atomic_json(CANDIDATE_PATH, candidate_document)
    model, devices = _load_model(config)
    weight_before = _model_sha256(model)
    if reuse:
        sanity = json.loads((ROOT / "execution_sanity.json").read_text())
        payload = torch.load(STATE_RESULTS_PATH, map_location="cpu", weights_only=True)
        weight_after_collection = weight_before
    else:
        sanity = validate_and_benchmark(model, manifest["states"][0], candidates)
        _atomic_json(ROOT / "execution_sanity.json", sanity)
        payload = {"version": 1, "state_sha256": STATE_SHA, "candidate_seed": SEED,
                   "candidate_sha256": _sha256(CANDIDATE_PATH), "batch_size": sanity["selected_candidates_per_batch"],
                   "results": collect(model, manifest, candidates, sanity["selected_candidates_per_batch"]),
                   "timestep_index": torch.tensor([row["timestep_index"] for row in manifest["states"]]),
                   "sequence_index": torch.tensor([row["sequence_index"] for row in manifest["states"]])}
        weight_after_collection = _model_sha256(model)
        if weight_after_collection != weight_before:
            raise RuntimeError("model weights changed during exact intervention collection")
        temporary = STATE_RESULTS_PATH.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(STATE_RESULTS_PATH)
        payload = torch.load(STATE_RESULTS_PATH, map_location="cpu", weights_only=True)
    if payload["state_sha256"] != STATE_SHA or payload["candidate_sha256"] != _sha256(CANDIDATE_PATH):
        raise RuntimeError("saved exact-ablation artifact source mismatch")
    gradient_payload = torch.load(GRADIENT_PATH, map_location="cpu", weights_only=True)
    layers, pooled, noise, stability, baselines, predictability, predicted, first_order = analyze(
        payload["results"], model, candidates, gradient_payload, manifest
    )
    payload["baselines"] = baselines
    payload["first_order_prediction"] = predicted
    temporary = STATE_RESULTS_PATH.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(STATE_RESULTS_PATH)
    weight_after_analysis = _model_sha256(model)
    if weight_after_analysis != weight_before:
        raise RuntimeError("model weights changed during analysis")
    result = {"status": "complete", "analysis_only": True, "cgq_used": False, "token_weighting_used": False,
              "mask_generated": False, "pruning_performed": False, "joint_ablation_performed": False,
              "downstream_evaluation_performed": False,
              "source": {"model": config["model"], "state_count": 80, "state_sha256": STATE_SHA,
                         "state_manifest": str(STATE_MANIFEST_PATH), "gradient_artifact": str(GRADIENT_PATH)},
              "candidates": candidate_document, "intervention": {"location": "ff_out input h after gated MLP product",
                         "operation": "one candidate channel h[...,k]=0 per variant; no parameter mutation"},
              "execution": {"strategy": "one dense full forward captures eight prefix states; batched suffix forwards each include row-0 sham; primary effects are same-shape ablated-minus-sham contrasts",
                            "sanity": sanity}, "measurement_noise": noise, "pooled": pooled, "layers": layers,
              "stability": stability,
              "baseline_predictability": predictability, "first_order_fidelity": first_order,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                              "cuda_runtime": torch.version.cuda, "devices": list(devices), "gpu": torch.cuda.get_device_name(devices[0])},
              "integrity": {"weight_sha256_before": weight_before, "weight_sha256_after_collection": weight_after_collection,
                            "weight_sha256_after_analysis": weight_after_analysis, "candidate_sha256": _sha256(CANDIDATE_PATH),
                            "state_results_sha256": _sha256(STATE_RESULTS_PATH), "state_results_path": str(STATE_RESULTS_PATH.resolve())},
              "runtime_seconds": time.time() - started}
    _atomic_json(MANIFEST_PATH, result)
    print(json.dumps({"status": "complete", "manifest": str(MANIFEST_PATH), "seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true")
    main(reuse=parser.parse_args().reuse)
