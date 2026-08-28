import argparse
import csv
import hashlib
import json
import math
import os
import resource
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.dlm_loss_aggregation.core import (
    EffectAccumulator,
    analyze_scores,
    mask_sha256,
    pack_mask,
    pairwise_diagnostics,
    profile_pairwise_spearman,
    rowwise_mask,
)
from lib.dlm_gradient_sensitivity import make_masked_state, official_dlm_loss


_HISTORICAL_STATE_KEYS = (
    "timestep_index",
    "timestep",
    "sequence_index",
    "mask_seed",
    "p_mask",
    "clean_ids",
    "noisy_ids",
    "mask",
)


def _canonical_json(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config


def validate_config(config):
    expected = {
        "experiment": "dlm_loss_aggregation_1a",
        "model.id": "GSAI-ML/LLaDA-8B-Base",
        "model.revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
        "model.dtype": "bfloat16",
        "dataset.id": "Salesforce/wikitext",
        "dataset.configuration": "wikitext-2-raw-v1",
        "dataset.loader_name": "wikitext2",
        "dataset.split": "train",
        "calibration.seed": 0,
        "calibration.sequence_indices": list(range(8)),
        "calibration.sequence_count": 8,
        "calibration.sequence_length": 256,
        "calibration.epsilon": 0.001,
        "calibration.timesteps": [
            0.05,
            0.15,
            0.25,
            0.35,
            0.45,
            0.55,
            0.65,
            0.75,
            0.85,
            0.95,
        ],
        "calibration.mask_id_source": "model.config.mask_token_id",
        "calibration.expected_state_sha256": "1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df",
        "scoring.methods": ["sum", "abs", "square"],
        "scoring.state_weight": "uniform",
        "scoring.sparsity": 0.5,
        "scoring.block_count": 32,
        "scoring.modules_per_block": 7,
        "evaluation.order": [
            "dense",
            "dlm_sum",
            "dlm_abs",
            "dlm_square",
            "wanda",
            "sparsegpt",
        ],
    }
    for path, wanted in expected.items():
        actual = config
        try:
            for key in path.split("."):
                actual = actual[key]
        except (KeyError, TypeError):
            raise ValueError(f"{path} is required") from None
        if actual != wanted:
            raise ValueError(f"{path} does not match Experiment 1A")
    return config


def build_calibration_manifest(clean_ids, mask_id, config):
    calibration = config["calibration"]
    sequence_indices = calibration["sequence_indices"]
    if len(clean_ids) != len(sequence_indices):
        raise ValueError("clean calibration sequence count does not match config")
    states = []
    for timestep_index, timestep in enumerate(calibration["timesteps"]):
        for sequence_position, (sequence_index, clean) in enumerate(
            zip(sequence_indices, clean_ids)
        ):
            clean = clean.detach().to(device="cpu", dtype=torch.long)
            if tuple(clean.shape) != (1, calibration["sequence_length"]):
                raise ValueError("clean calibration sequence shape does not match config")
            mask_seed = (
                calibration["seed"]
                + timestep_index * len(sequence_indices)
                + sequence_position
            )
            noisy, mask, p_mask = make_masked_state(
                clean,
                timestep,
                mask_id,
                mask_seed,
                eps=calibration["epsilon"],
            )
            states.append(
                {
                    "timestep_index": timestep_index,
                    "timestep": timestep,
                    "sequence_index": sequence_index,
                    "mask_seed": mask_seed,
                    "p_mask": p_mask,
                    "clean_ids": clean.tolist(),
                    "noisy_ids": noisy.tolist(),
                    "mask": mask.tolist(),
                }
            )
    return {
        "version": 1,
        "model": config["model"],
        "dataset": config["dataset"],
        "global_random_seed": calibration["seed"],
        "states": states,
    }


def historical_state_digest(manifest):
    states = [
        {key: state[key] for key in _HISTORICAL_STATE_KEYS}
        for state in manifest["states"]
    ]
    return hashlib.sha256(_canonical_json({"version": 1, "states": states})).hexdigest()


def require_historical_digest(manifest, expected):
    actual = historical_state_digest(manifest)
    if actual != expected:
        raise ValueError(f"calibration digest mismatch: expected {expected}, got {actual}")
    return actual


class BlockEffectCollector:
    def __init__(self, layers):
        if not layers:
            raise ValueError("layers must not be empty")
        self.layers = dict(layers)
        self.weights = {
            name: layer.weight.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, layer in self.layers.items()
        }
        self.accumulators = {
            name: EffectAccumulator(weight.shape) for name, weight in self.weights.items()
        }
        self._seen = set()
        self._state_weight = 1.0

    def set_state_weight(self, state_weight):
        if self._seen:
            raise RuntimeError("cannot change state weight during a backward")
        if not math.isfinite(state_weight) or state_weight <= 0:
            raise ValueError("state_weight must be finite and positive")
        self._state_weight = state_weight

    @contextmanager
    def hooks(self):
        handles = []
        for name, layer in self.layers.items():
            parameter = layer.weight

            def accumulate(current, parameter_name=name):
                gradient = current.grad
                if gradient is None or not torch.isfinite(gradient).all().item():
                    raise ValueError(f"missing or non-finite gradient: {parameter_name}")
                if parameter_name in self._seen:
                    raise RuntimeError(f"gradient accumulated twice: {parameter_name}")
                effect = gradient.detach().to(device="cpu", dtype=torch.float32)
                effect.mul_(self.weights[parameter_name]).neg_()
                self.accumulators[parameter_name].add(effect, self._state_weight)
                self._seen.add(parameter_name)
                current.grad = None

            handles.append(parameter.register_post_accumulate_grad_hook(accumulate))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def step(self):
        missing = self.layers.keys() - self._seen
        if missing:
            raise RuntimeError("missing gradients: " + ", ".join(sorted(missing)))
        self._seen.clear()
        self._state_weight = 1.0

    def finalize(self):
        if self._seen:
            raise RuntimeError("finish the pending backward before finalizing")
        counts = {accumulator.count for accumulator in self.accumulators.values()}
        if len(counts) != 1:
            raise RuntimeError("module gradient counts differ")
        return {
            name: accumulator.finalize()
            for name, accumulator in self.accumulators.items()
        }


def validate_smoke_module(
    dense_before,
    dense_after,
    loss,
    scores,
    count,
    expected_count,
    spearman_sample_size=None,
):
    if loss.numel() != 1 or not torch.isfinite(loss).item():
        raise ValueError("smoke loss must be finite")
    if count != expected_count:
        raise ValueError("smoke gradient count mismatch")
    if not torch.equal(dense_before, dense_after):
        raise RuntimeError("dense weights changed before mask application")
    if set(scores) != {"sum", "abs", "square"}:
        raise ValueError("smoke scores are incomplete")
    if any(score.shape != dense_before.shape for score in scores.values()):
        raise ValueError("smoke score shape mismatch")
    if any(not torch.isfinite(score).all().item() for score in scores.values()):
        raise ValueError("smoke scores must be finite")
    if scores["abs"].lt(0).any().item() or scores["square"].lt(0).any().item():
        raise ValueError("absolute and square scores must be nonnegative")
    expected_pruned = dense_before.shape[1] // 2
    for score in scores.values():
        counts = rowwise_mask(score, 0.5).sum(dim=1)
        if not counts.eq(expected_pruned).all().item():
            raise RuntimeError("smoke row sparsity is incorrect")
    return pairwise_diagnostics(scores, spearman_sample_size=spearman_sample_size)


def weighted_pair_summaries(rows):
    summaries = []
    pairs = {(row["left"], row["right"]) for row in rows}
    module_types = {row["module_type"] for row in rows}
    for left, right in sorted(pairs):
        pair_rows = [
            row for row in rows if (row["left"], row["right"]) == (left, right)
        ]
        groups = [("weighted_overall", "all", pair_rows)] + [
            (
                "module_type",
                module_type,
                [row for row in pair_rows if row["module_type"] == module_type],
            )
            for module_type in sorted(module_types)
        ]
        for scope, module_type, group in groups:
            weight = sum(row["num_weights"] for row in group)
            defined = [row for row in group if row["spearman"] is not None]
            spearman_weight = sum(row["num_weights"] for row in defined)
            summaries.append(
                {
                    "scope": scope,
                    "module_type": module_type,
                    "left": left,
                    "right": right,
                    "num_weights": weight,
                    "matrix_count": len(group),
                    "spearman": None if not defined else sum(
                        row["spearman"] * row["num_weights"] for row in group
                        if row["spearman"] is not None
                    ) / spearman_weight,
                    "mask_xor": sum(
                        row["mask_xor"] * row["num_weights"] for row in group
                    )
                    / weight,
                }
            )
    return summaries


def evaluate_in_memory_sequence(config, dlm_masks, load_dense, prune, evaluate):
    results = []
    labels = config["evaluation"]["calibration_labels"]
    for method in config["evaluation"]["order"]:
        model = load_dense()
        mask = dlm_masks.get(method)
        mask_hash = ""
        if method != "dense":
            mask_hash = prune(model, method, mask)
        measured = evaluate(model, method)
        if set(measured) != {"accuracy", "num_examples", "eval_seconds"}:
            raise ValueError("evaluation result fields are incomplete")
        label = labels.get("dlm" if method.startswith("dlm_") else method)
        results.append(
            {
                "method": method,
                "sparsity": 0.0 if method == "dense" else config["scoring"]["sparsity"],
                **measured,
                "model_revision": config["model"]["revision"],
                "calibration_label": label,
                "mask_hash": mask_hash,
                "checkpoint": "in-memory",
            }
        )
        if method.startswith("dlm_"):
            del dlm_masks[method]
        del model
    return results


def _atomic_write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", newline="", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rss_peak_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _available_memory_bytes():
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _load_clean_calibration(config):
    from model import LLaDAConfig
    from transformers import AutoTokenizer

    from lib.data import get_loaders

    model_config = LLaDAConfig.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"]
    )
    mask_id = model_config.mask_token_id
    if mask_id is None:
        raise ValueError("pinned model does not define mask_token_id")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    calibration = config["calibration"]
    loader, _ = get_loaders(
        config["dataset"]["loader_name"],
        nsamples=calibration["sequence_count"],
        seed=calibration["seed"],
        seqlen=calibration["sequence_length"],
        tokenizer=tokenizer,
    )
    return [sample[0] for sample in loader], mask_id


def generate_manifest(config, output_path):
    clean_ids, mask_id = _load_clean_calibration(config)
    manifest = build_calibration_manifest(clean_ids, mask_id, config)
    digest = require_historical_digest(
        manifest, config["calibration"]["expected_state_sha256"]
    )
    manifest["mask_id"] = mask_id
    manifest["historical_state_sha256"] = digest
    _atomic_write_json(output_path, manifest)
    saved = json.loads(Path(output_path).read_text())
    require_historical_digest(saved, digest)
    return saved


def _load_verified_manifest(config, path):
    manifest = json.loads(Path(path).read_text())
    require_historical_digest(
        manifest, config["calibration"]["expected_state_sha256"]
    )
    if manifest.get("model") != config["model"] or manifest.get("dataset") != config["dataset"]:
        raise ValueError("calibration manifest metadata does not match config")
    return manifest


def _load_model(config):
    from model import LLaDAModelLM

    model = LLaDAModelLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    model.seqlen = config["calibration"]["sequence_length"]
    devices = set()
    for parameter in model.parameters():
        if parameter.device.type != "cuda":
            raise RuntimeError("CPU/disk model offload is not allowed for profiling")
        devices.add(parameter.device.index or 0)
    return model, tuple(sorted(devices))


def _embed_state(model, noisy_ids):
    core = model.model
    transformer = core.transformer
    hidden = transformer.wte(noisy_ids.to(transformer.wte.weight.device))
    if core.config.input_emb_norm:
        hidden = hidden * math.sqrt(core.config.d_model)
    if not (core.config.alibi or core.config.rope):
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        hidden = hidden + transformer.wpe(positions)
    return transformer.emb_drop(hidden)


@torch.no_grad()
def _hidden_at_block(model, noisy_ids, block_index):
    hidden = _embed_state(model, noisy_ids)
    for block in model.model.transformer.blocks[:block_index]:
        parameter = next(block.parameters())
        hidden, _ = block(
            hidden.to(device=parameter.device, dtype=parameter.dtype),
            attention_bias=None,
            layer_past=None,
            use_cache=False,
            replace_position=None,
            attn_collector=None,
        )
    return hidden.detach()


def _suffix_logits(model, hidden, start_block):
    core = model.model
    for block in core.transformer.blocks[start_block:]:
        parameter = next(block.parameters())
        hidden, _ = block(
            hidden.to(device=parameter.device, dtype=parameter.dtype),
            attention_bias=None,
            layer_past=None,
            use_cache=False,
            replace_position=None,
            attn_collector=None,
        )
    parameter = next(core.transformer.ln_f.parameters())
    hidden = core.transformer.ln_f(
        hidden.to(device=parameter.device, dtype=parameter.dtype)
    )
    if core.config.weight_tying:
        weight = core.transformer.wte.weight
        logits = torch.nn.functional.linear(hidden.to(weight.device, weight.dtype), weight)
    else:
        parameter = next(core.transformer.ff_out.parameters())
        logits = core.transformer.ff_out(
            hidden.to(device=parameter.device, dtype=parameter.dtype)
        )
    if core.config.scale_logits:
        logits.mul_(1 / math.sqrt(core.config.d_model))
    return logits


def _state_tensors(state):
    return (
        torch.tensor(state["noisy_ids"], dtype=torch.long),
        torch.tensor(state["clean_ids"], dtype=torch.long),
        torch.tensor(state["mask"], dtype=torch.bool),
    )


def _score_module(model, block_index, module_name, states):
    from lib.prune_llada import find_layers

    block = model.model.transformer.blocks[block_index]
    layers = find_layers(block)
    if module_name not in layers:
        raise ValueError(f"unknown target module: {module_name}")
    layer = layers[module_name]
    hidden_states = []
    prefix_started = time.perf_counter()
    for state in states:
        noisy_ids, _, _ = _state_tensors(state)
        hidden_states.append(_hidden_at_block(model, noisy_ids, block_index))
    prefix_seconds = time.perf_counter() - prefix_started

    parameters = list(model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    collector = BlockEffectCollector({module_name: layer})
    losses = []
    state_seconds = []
    model.zero_grad(set_to_none=True)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        layer.weight.requires_grad_(True)
        with collector.hooks():
            for state, hidden in zip(states, hidden_states):
                started = time.perf_counter()
                parameter = next(block.parameters())
                target_output, _ = block(
                    hidden.to(device=parameter.device, dtype=parameter.dtype),
                    attention_bias=None,
                    layer_past=None,
                    use_cache=False,
                    replace_position=None,
                    attn_collector=None,
                )
                logits = _suffix_logits(model, target_output, block_index + 1)
                _, clean_ids, mask = _state_tensors(state)
                loss = official_dlm_loss(
                    logits,
                    clean_ids.to(logits.device),
                    mask.to(logits.device),
                    state["p_mask"],
                )
                if not torch.isfinite(loss).item():
                    raise ValueError("DLM loss is non-finite")
                loss.backward()
                collector.step()
                model.zero_grad(set_to_none=True)
                for device in _cuda_devices(model):
                    torch.cuda.synchronize(device)
                state_seconds.append(time.perf_counter() - started)
                losses.append(loss.detach().cpu().item())
                del target_output, logits, loss
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, requires_grad in zip(parameters, original_flags):
            parameter.requires_grad_(requires_grad)
    scores = collector.finalize()[module_name]
    dense_after = layer.weight.detach().to(device="cpu", dtype=torch.float32)
    return {
        "scores": scores,
        "count": collector.accumulators[module_name].count,
        "dense_before": collector.weights[module_name],
        "dense_after": dense_after,
        "losses": losses,
        "prefix_seconds": prefix_seconds,
        "state_seconds": state_seconds,
    }


def _cuda_devices(model):
    return tuple(
        sorted(
            {
                parameter.device.index or 0
                for parameter in model.parameters()
                if parameter.device.type == "cuda"
            }
        )
    )


def _reset_cuda_peaks(devices):
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peaks(devices):
    rows = [
        {
            "device": f"cuda:{device}",
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        for device in devices
    ]
    return {
        "devices": rows,
        "max_allocated_bytes": max(row["allocated_bytes"] for row in rows),
        "max_reserved_bytes": max(row["reserved_bytes"] for row in rows),
    }


def _process_memory():
    result = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmSwap:")):
                name, value, _ = line.split()
                result[name.rstrip(":").lower() + "_kib"] = int(value)
    return result


def _largest_module(model, block_index):
    from lib.prune_llada import find_layers

    layers = find_layers(model.model.transformer.blocks[block_index])
    return max(layers, key=lambda name: layers[name].weight.numel())


def run_smoke(config, manifest_path, output_path):
    manifest = _load_verified_manifest(config, manifest_path)
    model, devices = _load_model(config)
    block_index = len(model.model.transformer.blocks) - 1
    if block_index + 1 != config["scoring"]["block_count"]:
        raise ValueError("loaded model block count does not match config")
    module_name = _largest_module(model, block_index)
    states = manifest["states"][:2]
    _reset_cuda_peaks(devices)
    measured = _score_module(model, block_index, module_name, states)
    diagnostics = validate_smoke_module(
        measured["dense_before"],
        measured["dense_after"],
        torch.tensor(measured["losses"][-1]),
        measured["scores"],
        measured["count"],
        len(states),
        spearman_sample_size=config["scoring"]["spearman"]["fallback_sample_size"],
    )
    result = {
        "status": "passed",
        "block_index": block_index,
        "module": module_name,
        "shape": list(measured["dense_before"].shape),
        "state_count": measured["count"],
        "losses": measured["losses"],
        "prefix_seconds": measured["prefix_seconds"],
        "state_seconds": measured["state_seconds"],
        "cuda_peak": _cuda_peaks(devices),
        "process_memory": _process_memory(),
        "diagnostics": diagnostics,
    }
    maximum = config["resources"]["max_cuda_gib"] * 1024**3
    if result["cuda_peak"]["max_allocated_bytes"] > maximum or result["cuda_peak"]["max_reserved_bytes"] > maximum:
        raise RuntimeError("smoke CUDA peak exceeds configured limit")
    _atomic_write_json(output_path, result)
    return result


def run_profile(config, manifest_path, output_path):
    manifest = _load_verified_manifest(config, manifest_path)
    model, devices = _load_model(config)
    block_index = len(model.model.transformer.blocks) - 1
    module_name = _largest_module(model, block_index)
    _reset_cuda_peaks(devices)
    memory_before = _process_memory()
    measured = _score_module(model, block_index, module_name, manifest["states"][:1])
    spearman = profile_pairwise_spearman(measured["scores"])
    limits = config["scoring"]["spearman"]
    comfortable = (
        spearman["elapsed_seconds"] <= limits["comfortable_max_seconds"]
        and spearman["rss_delta_kib"] <= limits["comfortable_max_rss_delta_gib"] * 1024**2
    )
    result = {
        "status": "passed",
        "block_index": block_index,
        "module": module_name,
        "shape": list(measured["dense_before"].shape),
        "loss": measured["losses"][0],
        "prefix_seconds": measured["prefix_seconds"],
        "state_seconds": measured["state_seconds"][0],
        "projected_32_block_80_state_seconds": measured["state_seconds"][0] * 32 * 80,
        "cuda_peak": _cuda_peaks(devices),
        "process_memory_before": memory_before,
        "process_memory_after": _process_memory(),
        "exact_spearman": spearman,
        "spearman_decision": {
            "mode": "exact" if comfortable else "sampled",
            "sample_size": None if comfortable else limits["fallback_sample_size"],
            "comfortable": comfortable,
            "limits": {
                "max_seconds": limits["comfortable_max_seconds"],
                "max_rss_delta_gib": limits["comfortable_max_rss_delta_gib"],
            },
        },
    }
    maximum = config["resources"]["max_cuda_gib"] * 1024**3
    if result["cuda_peak"]["max_allocated_bytes"] > maximum or result["cuda_peak"]["max_reserved_bytes"] > maximum:
        raise RuntimeError("profile CUDA peak exceeds configured limit")
    _atomic_write_json(output_path, result)
    return result


@torch.no_grad()
def _initial_cached_states(model, manifest):
    cached = []
    for state in manifest["states"]:
        noisy_ids, clean_ids, mask = _state_tensors(state)
        cached.append(
            {
                "hidden": _embed_state(model, noisy_ids).detach().cpu(),
                "clean_ids": clean_ids,
                "mask": mask,
                "p_mask": state["p_mask"],
            }
        )
    return cached


def _distribution_group():
    return {"count": 0, "sum": 0.0, "min": math.inf, "max": -math.inf, "samples": []}


def _add_distribution(groups, module_type, values, summary, overall_quota, type_quota, seed):
    flat = values.detach().cpu().reshape(-1)
    sample_count = min(flat.numel(), max(overall_quota, type_quota))
    indices = np.random.default_rng(seed).choice(flat.numel(), sample_count, replace=False)
    sample = flat[torch.from_numpy(indices.astype(np.int64, copy=False))].clone()
    for key, quota in (("all", overall_quota), (module_type, type_quota)):
        group = groups.setdefault(key, _distribution_group())
        group["count"] += flat.numel()
        group["sum"] += summary["mean"] * flat.numel()
        group["min"] = min(group["min"], summary["min"])
        group["max"] = max(group["max"], summary["max"])
        group["samples"].append(sample[:quota])


def _finish_distribution_groups(groups):
    rows = []
    for key, group in sorted(groups.items()):
        sample = torch.cat(group.pop("samples"))
        quantiles = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.99]))
        rows.append(
            {
                "scope": "weighted_overall" if key == "all" else "module_type",
                "module_type": key,
                "num_weights": group["count"],
                "mean": group["sum"] / group["count"],
                "min": group["min"],
                "p50": quantiles[0].item(),
                "p90": quantiles[1].item(),
                "p99": quantiles[2].item(),
                "max": group["max"],
                "quantile_sample_size": sample.numel(),
                "sample_indices_sha256": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
            }
        )
    return rows


def _negative_sum_summaries(rows):
    summaries = []
    for module_type in ("all", *sorted({row["module_type"] for row in rows})):
        group = rows if module_type == "all" else [
            row for row in rows if row["module_type"] == module_type
        ]
        count = sum(row["num_weights"] for row in group)
        negative = sum(row["negative_count"] for row in group)
        summaries.append(
            {
                "scope": "weighted_overall" if module_type == "all" else "module_type",
                "module_type": module_type,
                "num_weights": count,
                "matrix_count": len(group),
                "negative_count": negative,
                "negative_sum_fraction": negative / count,
            }
        )
    return summaries


def _write_transient_mask(mask_dir, method, block_index, module_name, mask):
    payload = pack_mask(mask)
    digest = mask_sha256(payload)
    destination = mask_dir / method / f"block_{block_index:02d}__{module_name.replace('.', '_')}.bin"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload["bits"])
        handle.flush()
        os.fsync(handle.fileno())
    saved = {"shape": payload["shape"], "bits": destination.read_bytes()}
    if mask_sha256(saved) != digest:
        raise RuntimeError("transient mask readback failed")
    return {
        "method": method,
        "block_index": block_index,
        "module": module_name,
        "shape": list(payload["shape"]),
        "byte_length": len(payload["bits"]),
        "sha256": digest,
        "runtime_path": str(destination),
    }


def _module_specs(model):
    from lib.prune_llada import find_layers

    specs = []
    for block_index, block in enumerate(model.model.transformer.blocks):
        layers = find_layers(block)
        if len(layers) != 7:
            raise ValueError(f"block {block_index} has {len(layers)} prunable matrices, expected 7")
        for name, layer in layers.items():
            specs.append(
                {
                    "block_index": block_index,
                    "module": name,
                    "module_type": name.rsplit(".", 1)[-1],
                    "num_weights": layer.weight.numel(),
                    "shape": list(layer.weight.shape),
                }
            )
    return specs


def run_full_scoring(config, manifest_path, root):
    from lib.prune_llada import find_layers

    started = time.perf_counter()
    manifest = _load_verified_manifest(config, manifest_path)
    profile = json.loads((root / "diagnostics" / "spearman_profile.json").read_text())
    if profile["spearman_decision"]["mode"] != "exact":
        spearman_sample_size = profile["spearman_decision"]["sample_size"]
    else:
        spearman_sample_size = None
    model, devices = _load_model(config)
    specs = _module_specs(model)
    expected_matrices = config["scoring"]["block_count"] * config["scoring"]["modules_per_block"]
    if len(specs) != expected_matrices:
        raise ValueError(f"found {len(specs)} prunable matrices, expected {expected_matrices}")

    total_weights = sum(spec["num_weights"] for spec in specs)
    weights_by_type = {}
    for spec in specs:
        weights_by_type[spec["module_type"]] = weights_by_type.get(spec["module_type"], 0) + spec["num_weights"]
    mask_bytes_expected = 3 * sum(math.ceil(spec["num_weights"] / 8) for spec in specs)
    shm = Path("/dev/shm")
    free_shm = os.statvfs(shm).f_bavail * os.statvfs(shm).f_frsize
    if free_shm < mask_bytes_expected + 512 * 1024**2:
        raise RuntimeError(
            f"insufficient /dev/shm: need {mask_bytes_expected + 512 * 1024**2}, have {free_shm}"
        )
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    mask_dir = shm / f"dlm_loss_aggregation_{run_id}"
    mask_dir.mkdir()
    _atomic_write_json(
        root / "logs" / "full_score_start.json",
        {
            "run_id": run_id,
            "started_unix": time.time(),
            "manifest_sha256": manifest["historical_state_sha256"],
            "matrix_count": len(specs),
            "state_count": len(manifest["states"]),
            "transient_mask_directory": str(mask_dir),
            "expected_transient_mask_bytes": mask_bytes_expected,
            "available_memory_bytes": _available_memory_bytes(),
            "available_shm_bytes": free_shm,
        },
    )

    cached_states = _initial_cached_states(model, manifest)
    parameters = list(model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    _reset_cuda_peaks(devices)

    pair_rows = []
    distribution_rows = {"sign_consistency": [], "spike_ratio": []}
    distribution_groups = {"sign_consistency": {}, "spike_ratio": {}}
    negative_rows = []
    mask_entries = []
    module_metadata = []
    block_timings = []
    loss_sum = 0.0
    loss_min = math.inf
    loss_max = -math.inf
    loss_count = 0
    mask_bytes_written = 0
    peak_accounted_cpu_bytes = 0

    try:
        for block_index, block in enumerate(model.model.transformer.blocks):
            block_started = time.perf_counter()
            layers = find_layers(block)
            for layer in layers.values():
                layer.weight.requires_grad_(True)
            collector = BlockEffectCollector(layers)
            next_hiddens = []
            model.zero_grad(set_to_none=True)
            with collector.hooks():
                for state in cached_states:
                    parameter = next(block.parameters())
                    target_output, _ = block(
                        state["hidden"].to(device=parameter.device, dtype=parameter.dtype),
                        attention_bias=None,
                        layer_past=None,
                        use_cache=False,
                        replace_position=None,
                        attn_collector=None,
                    )
                    if block_index + 1 < config["scoring"]["block_count"]:
                        next_hiddens.append(target_output.detach().cpu())
                    logits = _suffix_logits(model, target_output, block_index + 1)
                    loss = official_dlm_loss(
                        logits,
                        state["clean_ids"].to(logits.device),
                        state["mask"].to(logits.device),
                        state["p_mask"],
                    )
                    if not torch.isfinite(loss).item():
                        raise ValueError("DLM loss is non-finite")
                    loss_value = loss.detach().cpu().item()
                    loss_sum += loss_value
                    loss_min = min(loss_min, loss_value)
                    loss_max = max(loss_max, loss_value)
                    loss_count += 1
                    loss.backward()
                    collector.step()
                    model.zero_grad(set_to_none=True)
                    del target_output, logits, loss
            for layer in layers.values():
                layer.weight.requires_grad_(False)
            if next_hiddens:
                for state, hidden in zip(cached_states, next_hiddens):
                    state["hidden"] = hidden

            counts = {accumulator.count for accumulator in collector.accumulators.values()}
            if counts != {len(manifest["states"])}:
                raise RuntimeError(f"block {block_index} update counts are {sorted(counts)}")
            for module_position, (name, layer) in enumerate(layers.items()):
                module_type = name.rsplit(".", 1)[-1]
                accumulator = collector.accumulators.pop(name)
                if accumulator.count != len(manifest["states"]):
                    raise RuntimeError(f"wrong update count: block {block_index} {name}")
                scores = accumulator.finalize()
                del accumulator
                if any(score.shape != layer.weight.shape for score in scores.values()):
                    raise RuntimeError(f"wrong score shape: block {block_index} {name}")
                if scores["abs"].lt(0).any().item() or scores["square"].lt(0).any().item():
                    raise RuntimeError(f"negative magnitude score: block {block_index} {name}")
                dense_before = collector.weights.pop(name)
                dense_after = layer.weight.detach().to(device="cpu", dtype=torch.float32)
                if not torch.equal(dense_before, dense_after):
                    raise RuntimeError(f"dense weights changed: block {block_index} {name}")
                del dense_before, dense_after
                diagnostics, masks, distributions = analyze_scores(
                    scores,
                    spearman_sample_size=spearman_sample_size,
                    seed=config["scoring"]["spearman"]["seed"],
                )
                num_weights = layer.weight.numel()
                expected_per_row = layer.weight.shape[1] // 2
                for method, mask in masks.items():
                    if not mask.sum(dim=1).eq(expected_per_row).all().item():
                        raise RuntimeError(f"wrong row sparsity: block {block_index} {name} {method}")
                    entry = _write_transient_mask(mask_dir, method, block_index, name, mask)
                    mask_entries.append(entry)
                    mask_bytes_written += entry["byte_length"]
                for pair in diagnostics["pairs"]:
                    pair_rows.append(
                        {
                            "scope": "matrix",
                            "layer": block_index,
                            "module": name,
                            "module_type": module_type,
                            "num_weights": num_weights,
                            **pair,
                        }
                    )
                negative_count = int(scores["sum"].lt(0).sum().item())
                negative_rows.append(
                    {
                        "scope": "matrix",
                        "layer": block_index,
                        "module": name,
                        "module_type": module_type,
                        "num_weights": num_weights,
                        "negative_count": negative_count,
                        "negative_sum_fraction": negative_count / num_weights,
                    }
                )
                spec_seed = block_index * config["scoring"]["modules_per_block"] + module_position
                overall_quota = max(1, round(1_000_000 * num_weights / total_weights))
                type_quota = max(1, round(1_000_000 * num_weights / weights_by_type[module_type]))
                for metric, values in distributions.items():
                    summary = diagnostics[metric]
                    distribution_rows[metric].append(
                        {
                            "scope": "matrix",
                            "layer": block_index,
                            "module": name,
                            "module_type": module_type,
                            "num_weights": num_weights,
                            **summary,
                        }
                    )
                    _add_distribution(
                        distribution_groups[metric],
                        module_type,
                        values,
                        summary,
                        overall_quota,
                        type_quota,
                        config["scoring"]["spearman"]["seed"] + spec_seed,
                    )
                module_metadata.append(
                    {
                        "layer": block_index,
                        "module": name,
                        "module_type": module_type,
                        "shape": list(layer.weight.shape),
                        "num_weights": num_weights,
                        "update_count": len(manifest["states"]),
                        "finite": True,
                        "dense_unchanged": True,
                        "per_row_prune_count": expected_per_row,
                    }
                )
                del scores, masks, distributions, diagnostics
                peak_accounted_cpu_bytes = max(
                    peak_accounted_cpu_bytes,
                    _rss_peak_kib() * 1024 + mask_bytes_written,
                )
            del collector
            for device in devices:
                torch.cuda.synchronize(device)
            block_seconds = time.perf_counter() - block_started
            block_timings.append({"layer": block_index, "seconds": block_seconds})
            progress = {
                "run_id": run_id,
                "completed_blocks": block_index + 1,
                "completed_matrices": len(module_metadata),
                "elapsed_seconds": time.perf_counter() - started,
                "last_block_seconds": block_seconds,
                "peak_process_rss_kib": _rss_peak_kib(),
                "transient_mask_bytes": mask_bytes_written,
                "cuda_peak": _cuda_peaks(devices),
            }
            maximum_cuda = config["resources"]["max_cuda_gib"] * 1024**3
            if (
                progress["cuda_peak"]["max_allocated_bytes"] > maximum_cuda
                or progress["cuda_peak"]["max_reserved_bytes"] > maximum_cuda
            ):
                raise RuntimeError("full scoring CUDA peak exceeds configured limit")
            _atomic_write_json(root / "logs" / "full_score_progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, requires_grad in zip(parameters, original_flags):
            parameter.requires_grad_(requires_grad)

    if len(module_metadata) != expected_matrices or any(
        row["update_count"] != len(manifest["states"]) for row in module_metadata
    ):
        raise RuntimeError("not all prunable matrices received every state")
    if mask_bytes_written != mask_bytes_expected:
        raise RuntimeError(f"mask byte count mismatch: expected {mask_bytes_expected}, got {mask_bytes_written}")

    pair_summaries = weighted_pair_summaries(pair_rows)
    spearman_rows = [
        {**row, "exact": spearman_sample_size is None} for row in pair_rows
    ] + [{**row, "exact": spearman_sample_size is None} for row in pair_summaries]
    xor_rows = pair_rows + pair_summaries
    for metric in distribution_rows:
        distribution_rows[metric].extend(
            _finish_distribution_groups(distribution_groups[metric])
        )
    negative_rows.extend(_negative_sum_summaries(negative_rows))

    pair_fields = [
        "scope", "layer", "module", "module_type", "num_weights", "matrix_count",
        "left", "right", "spearman", "reason", "sample_size",
        "sample_indices_sha256", "exact", "mask_xor",
    ]
    _atomic_write_csv(root / "diagnostics" / "score_spearman.csv", spearman_rows, pair_fields)
    _atomic_write_csv(root / "diagnostics" / "mask_xor.csv", xor_rows, pair_fields)
    distribution_fields = [
        "scope", "layer", "module", "module_type", "num_weights", "mean", "min",
        "p50", "p90", "p99", "max", "quantile_sample_size", "sample_indices_sha256",
    ]
    _atomic_write_csv(
        root / "diagnostics" / "sign_consistency.csv",
        distribution_rows["sign_consistency"],
        distribution_fields,
    )
    _atomic_write_csv(
        root / "diagnostics" / "spike_ratio.csv",
        distribution_rows["spike_ratio"],
        distribution_fields,
    )
    _atomic_write_csv(
        root / "diagnostics" / "negative_sum_fraction.csv",
        negative_rows,
        [
            "scope", "layer", "module", "module_type", "num_weights", "matrix_count",
            "negative_count", "negative_sum_fraction",
        ],
    )

    overall_hashes = {}
    for method in config["scoring"]["methods"]:
        hasher = hashlib.sha256()
        for entry in mask_entries:
            if entry["method"] == method:
                hasher.update(_canonical_json({key: entry[key] for key in ("block_index", "module", "shape", "sha256")}))
        overall_hashes[method] = hasher.hexdigest()
    mask_metadata = {
        "version": 1,
        "storage": "transient RAM-backed /dev/shm raw packed bits",
        "runtime_directory": str(mask_dir),
        "total_bytes": mask_bytes_written,
        "overall_sha256": overall_hashes,
        "entries": mask_entries,
    }
    _atomic_write_json(root / "masks" / "hashes.json", mask_metadata)
    _atomic_write_json(
        root / "scores" / "metadata.json",
        {
            "equations": {
                "effect": "-weight * grad",
                "sum": "mean(effect)",
                "abs": "mean(abs(effect))",
                "square": "mean(effect ** 2)",
            },
            "state_weight": "uniform",
            "state_count": len(manifest["states"]),
            "matrix_count": len(module_metadata),
            "modules": module_metadata,
        },
    )
    total_seconds = time.perf_counter() - started
    result = {
        "status": "passed",
        "run_id": run_id,
        "total_runtime_seconds": total_seconds,
        "block_timings": block_timings,
        "matrix_count": len(module_metadata),
        "state_count": len(manifest["states"]),
        "gradient_update_count": sum(row["update_count"] for row in module_metadata),
        "loss": {"count": loss_count, "mean": loss_sum / loss_count, "min": loss_min, "max": loss_max},
        "peak_process_rss_kib": _rss_peak_kib(),
        "transient_mask_bytes": mask_bytes_written,
        "peak_accounted_cpu_bytes": peak_accounted_cpu_bytes,
        "available_memory_bytes_after": _available_memory_bytes(),
        "cuda_peak": _cuda_peaks(devices),
        "spearman_mode": "exact" if spearman_sample_size is None else "sampled",
        "transient_mask_directory": str(mask_dir),
        "mask_overall_sha256": overall_hashes,
        "evaluation_started": False,
    }
    _atomic_write_json(root / "logs" / "full_score.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="DLM aggregation ablation")
    parser.add_argument(
        "command", choices=("manifest", "smoke", "profile", "score")
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = validate_config(load_config(args.config))
    root = args.config.parent
    manifest_path = root / "calibration_manifest.json"
    if args.command == "manifest":
        result = generate_manifest(config, manifest_path)
        summary = {
            "state_count": len(result["states"]),
            "historical_state_sha256": result["historical_state_sha256"],
            "path": str(manifest_path),
        }
    elif args.command == "smoke":
        summary = run_smoke(config, manifest_path, root / "logs" / "smoke.json")
    elif args.command == "profile":
        summary = run_profile(
            config,
            manifest_path,
            root / "diagnostics" / "spearman_profile.json",
        )
    else:
        summary = run_full_scoring(config, manifest_path, root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
