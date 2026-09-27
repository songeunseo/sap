import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers

from experiments.dlm_loss_aggregation.exp004.run import (
    allow_repeated_compiled_backwards,
    symmetric_token_weights,
    validate_partition_artifact,
    weighted_dlm_losses,
)
from experiments.dlm_loss_aggregation.run import (
    _load_model,
    load_config,
    require_historical_digest,
    validate_config,
)
from experiments.mlp_token_contrast_diagnostic.core import (
    abs_scores,
    contrast_scores,
    gate_directional_derivative,
    mirror_from_uniform_contrast,
    rank_comparison,
    ratio_summary,
    symmetry_error,
    stability_summary,
    timestep_scores,
)


ROOT = Path(__file__).parent
STATE_PATH = ROOT / "state_gate_derivatives.pt"
MANIFEST_PATH = ROOT / "artifact_manifest.json"
EXP001_CONFIG = Path("experiments/dlm_loss_aggregation/config.yaml")
STATE_MANIFEST = Path("experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json")
PARTITION_PATH = Path("experiments/dlm_loss_aggregation/exp004/token_partition_summary.json")
EXPECTED_STATE_SHA = "1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df"
EXPECTED_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
SYMMETRY_GLOBAL_RELATIVE_LIMIT = 0.025


def _file_sha256(path):
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


def _load_sources():
    config = load_config(EXP001_CONFIG)
    validate_config(config)
    if config["model"]["revision"] != EXPECTED_REVISION:
        raise ValueError("model revision differs from the frozen setup")
    manifest = json.loads(STATE_MANIFEST.read_text())
    require_historical_digest(manifest, EXPECTED_STATE_SHA)
    partition = json.loads(PARTITION_PATH.read_text())
    partition_validation = validate_partition_artifact(partition, manifest, expected_count=80)
    return config, manifest, partition, partition_validation


def _alphas(state, partition_state, device):
    mask = torch.tensor(state["mask"], dtype=torch.bool, device=device)
    reveal = torch.zeros_like(mask)
    reveal[0, partition_state["reveal_indices"]] = True
    reveal_alpha, remain_alpha = symmetric_token_weights(mask, reveal, rho=0.5)
    uniform = mask.float()
    if not torch.isclose(reveal_alpha[mask].mean(), torch.tensor(1.0, device=device), atol=1e-6, rtol=0):
        raise RuntimeError("SYM-REVEAL masked mean is not one")
    if not torch.isclose(remain_alpha[mask].mean(), torch.tensor(1.0, device=device), atol=1e-6, rtol=0):
        raise RuntimeError("SYM-REMAIN masked mean is not one")
    if not torch.allclose(reveal_alpha[mask] + remain_alpha[mask], 2 * uniform[mask], atol=1e-6, rtol=0):
        raise RuntimeError("token weights are not exactly symmetric")
    return mask, {
        "uniform": uniform,
        "reveal": reveal_alpha,
        "remain": remain_alpha,
        "contrast": (reveal_alpha - remain_alpha) / 2,
    }


def collect_state_derivatives(model, manifest, partition):
    blocks = model.model.transformer.blocks
    captured = [None] * len(blocks)
    handles = []
    for layer, block in enumerate(blocks):
        def capture(_, inputs, __, layer_index=layer):
            captured[layer_index] = inputs[0]
        handles.append(block.ff_out.register_forward_hook(capture))
    device = model.model.transformer.wte.weight.device
    values = {
        name: torch.empty((80, len(blocks), model.model.config.mlp_hidden_size), dtype=torch.float32)
        for name in ("uniform", "reveal", "remain")
    }
    symmetry = {"max_absolute": 0.0, "max_relative": 0.0, "max_global_relative": 0.0,
                "loss_max_absolute": 0.0, "loss_max_relative": 0.0}
    try:
        for state_index, (state, partition_state) in enumerate(zip(manifest["states"], partition["states"])):
            captured[:] = [None] * len(blocks)
            noisy = torch.tensor(state["noisy_ids"], dtype=torch.long, device=device)
            clean = torch.tensor(state["clean_ids"], dtype=torch.long, device=device)
            mask, alphas = _alphas(state, partition_state, device)
            logits = model(noisy).logits
            if any(hidden is None for hidden in captured):
                raise RuntimeError("an MLP ff_out input hook did not fire")
            losses = weighted_dlm_losses(logits, clean, mask, state["p_mask"], alphas)
            loss_residual = (losses["reveal"] + losses["remain"] - 2 * losses["uniform"]).abs().item()
            loss_scale = max(abs(losses["reveal"].item()), abs(losses["remain"].item()),
                             2 * abs(losses["uniform"].item()), 1e-30)
            symmetry["loss_max_absolute"] = max(symmetry["loss_max_absolute"], loss_residual)
            symmetry["loss_max_relative"] = max(symmetry["loss_max_relative"], loss_residual / loss_scale)
            gradients = {}
            basis_effects = {}
            for condition in ("uniform", "contrast"):
                gradients[condition] = torch.autograd.grad(
                    losses[condition], captured, retain_graph=condition != "contrast",
                    create_graph=False, allow_unused=False,
                )
                for layer, (hidden, gradient) in enumerate(zip(captured, gradients[condition])):
                    basis_effects.setdefault(condition, torch.empty_like(values["uniform"][state_index]))[layer] = (
                        gate_directional_derivative(hidden, gradient).cpu()
                    )
            values["uniform"][state_index] = basis_effects["uniform"]
            reveal_effect, remain_effect = mirror_from_uniform_contrast(
                basis_effects["uniform"], basis_effects["contrast"]
            )
            values["reveal"][state_index] = reveal_effect
            values["remain"][state_index] = remain_effect
            residual = values["reveal"][state_index] + values["remain"][state_index] - 2 * values["uniform"][state_index]
            reference = torch.maximum(
                torch.maximum(values["reveal"][state_index].abs(), values["remain"][state_index].abs()),
                2 * values["uniform"][state_index].abs(),
            )
            check = symmetry_error(
                values["uniform"][state_index], values["reveal"][state_index], values["remain"][state_index]
            )
            global_relative = residual.abs().max().item() / reference.max().clamp_min(1e-30).item()
            symmetry["max_absolute"] = max(symmetry["max_absolute"], check["max_absolute"])
            symmetry["max_relative"] = max(symmetry["max_relative"], check["max_relative"])
            symmetry["max_global_relative"] = max(symmetry["max_global_relative"], global_relative)
            if global_relative > SYMMETRY_GLOBAL_RELATIVE_LIMIT:
                raise RuntimeError(
                    f"gate-gradient symmetry failed at state {state_index}: "
                    f"global_relative={global_relative}, max_absolute={residual.abs().max().item()}, "
                    f"reference_max={reference.max().item()}, loss_absolute={loss_residual}"
                )
            print(f"state {state_index + 1}/80 symmetry_global_relative={global_relative:.6g}", flush=True)
            del noisy, clean, logits, losses, gradients
    finally:
        for handle in handles:
            handle.remove()
    return values, symmetry


def _summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    q = torch.quantile(values, torch.tensor([.01, .1, .5, .9, .99], dtype=torch.float64))
    mean = values.mean().item()
    std = values.std(unbiased=False).item()
    return {"mean": mean, "median": q[2].item(), "std": std, "cv": std / mean if mean else None,
            "p01": q[0].item(), "p10": q[1].item(), "p50": q[2].item(),
            "p90": q[3].item(), "p99": q[4].item(), "min": values.min().item(), "max": values.max().item()}


def _ratio(numerator, denominator):
    return ratio_summary(numerator, denominator, relative_floor=1e-6)


def _pair_stats(scores):
    return {
        "uniform_vs_reveal": rank_comparison(scores["uniform"], scores["reveal"]),
        "uniform_vs_remain": rank_comparison(scores["uniform"], scores["remain"]),
        "reveal_vs_remain": rank_comparison(scores["reveal"], scores["remain"]),
    }


def analyze(payload):
    derivatives = payload["derivatives"]
    timestep_index = payload["timestep_index"]
    sequence_index = payload["sequence_index"]
    layer_rows = []
    timestep_rows = []
    for layer in range(32):
        state = {name: value[:, layer] for name, value in derivatives.items()}
        scores = abs_scores(state["uniform"], state["reveal"], state["remain"])
        contrast = contrast_scores(state["uniform"], state["reveal"], state["remain"])
        contrast_rank = rank_comparison(scores["uniform"], contrast["score"])
        layer_row = {
            "layer": layer,
            "secondary_abs_rankings": _pair_stats(scores),
            "score_distributions": {name: _summary(value) for name, value in {**scores, "contrast": contrast["score"]}.items()},
            "ratios": {
                "q_reveal": _ratio(scores["reveal"], scores["uniform"]),
                "q_remain": _ratio(scores["remain"], scores["uniform"]),
                "relative_contrast": _ratio(contrast["score"], scores["uniform"]),
                "relative_contrast_floor_checks": {
                    str(floor): ratio_summary(contrast["score"], scores["uniform"], relative_floor=floor)
                    for floor in (1e-8, 1e-6, 1e-4)
                },
            },
            "uniform_vs_contrast": contrast_rank,
        }
        sequence_contrast = torch.stack([
            contrast["signed"][sequence_index == index].abs().mean(dim=0) for index in range(8)
        ])
        sequence_uniform = torch.stack([
            state["uniform"][sequence_index == index].abs().mean(dim=0) for index in range(8)
        ])
        sequence_relative = sequence_contrast / sequence_uniform.clamp_min(scores["uniform"].max() * 1e-6)
        global_relative = contrast["score"] / scores["uniform"].clamp_min(scores["uniform"].max() * 1e-6)
        layer_row["contrast_stability"] = {
            "sequence": {
                "absolute_contrast": stability_summary(sequence_contrast, contrast["score"]),
                "relative_contrast": stability_summary(sequence_relative, global_relative),
            },
        }
        by_timestep = {name: timestep_scores(value, timestep_index) for name, value in state.items()}
        timestep_contrast = {}
        for t in range(10):
            local_scores = {name: by_timestep[name][t] for name in by_timestep}
            local_contrast = (state["reveal"][timestep_index == t] - state["remain"][timestep_index == t]).div(2).abs().mean(dim=0)
            timestep_contrast[t] = local_contrast
            timestep_rows.append({
                "layer": layer, "timestep_index": t, "timestep": 0.05 + 0.1 * t,
                "uniform_vs_reveal_spearman": rank_comparison(local_scores["uniform"], local_scores["reveal"], fractions=(.1,))["spearman"],
                "uniform_vs_remain_spearman": rank_comparison(local_scores["uniform"], local_scores["remain"], fractions=(.1,))["spearman"],
                "reveal_vs_remain_spearman": rank_comparison(local_scores["reveal"], local_scores["remain"], fractions=(.1,))["spearman"],
                "q_reveal_cv": _ratio(local_scores["reveal"], local_scores["uniform"])["cv"],
                "q_remain_cv": _ratio(local_scores["remain"], local_scores["uniform"])["cv"],
                "relative_contrast_cv": _ratio(local_contrast, local_scores["uniform"])["cv"],
            })
        timestep_contrast_stack = torch.stack([timestep_contrast[t] for t in range(10)])
        timestep_uniform_stack = torch.stack([by_timestep["uniform"][t] for t in range(10)])
        timestep_relative = timestep_contrast_stack / timestep_uniform_stack.clamp_min(scores["uniform"].max() * 1e-6)
        layer_row["contrast_stability"]["timestep"] = {
            "absolute_contrast": stability_summary(timestep_contrast_stack, contrast["score"]),
            "relative_contrast": stability_summary(timestep_relative, global_relative),
        }
        layer_rows.append(layer_row)
    return layer_rows, timestep_rows


def _aggregate_layers(rows):
    paths = {
        "uniform_reveal_spearman": lambda row: row["secondary_abs_rankings"]["uniform_vs_reveal"]["spearman"],
        "uniform_remain_spearman": lambda row: row["secondary_abs_rankings"]["uniform_vs_remain"]["spearman"],
        "reveal_remain_spearman": lambda row: row["secondary_abs_rankings"]["reveal_vs_remain"]["spearman"],
        "q_reveal_cv": lambda row: row["ratios"]["q_reveal"]["cv"],
        "q_remain_cv": lambda row: row["ratios"]["q_remain"]["cv"],
        "relative_contrast_cv": lambda row: row["ratios"]["relative_contrast"]["cv"],
        "uniform_contrast_spearman": lambda row: row["uniform_vs_contrast"]["spearman"],
        "contrast_sequence_stability": lambda row: row["contrast_stability"]["sequence"]["absolute_contrast"]["spearman"]["mean"],
        "relative_contrast_sequence_stability": lambda row: row["contrast_stability"]["sequence"]["relative_contrast"]["spearman"]["mean"],
        "contrast_timestep_stability": lambda row: row["contrast_stability"]["timestep"]["absolute_contrast"]["spearman"]["mean"],
        "relative_contrast_timestep_stability": lambda row: row["contrast_stability"]["timestep"]["relative_contrast"]["spearman"]["mean"],
    }
    return {name: _summary([getter(row) for row in rows]) for name, getter in paths.items()}


def main(reuse=False):
    started = time.time()
    config, manifest, partition, partition_validation = _load_sources()
    model, devices = _load_model(config)
    if len(model.model.transformer.blocks) != 32 or model.model.config.mlp_hidden_size != 12288:
        raise RuntimeError("unexpected MLP architecture")
    allow_repeated_compiled_backwards()
    ROOT.mkdir(parents=True, exist_ok=True)
    weight_before = _model_sha256(model)
    if reuse:
        payload = torch.load(STATE_PATH, map_location="cpu", weights_only=True)
        symmetry = payload["symmetry"]
        weight_after_collection = weight_before
    else:
        derivatives, symmetry = collect_state_derivatives(model, manifest, partition)
        weight_after_collection = _model_sha256(model)
        if weight_after_collection != weight_before:
            raise RuntimeError("model weights changed during derivative collection")
        payload = {
            "version": 1,
            "source_state_sha256": EXPECTED_STATE_SHA,
            "partition_sha256": partition["sha256"],
            "derivatives": derivatives,
            "timestep_index": torch.tensor([row["timestep_index"] for row in manifest["states"]]),
            "sequence_index": torch.tensor([row["sequence_index"] for row in manifest["states"]]),
            "symmetry": symmetry,
            "definition": "d=-sum_position h*grad(loss,h); FP32 product/sum; independently collected U/R/M",
        }
        temporary = STATE_PATH.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(STATE_PATH)
        payload = torch.load(STATE_PATH, map_location="cpu", weights_only=True)
    if payload["source_state_sha256"] != EXPECTED_STATE_SHA:
        raise RuntimeError("state derivative artifact source mismatch")
    if symmetry["max_global_relative"] > SYMMETRY_GLOBAL_RELATIVE_LIMIT:
        raise RuntimeError("saved derivative symmetry exceeds the predeclared numerical limit")
    layer_rows, timestep_rows = analyze(payload)
    weight_after_analysis = _model_sha256(model)
    if weight_after_analysis != weight_before:
        raise RuntimeError("model weights changed during analysis")
    result = {
        "status": "complete", "analysis_only": True,
        "cgq_used": False, "wanda_used": False, "sparsegpt_used": False,
        "mask_generated": False, "pruning_performed": False, "downstream_evaluation_performed": False,
        "source": {"model": config["model"], "run_id": "20260828T175010-2484545", "state_count": 80,
                   "state_sha256": EXPECTED_STATE_SHA, "partition_path": str(PARTITION_PATH),
                   "partition_sha256": partition["sha256"], "partition_validation": partition_validation},
        "gate": {"location": "input to each block.ff_out after silu(ff_proj(ff_norm(x))) * up_proj(ff_norm(x))",
                 "definition": "d[l,k,s] = -sum_p h[l,p,k] * dL_s/dh[l,p,k]", "trainable_parameter_added": False},
        "loss": {"normalization": "sum_masked(alpha*CE)/(p_mask*256)", "rho": 0.5,
                 "uniform": "alpha=1", "symmetric_reveal": "alpha=1+0.5*c",
                 "symmetric_remain": "alpha=1-0.5*c", "contrast": "d_C=(d_R-d_M)/2=d_R-d_U"},
        "symmetry_sanity": {**symmetry, "global_relative_failure_limit": SYMMETRY_GLOBAL_RELATIVE_LIMIT,
                            "interpretation": "construction sanity check only; not empirical specialization evidence"},
        "aggregation": {"primary_baseline": "S_U=mean_state(abs(d_U))", "primary_contrast": "S_C=mean_state(abs(d_C))",
                        "relative_contrast": "R_C=S_C/(S_U+safe handling)",
                        "secondary": "S_R/S_M ABS rankings are descriptive only"},
        "layer_summary": _aggregate_layers(layer_rows), "layers": layer_rows, "timesteps": timestep_rows,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
                        "devices": list(devices), "gpu": torch.cuda.get_device_name(devices[0])},
        "integrity": {"weight_sha256_before": weight_before, "weight_sha256_after_collection": weight_after_collection,
                      "weight_sha256_after_analysis": weight_after_analysis, "state_derivative_path": str(STATE_PATH.resolve()),
                      "state_derivative_sha256": _file_sha256(STATE_PATH)},
        "runtime_seconds": time.time() - started,
    }
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST_PATH)
    print(json.dumps({"status": "complete", "manifest": str(MANIFEST_PATH), "seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true")
    main(reuse=parser.parse_args().reuse)
