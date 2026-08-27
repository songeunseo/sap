import argparse
import gc
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from lib.dlm_gradient_sensitivity import make_masked_state
from lib.prune_llada import find_layers, prune_sparsegpt


def cgq_token_weights(input_ids, logits, mask_id):
    if logits.shape[:-1] != input_ids.shape:
        raise ValueError("logits and input_ids token shapes must match")
    confidence = logits.float().softmax(dim=-1).amax(dim=-1).detach()
    if not torch.isfinite(confidence).all() or not (
        (0 <= confidence) & (confidence <= 1)
    ).all():
        raise ValueError("confidence must be finite and in [0, 1]")
    mask_weight = torch.where(
        input_ids.eq(mask_id),
        torch.tensor(1.0, device=input_ids.device),
        torch.tensor(0.7, device=input_ids.device),
    )
    weights = mask_weight + confidence.sqrt()
    if not torch.isfinite(weights).all():
        raise ValueError("CGQ weights must be finite")
    return weights.detach(), confidence, mask_weight


def build_corrupted_states(clean_ids, timesteps, mask_id, seed, eps=1e-3):
    states = []
    for timestep_index, timestep in enumerate(timesteps):
        for sequence_index, ids in enumerate(clean_ids):
            mask_seed = seed + timestep_index * len(clean_ids) + sequence_index
            input_ids, mask, p_mask = make_masked_state(
                ids, timestep, mask_id, mask_seed, eps
            )
            states.append(
                {
                    "input_ids": input_ids.cpu(),
                    "mask": mask.cpu(),
                    "p_mask": p_mask,
                    "timestep": float(timestep),
                    "timestep_index": timestep_index,
                    "sequence_index": sequence_index,
                    "mask_seed": mask_seed,
                }
            )
    return states


def state_digest(states):
    digest = hashlib.sha256()
    for state in states:
        ids = state["input_ids"].detach().cpu().contiguous()
        digest.update(str(tuple(ids.shape)).encode())
        digest.update(str(ids.dtype).encode())
        digest.update(format(float(state["timestep"]), ".17g").encode())
        digest.update(ids.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def cache_dense_references(model, states, device):
    references = []
    for state in states:
        input_ids = state["input_ids"].to(device)
        mask = state["mask"].to(device)
        logits = model(input_ids).logits
        references.append(
            {
                "logits": logits[mask].detach().to(device="cpu", dtype=torch.float16),
                "timestep": state["timestep"],
            }
        )
    return references


def masked_logit_sums(dense_logits, sparse_logits):
    if dense_logits.shape != sparse_logits.shape or dense_logits.ndim != 2:
        raise ValueError("masked dense and sparse logits must have the same 2D shape")
    dense_logp = F.log_softmax(dense_logits.float(), dim=-1)
    sparse_logp = F.log_softmax(sparse_logits.float(), dim=-1)
    dense_probability = dense_logp.exp()
    dense_confidence, dense_prediction = dense_probability.max(dim=-1)
    sparse_confidence, sparse_prediction = sparse_logp.exp().max(dim=-1)
    return {
        "count": dense_logits.shape[0],
        "kl_sum": (
            dense_probability * (dense_logp - sparse_logp)
        ).sum(dtype=torch.float32).item(),
        "agreement_count": dense_prediction.eq(sparse_prediction).sum().item(),
        "confidence_abs_error_sum": dense_confidence.sub(
            sparse_confidence
        ).abs().sum(dtype=torch.float32).item(),
    }


def diagonal_change(plain, cgq):
    if plain.shape != cgq.shape:
        raise ValueError("Hessian diagonals must have the same shape")
    plain = plain.float()
    cgq = cgq.float()
    return {
        "cosine_similarity": F.cosine_similarity(plain, cgq, dim=0).item(),
        "relative_l2_difference": (
            torch.linalg.vector_norm(cgq - plain)
            / torch.linalg.vector_norm(plain)
        ).item(),
    }


def _group_statistics(states, confidence, weights, indices):
    masks = torch.cat([states[index]["mask"].reshape(-1) for index in indices])
    confidence_values = torch.cat([confidence[index].reshape(-1) for index in indices])
    weight_values = torch.cat([weights[index].reshape(-1) for index in indices])

    def side(selected):
        selected_confidence = confidence_values[selected].float()
        selected_weights = weight_values[selected].float()
        result = {"count": selected.sum().item()}
        for name, values in (
            ("confidence", selected_confidence),
            ("r", selected_weights),
        ):
            if values.numel() == 0:
                result.update(
                    {
                        f"{name}_mean": None,
                        f"{name}_std": None,
                        f"{name}_min": None,
                        f"{name}_max": None,
                    }
                )
            else:
                result.update(
                    {
                        f"{name}_mean": values.mean().item(),
                        f"{name}_std": values.std(unbiased=False).item(),
                        f"{name}_min": values.min().item(),
                        f"{name}_max": values.max().item(),
                    }
                )
        return result

    return {
        "token_count": masks.numel(),
        "masked_ratio": masks.float().mean().item(),
        "masked": side(masks),
        "unmasked": side(~masks),
    }


def calibration_statistics(states, confidence, weights):
    if not (len(states) == len(confidence) == len(weights)) or not states:
        raise ValueError("states, confidence, and weights must have equal positive counts")
    timesteps = sorted({state["timestep"] for state in states})
    by_timestep = []
    for timestep in timesteps:
        indices = [
            index for index, state in enumerate(states)
            if state["timestep"] == timestep
        ]
        by_timestep.append(
            {"timestep": timestep, **_group_statistics(states, confidence, weights, indices)}
        )
    return {
        "overall": _group_statistics(states, confidence, weights, range(len(states))),
        "by_timestep": by_timestep,
    }


def write_packed_mask(path, mask):
    flat = mask.detach().bool().cpu().numpy().reshape(-1)
    np.packbits(flat, bitorder="little").tofile(path)


def compare_packed_mask(path, mask):
    flat = mask.detach().bool().cpu().numpy().reshape(-1)
    current = np.packbits(flat, bitorder="little")
    plain = np.fromfile(path, dtype=np.uint8)
    if current.shape != plain.shape:
        raise ValueError("packed mask size mismatch")
    xor_count = int(np.unpackbits(np.bitwise_xor(plain, current)).sum())
    return {
        "xor_count": xor_count,
        "total_count": int(flat.size),
        "xor_fraction": xor_count / flat.size,
    }


def summarize_mask_cells(cells):
    total_xor = sum(cell["xor_count"] for cell in cells)
    total_count = sum(cell["total_count"] for cell in cells)
    blocks = sorted({cell["block"] for cell in cells})
    layer_fractions = []
    for block in blocks:
        block_cells = [cell for cell in cells if cell["block"] == block]
        layer_fractions.append(
            sum(cell["xor_count"] for cell in block_cells)
            / sum(cell["total_count"] for cell in block_cells)
        )
    return {
        "global_xor_count": total_xor,
        "global_total_count": total_count,
        "global_xor_fraction": total_xor / total_count,
        "mean_layer_xor_fraction": sum(layer_fractions) / len(layer_fractions),
        "mean_module_xor_fraction": sum(
            cell["xor_fraction"] for cell in cells
        ) / len(cells),
        "cells": cells,
        "top_cells": sorted(cells, key=lambda cell: cell["xor_fraction"], reverse=True)[:10],
    }


def _model_modules(model):
    for block_index, block in enumerate(model.model.transformer.blocks):
        for module_name, layer in find_layers(block).items():
            yield block_index, module_name, layer


def _sparsity(zero_count, total_count):
    return {
        "zero_count": zero_count,
        "total_count": total_count,
        "sparsity": zero_count / total_count,
    }


def save_plain_model_masks(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    zero_count = 0
    total_count = 0
    for block_index, module_name, layer in _model_modules(model):
        mask = layer.weight.data.eq(0)
        write_packed_mask(directory / f"block-{block_index:03d}-{module_name}.bin", mask)
        zero_count += mask.sum().item()
        total_count += mask.numel()
    return _sparsity(zero_count, total_count)


def compare_model_masks(model, directory):
    directory = Path(directory)
    zero_count = 0
    total_count = 0
    cells = []
    for block_index, module_name, layer in _model_modules(model):
        mask = layer.weight.data.eq(0)
        cells.append(
            {
                "block": block_index,
                "module": module_name,
                **compare_packed_mask(
                    directory / f"block-{block_index:03d}-{module_name}.bin", mask
                ),
            }
        )
        zero_count += mask.sum().item()
        total_count += mask.numel()
    return _sparsity(zero_count, total_count), cells


def _empty_metric_sums():
    return {
        "count": 0,
        "kl_sum": 0.0,
        "agreement_count": 0,
        "confidence_abs_error_sum": 0.0,
    }


def _add_metric_sums(total, values):
    for key in total:
        total[key] += values[key]


def _finalize_metric_sums(values):
    count = values["count"]
    return {
        "masked_token_count": count,
        "kl": values["kl_sum"] / count,
        "prediction_agreement": values["agreement_count"] / count,
        "confidence_mae": values["confidence_abs_error_sum"] / count,
    }


@torch.no_grad()
def evaluate_heldout(model, states, references, device):
    if len(states) != len(references):
        raise ValueError("held-out states and dense references must have equal counts")
    overall = _empty_metric_sums()
    by_timestep = {}
    for state, reference in zip(states, references):
        if state["timestep"] != reference["timestep"]:
            raise ValueError("dense reference timestep mismatch")
        input_ids = state["input_ids"].to(device)
        mask = state["mask"].to(device)
        sparse_logits = model(input_ids).logits[mask]
        values = masked_logit_sums(reference["logits"].to(device), sparse_logits)
        _add_metric_sums(overall, values)
        timestep_total = by_timestep.setdefault(
            state["timestep"], _empty_metric_sums()
        )
        _add_metric_sums(timestep_total, values)
    return {
        "overall": _finalize_metric_sums(overall),
        "by_timestep": [
            {"timestep": timestep, **_finalize_metric_sums(by_timestep[timestep])}
            for timestep in sorted(by_timestep)
        ],
    }


EXPECTED_MODULES = {
    "q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"
}


def resolve_mask_id(tokenizer, model):
    tokenizer_id = getattr(tokenizer, "mask_token_id", None)
    model_id = getattr(model.config, "mask_token_id", None)
    if tokenizer_id is not None and model_id is not None and tokenizer_id != model_id:
        raise ValueError("model and tokenizer mask IDs disagree")
    mask_id = tokenizer_id if tokenizer_id is not None else model_id
    if mask_id is None:
        raise ValueError("model and tokenizer have no mask_token_id")
    return mask_id


def _write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _reference_digest(references):
    digest = hashlib.sha256()
    for reference in references:
        logits = reference["logits"].contiguous()
        digest.update(format(float(reference["timestep"]), ".17g").encode())
        digest.update(str(tuple(logits.shape)).encode())
        digest.update(logits.numpy().tobytes())
    return digest.hexdigest()


def _validate_model(model):
    blocks = model.model.transformer.blocks
    if len(blocks) != 32:
        raise ValueError(f"expected 32 transformer blocks, found {len(blocks)}")
    for block_index, block in enumerate(blocks):
        names = set(find_layers(block))
        if names != EXPECTED_MODULES:
            raise ValueError(
                f"block {block_index} modules {sorted(names)} do not match "
                f"{sorted(EXPECTED_MODULES)}"
            )


def _load_model(config):
    from main_llada import get_llm

    model = get_llm(
        config["model"]["id"],
        seqlen=config["calibration"]["sequence_length"],
        revision=config["model"]["revision"],
    )
    model.eval()
    _validate_model(model)
    return model


def _release_model():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def _cache_calibration_weights(model, states, mask_id, device):
    weights = []
    confidence = []
    for state in states:
        input_ids = state["input_ids"].to(device)
        state_weights, state_confidence, mask_weight = cgq_token_weights(
            input_ids, model(input_ids).logits, mask_id
        )
        mask = input_ids.eq(mask_id)
        if not mask_weight[mask].eq(1.0).all():
            raise AssertionError("masked token mask_weight is not 1.0")
        if not mask_weight[~mask].eq(0.7).all():
            raise AssertionError("unmasked token mask_weight is not 0.7")
        weights.append(state_weights.cpu())
        confidence.append(state_confidence.cpu())
    return weights, confidence


def _one_block_prune(model, args, loader, weights, hessian_diagonals, mask_dir, compare):
    blocks = model.model.transformer.blocks
    model.model.transformer.blocks = nn.ModuleList([blocks[0]])
    try:
        prune_sparsegpt(
            args,
            model,
            tokenizer=None,
            dev=torch.device("cuda:0"),
            calibration_loader=loader,
            token_weights=weights,
            hessian_diagonals=hessian_diagonals,
        )
        if compare:
            sparsity, cells = compare_model_masks(model, mask_dir)
            return sparsity, summarize_mask_cells(cells)
        return save_plain_model_masks(model, mask_dir), None
    finally:
        model.model.transformer.blocks = blocks


def _run_pruning(model, args, loader, weights, references, heldout_states):
    hessian_diagonals = {}
    started = time.monotonic()
    prune_sparsegpt(
        args,
        model,
        tokenizer=None,
        dev=torch.device("cuda:0"),
        calibration_loader=loader,
        token_weights=weights,
        hessian_diagonals=hessian_diagonals,
    )
    pruning_seconds = time.monotonic() - started
    heldout = evaluate_heldout(
        model, heldout_states, references, torch.device("cuda:0")
    )
    return hessian_diagonals, heldout, pruning_seconds


def run_experiment(config_path, output_dir):
    config = json.loads(Path(config_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    calibration = config["calibration"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    from lib.data import get_loaders

    clean_loader, _ = get_loaders(
        "wikitext2",
        nsamples=calibration["sequence_count"] + calibration["heldout_sequence_count"],
        seed=calibration["seed"],
        seqlen=calibration["sequence_length"],
        tokenizer=tokenizer,
    )
    clean = [sample[0].cpu() for sample in clean_loader]
    calibration_clean = clean[:calibration["sequence_count"]]
    heldout_clean = clean[calibration["sequence_count"]:]
    if {
        hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        for ids in calibration_clean
    } & {
        hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        for ids in heldout_clean
    }:
        raise ValueError("calibration and held-out clean sequences overlap")

    dense_model = _load_model(config)
    mask_id = resolve_mask_id(tokenizer, dense_model)
    calibration_states = build_corrupted_states(
        calibration_clean,
        calibration["timesteps"],
        mask_id,
        calibration["mask_seed"],
        calibration["epsilon"],
    )
    heldout_states = build_corrupted_states(
        heldout_clean,
        calibration["timesteps"],
        mask_id,
        calibration["heldout_mask_seed"],
        calibration["epsilon"],
    )
    calibration_digest = state_digest(calibration_states)
    heldout_digest = state_digest(heldout_states)
    torch.save(
        {"calibration": calibration_states, "heldout": heldout_states},
        output_dir / "states.pt",
    )
    loader = [(state["input_ids"],) for state in calibration_states]
    args = SimpleNamespace(
        nsamples=len(calibration_states),
        seed=calibration["seed"],
        sparsity_ratio=config["pruning"]["sparsity"],
    )

    references = cache_dense_references(
        dense_model, heldout_states, torch.device("cuda:0")
    )
    if len(references) != len(heldout_states):
        raise AssertionError("dense held-out reference cache is incomplete")
    reference_digest = _reference_digest(references)
    cgq_weights, confidence = _cache_calibration_weights(
        dense_model, calibration_states, mask_id, torch.device("cuda:0")
    )
    calibration_report = calibration_statistics(
        calibration_states, confidence, cgq_weights
    )
    print(json.dumps(calibration_report, indent=2, sort_keys=True))

    smoke_dir = Path(tempfile.mkdtemp(prefix="cgq-sparsegpt-smoke-", dir="/dev/shm"))
    try:
        smoke_plain_hessian = {}
        smoke_plain_sparsity, _ = _one_block_prune(
            dense_model, args, loader, None, smoke_plain_hessian, smoke_dir, False
        )
        del dense_model
        _release_model()
        ones_model = _load_model(config)
        smoke_ones_hessian = {}
        smoke_ones_sparsity, smoke_masks = _one_block_prune(
            ones_model,
            args,
            loader,
            [torch.ones_like(weight) for weight in cgq_weights],
            smoke_ones_hessian,
            smoke_dir,
            True,
        )
        hessian_equal = all(
            torch.equal(smoke_plain_hessian[name], smoke_ones_hessian[name])
            for name in smoke_plain_hessian
        )
        smoke_report = {
            "plain_sparsity": smoke_plain_sparsity,
            "ones_sparsity": smoke_ones_sparsity,
            "mask_comparison": smoke_masks,
            "hessian_diagonals_exact": hessian_equal,
        }
        del ones_model
        _release_model()
        if smoke_masks["global_xor_count"] != 0 or not hessian_equal:
            raise RuntimeError("all-ones SparseGPT control failed")
    finally:
        shutil.rmtree(smoke_dir)

    full_mask_dir = Path(tempfile.mkdtemp(prefix="cgq-sparsegpt-masks-", dir="/dev/shm"))
    try:
        if state_digest(calibration_states) != calibration_digest:
            raise RuntimeError("calibration states changed before Plain pruning")
        plain_model = _load_model(config)
        plain_hessian, plain_heldout, plain_seconds = _run_pruning(
            plain_model, args, loader, None, references, heldout_states
        )
        plain_sparsity = save_plain_model_masks(plain_model, full_mask_dir)
        plain_report = {
            "sparsity": plain_sparsity,
            "heldout": plain_heldout,
            "pruning_seconds": plain_seconds,
        }
        _write_json(output_dir / "plain.json", plain_report)
        del plain_model
        _release_model()

        if state_digest(calibration_states) != calibration_digest:
            raise RuntimeError("calibration states changed before CGQ pruning")
        cgq_model = _load_model(config)
        cgq_hessian, cgq_heldout, cgq_seconds = _run_pruning(
            cgq_model, args, loader, cgq_weights, references, heldout_states
        )
        cgq_sparsity, mask_cells = compare_model_masks(cgq_model, full_mask_dir)
        cgq_report = {
            "sparsity": cgq_sparsity,
            "heldout": cgq_heldout,
            "pruning_seconds": cgq_seconds,
        }
        _write_json(output_dir / "cgq.json", cgq_report)

        hessian_cells = [
            {"module": name, **diagonal_change(plain_hessian[name], cgq_hessian[name])}
            for name in sorted(plain_hessian)
        ]
        hessian_report = {
            "mean_diagonal_cosine_similarity": sum(
                cell["cosine_similarity"] for cell in hessian_cells
            ) / len(hessian_cells),
            "mean_relative_l2_difference": sum(
                cell["relative_l2_difference"] for cell in hessian_cells
            ) / len(hessian_cells),
            "cells": hessian_cells,
            "most_changed": sorted(
                hessian_cells,
                key=lambda cell: cell["relative_l2_difference"],
                reverse=True,
            )[:10],
        }
        mask_report = summarize_mask_cells(mask_cells)
        del cgq_model
        _release_model()
    finally:
        shutil.rmtree(full_mask_dir)

    report = {
        "model": config["model"],
        "config": config,
        "calibration_state_digest": calibration_digest,
        "heldout_state_digest": heldout_digest,
        "dense_reference_digest": reference_digest,
        "dense_reference_count": len(references),
        "dense_reference_dtype": str(references[0]["logits"].dtype),
        "calibration": calibration_report,
        "smoke": smoke_report,
        "plain": plain_report,
        "cgq": cgq_report,
        "mask_comparison": mask_report,
        "hessian_comparison": hessian_report,
        "total_seconds": time.monotonic() - started,
        "winogrande": "not run (excluded by approved experiment scope)",
    }
    _write_json(output_dir / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="codex/cgq_sparsegpt_2h/config.json"
    )
    parser.add_argument(
        "--output-dir", default="codex/cgq_sparsegpt_2h/results"
    )
    args = parser.parse_args()
    run_experiment(args.config, args.output_dir)


if __name__ == "__main__":
    main()
