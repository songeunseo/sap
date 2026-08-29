import argparse
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from cgq_sparsegpt import (
    _load_model,
    _write_json,
    build_corrupted_states,
    cache_original_and_inverse_weights,
    compare_model_masks,
    prepare_calibration_bundles,
    resolve_mask_id,
    sample_disjoint_article_spans,
    save_plain_model_masks,
    state_digest,
    summarize_mask_cells,
    top_level_article_token_ranges,
)
from cgq_sparsegpt_downstream import (
    _evaluate_winogrande,
    _load_existing_dense,
    _prune,
    _release,
    _write_jsonl,
    assert_same_examples,
    summarize_prediction_contrast,
    validate_frozen_config,
)


def validate_config(config):
    calibration = config["calibration"]
    validate_frozen_config(
        {
            "calibration": {
                "sequence_length": calibration["sequence_length"],
                "state_count": calibration["sequence_count"]
                * len(calibration["timesteps"]),
                "timesteps": calibration["timesteps"],
                "seed": calibration["reference_content_seed"],
            },
            "pruning": config["pruning"],
            "evaluation": config["evaluation"],
        }
    )
    bundles = calibration["bundles"]
    if len(bundles) != 3:
        raise ValueError("pilot requires exactly three new bundles")
    if len({bundle["content_seed"] for bundle in bundles}) != len(bundles):
        raise ValueError("content seeds must be distinct")
    if len({bundle["mask_seed"] for bundle in bundles}) != len(bundles):
        raise ValueError("mask seeds must be distinct")


def prepare_bundles(config, output_dir):
    validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = config["calibration"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    data = load_dataset(
        calibration["dataset"],
        calibration["subset"],
        split=calibration["split"],
    )
    rows = data["text"]
    encoded = tokenizer(" ".join(rows), return_offsets_mapping=True)
    full_ids = torch.tensor(encoded["input_ids"], dtype=torch.long).reshape(-1)
    article_ranges = top_level_article_token_ranges(rows, encoded["offset_mapping"])
    reference_spans = sample_disjoint_article_spans(
        article_ranges,
        full_ids.numel(),
        calibration["sequence_length"],
        calibration["reference_span_count"],
        calibration["reference_content_seed"],
    )
    rng = random.Random(calibration["reference_content_seed"])
    legacy_starts = [
        rng.randint(0, full_ids.numel() - calibration["sequence_length"] - 1)
        for _ in range(calibration["reference_span_count"])
    ]
    if [span["token_start"] for span in reference_spans] != legacy_starts:
        raise RuntimeError("reference spans no longer reproduce the legacy loader")
    reference_clean = [
        full_ids[span["token_start"] : span["token_end"]]
        .reshape(1, calibration["sequence_length"])
        .clone()
        for span in reference_spans
    ]
    reference_result = json.loads(
        Path(config["source_experiment"]["results_path"]).read_text()
    )
    reference_calibration = build_corrupted_states(
        reference_clean[: calibration["sequence_count"]],
        calibration["timesteps"],
        calibration["mask_id"],
        calibration["reference_mask_seed"],
    )
    reference_heldout = build_corrupted_states(
        reference_clean[calibration["sequence_count"] :],
        calibration["timesteps"],
        calibration["mask_id"],
        calibration["reference_heldout_mask_seed"],
    )
    if state_digest(reference_calibration) != reference_result["calibration_state_digest"]:
        raise RuntimeError("regenerated discovery calibration states differ")
    if state_digest(reference_heldout) != reference_result["heldout_state_digest"]:
        raise RuntimeError("regenerated held-out states differ")

    bundles = prepare_calibration_bundles(
        full_ids,
        article_ranges,
        calibration["bundles"],
        calibration["sequence_count"],
        calibration["sequence_length"],
        calibration["timesteps"],
        calibration["mask_id"],
        {span["article_index"] for span in reference_spans},
    )
    all_articles = [span["article_index"] for span in reference_spans]
    all_articles.extend(
        span["article_index"] for bundle in bundles for span in bundle["spans"]
    )
    if len(all_articles) != len(set(all_articles)):
        raise RuntimeError("clean spans reuse an article")
    all_hashes = [hashlib.sha256(ids.numpy().tobytes()).hexdigest() for ids in reference_clean]
    all_hashes.extend(digest for bundle in bundles for digest in bundle["clean_hashes"])
    if len(all_hashes) != len(set(all_hashes)):
        raise RuntimeError("clean spans contain an exact duplicate")

    payload = {
        "reference_spans": reference_spans,
        "reference_clean_hashes": all_hashes[: calibration["reference_span_count"]],
        "bundles": bundles,
    }
    torch.save(payload, output_dir / "bundles.pt")
    manifest = {
        "corpus_token_count": full_ids.numel(),
        "article_count": len(article_ranges),
        "article_aligned_256_token_capacity": sum(
            (end - start) // calibration["sequence_length"]
            for start, end in article_ranges
        ),
        "all_20_articles_are_distinct": len(all_articles) == len(set(all_articles)),
        "reference_spans": reference_spans,
        "reference_clean_hashes": payload["reference_clean_hashes"],
        "bundles": [
            {
                key: value
                for key, value in bundle.items()
                if key not in {"clean_ids", "states"}
            }
            for bundle in bundles
        ],
    }
    _write_json(output_dir / "bundle_manifest.json", manifest)
    return payload, manifest


def run_bundle(config, bundle, dense_records, tokenizer, output_dir):
    output_dir = Path(output_dir) / bundle["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "results.json"
    if complete_path.exists():
        return json.loads(complete_path.read_text())
    states = bundle["states"]
    digest = state_digest(states)
    if digest != bundle["state_digest"]:
        raise RuntimeError("bundle state digest differs")
    loader = [(state["input_ids"],) for state in states]
    args = SimpleNamespace(
        nsamples=len(states),
        seed=bundle["content_seed"],
        sparsity_ratio=config["pruning"]["sparsity"],
    )
    device = torch.device("cuda:0")
    confidence_model = _load_model(config)
    mask_id = resolve_mask_id(tokenizer, confidence_model)
    if mask_id != config["calibration"]["mask_id"]:
        raise RuntimeError("model mask token differs from the frozen protocol")
    original_weights, inverse_weights, _ = cache_original_and_inverse_weights(
        confidence_model, states, mask_id, device
    )
    del confidence_model
    _release()
    energy_error = max(
        (left.square().sum() - right.square().sum()).abs().item()
        for left, right in zip(original_weights, inverse_weights)
    )

    methods = {}
    records = {}
    masks = {}
    mask_dir = Path(
        tempfile.mkdtemp(prefix=f"cgq-seed-pilot-{bundle['name']}-", dir="/dev/shm")
    )
    try:
        for name, weights in (
            ("plain", None),
            ("cgq", original_weights),
            ("inverse", inverse_weights),
        ):
            if state_digest(states) != digest:
                raise RuntimeError("bundle states changed between methods")
            model = _load_model(config)
            pruning_seconds = _prune(model, args, loader, weights)
            if name == "plain":
                sparsity = save_plain_model_masks(model, mask_dir)
            else:
                sparsity, cells = compare_model_masks(model, mask_dir)
                masks[name] = summarize_mask_cells(cells)
            evaluation, method_records = _evaluate_winogrande(
                model, tokenizer, device
            )
            assert_same_examples(dense_records, method_records)
            methods[name] = {
                "pruning": sparsity,
                "pruning_seconds": pruning_seconds,
                "winogrande": evaluation,
            }
            records[name] = method_records
            _write_json(output_dir / f"{name}.json", methods[name])
            _write_jsonl(output_dir / f"{name}_predictions.jsonl", method_records)
            del model
            _release()
    finally:
        shutil.rmtree(mask_dir)

    result = {
        "bundle": {
            key: value
            for key, value in bundle.items()
            if key not in {"clean_ids", "states"}
        },
        "inverse_energy_max_absolute_error": energy_error,
        "methods": methods,
        "mask_xor_vs_plain": masks,
        "contrasts": {
            "cgq_minus_plain": summarize_prediction_contrast(
                records["plain"], records["cgq"]
            ),
            "inverse_minus_plain": summarize_prediction_contrast(
                records["plain"], records["inverse"]
            ),
            "inverse_minus_cgq": summarize_prediction_contrast(
                records["cgq"], records["inverse"]
            ),
        },
    }
    _write_json(complete_path, result)
    return result


def run_pilot(config_path, output_dir):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    output_dir = Path(output_dir)
    bundle_path = output_dir / "bundles.pt"
    if bundle_path.exists():
        payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    else:
        payload, _ = prepare_bundles(config, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    dense, dense_records = _load_existing_dense(config)
    results = [
        run_bundle(config, bundle, dense_records, tokenizer, output_dir)
        for bundle in payload["bundles"]
    ]
    summary = {
        "model": config["model"],
        "dense_winogrande": dense,
        "bundle_count": len(results),
        "results": results,
    }
    _write_json(output_dir / "pilot_results.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="codex/cgq_sparsegpt_seed_pilot/config.json"
    )
    parser.add_argument(
        "--output-dir", default="codex/cgq_sparsegpt_seed_pilot/results"
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.prepare_only:
        prepare_bundles(config, args.output_dir)
    else:
        run_pilot(args.config, args.output_dir)


if __name__ == "__main__":
    main()
