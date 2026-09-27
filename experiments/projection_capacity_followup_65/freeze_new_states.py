#!/usr/bin/env python3
"""Freeze a third disjoint WikiText-2 DLM state split before comparator evaluation."""
import copy
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from model import LLaDAConfig
from lib.data import get_loaders
from experiments.dlm_loss_aggregation.run import (
    build_calibration_manifest,
    historical_state_digest,
    load_config,
)
from experiments.projection_capacity_followup_65.core import interval_overlaps


ROOT = Path("experiments/projection_capacity_followup_65")
OUTPUT = ROOT / "new_heldout_state_manifest.json"
VERIFICATION = ROOT / "new_state_verification.json"
PRIOR = [
    Path("experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json"),
    Path("experiments/wanda_failure_characterization/heldout_state_manifest.json"),
]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals_for(document, corpus_tokens):
    clean = {int(row["sequence_index"]): row["clean_ids"][0] for row in document["states"]}
    rng = random.Random(document["global_random_seed"])
    rows = []
    for sequence_index in sorted(clean):
        start = rng.randint(0, len(corpus_tokens) - 256 - 1)
        if corpus_tokens[start:start + 256].tolist() != clean[sequence_index]:
            raise RuntimeError(f"cannot reconstruct corpus interval for sequence {sequence_index}")
        rows.append({"sequence_index": sequence_index, "start": start,
                     "end_exclusive": start + 256})
    return rows


def main():
    base = copy.deepcopy(load_config("experiments/dlm_loss_aggregation/config.yaml"))
    base["calibration"].update({
        "seed": 2,
        "sequence_count": 8,
        "sequence_indices": list(range(16, 24)),
        "timesteps": [0.1, 0.3, 0.5, 0.7, 0.9],
    })
    model_config = LLaDAConfig.from_pretrained(
        base["model"]["id"], revision=base["model"]["revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base["model"]["id"], revision=base["model"]["revision"], trust_remote_code=True
    )
    loader, _ = get_loaders(
        "wikitext2", nsamples=8, seed=2, seqlen=256, tokenizer=tokenizer
    )
    document = build_calibration_manifest(
        [sample[0] for sample in loader], model_config.mask_token_id, base
    )
    document.update({
        "mask_id": model_config.mask_token_id,
        "historical_state_sha256": historical_state_digest(document),
        "frozen_before_comparator_evaluation": True,
        "purpose": "fresh confirmatory held-out evaluation for oracle explanation controls",
    })

    corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    corpus_tokens = tokenizer(" ".join(corpus["text"]), return_tensors="pt").input_ids[0]
    prior_documents = [json.loads(path.read_text()) for path in PRIOR]
    prior_intervals = [row for old in prior_documents for row in intervals_for(old, corpus_tokens)]
    new_intervals = intervals_for(document, corpus_tokens)
    overlaps = interval_overlaps(prior_intervals, new_intervals)
    if overlaps:
        raise RuntimeError(f"new held-out corpus spans overlap prior splits: {overlaps}")

    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text())
        if existing != document:
            raise RuntimeError("attempt to alter frozen new held-out states")
    else:
        write_json(OUTPUT, document)
    verification = {
        "status": "verified",
        "manifest_sha256": file_sha(OUTPUT),
        "state_sha256": document["historical_state_sha256"],
        "states": len(document["states"]),
        "sequence_indices": list(range(16, 24)),
        "timesteps": [0.1, 0.3, 0.5, 0.7, 0.9],
        "corpus_token_count": len(corpus_tokens),
        "prior_manifest_sha256": {str(path): file_sha(path) for path in PRIOR},
        "prior_intervals": prior_intervals,
        "new_intervals": new_intervals,
        "overlaps": overlaps,
        "all_three_splits_disjoint": True,
    }
    write_json(VERIFICATION, verification)
    print(json.dumps({key: verification[key] for key in
                      ("status", "states", "state_sha256", "all_three_splits_disjoint")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
