"""Downstream attribution control for the one frozen DLMW and FG target masks."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cgq_sparsegpt_downstream import (
    _evaluate_winogrande,
    assert_same_examples as assert_same_wino,
)
from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask, unpack_mask
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _overall_mask_hash,
    _read_jsonl,
    _write_jsonl,
    _zero_mask_summary,
    load_config as load_exp002_config,
)
from experiments.dlm_loss_aggregation.run import _load_model, load_config as load_base_config
from experiments.fg_wanda_prototype1.attribution_core import (
    TARGET,
    attribution_decomposition,
    build_attribution_masks,
    paired_correctness,
    three_way_correctness,
)
from experiments.fg_wanda_prototype1.freeze import ROOT, STORE, sha, write_json
from experiments.fg_wanda_prototype1.gate3 import _load_gsm_references, _load_packed_masks
from experiments.fg_wanda_prototype1.gate3_core import apply_masks, raw_mask_sha256
from experiments.wanda_failure_characterization.run_failure_map import model_sha, modules


ATTR_ROOT = ROOT / "attribution"
ATTR_STORE = STORE / "attribution_masks" / "Ours-DLMW"
PREREG = ROOT / "attribution_preregistered.json"
EXP002 = Path("experiments/dlm_loss_aggregation/exp002")


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _canonical_json(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _protocol_hash(document):
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _verify_preregistered_sources():
    fixed = json.loads(PREREG.read_text())
    gate1 = json.loads((ROOT / "gate1.json").read_text())
    gate3 = json.loads((ROOT / "gate3_mask_manifest.json").read_text())
    if gate3["overall_sha256"]["Wanda"] != fixed["standard_wanda_mask_sha256"]:
        raise RuntimeError("Standard Wanda full-model mask hash mismatch")
    if gate1["mask_raw_hashes"]["DLMW"] != fixed["persisted_dlmw_target_raw_sha256"]:
        raise RuntimeError("persisted DLMW target hash mismatch")
    if gate1["mask_raw_hashes"]["FG"] != fixed["persisted_fg_target_raw_sha256"]:
        raise RuntimeError("persisted FG target hash mismatch")
    if sha(ROOT / "geometry_calibration_manifest.json") != fixed["geometry_calibration_manifest_sha256"]:
        raise RuntimeError("geometry calibration manifest hash mismatch")
    return fixed, gate1, gate3


def _records_audit(path, expected_hash=None, expected_count=None):
    path = Path(path)
    records = _read_jsonl(path)
    if expected_hash is not None and sha(path) != expected_hash:
        raise RuntimeError(f"prediction hash mismatch: {path}")
    if expected_count is not None and len(records) != expected_count:
        raise RuntimeError(f"prediction count mismatch: {path}")
    return records


@torch.no_grad()
def prepare():
    fixed, gate1, gate3_manifest = _verify_preregistered_sources()
    if json.loads((ROOT / "gate2.json").read_text())["status"] != "pass":
        raise RuntimeError("Prototype Gate 2 is not passed")

    standard, _ = _load_packed_masks("Wanda")
    dlmw = torch.load(STORE / "mask_DLMW.pt", map_location="cpu", weights_only=True)
    fg = torch.load(STORE / "mask_FG.pt", map_location="cpu", weights_only=True)
    if raw_mask_sha256(dlmw) != fixed["persisted_dlmw_target_raw_sha256"]:
        raise RuntimeError("DLMW tensor differs from persisted Gate-1 hash")
    if raw_mask_sha256(fg) != fixed["persisted_fg_target_raw_sha256"]:
        raise RuntimeError("FG tensor differs from persisted Gate-1 hash")
    ours, changed_w, changed_fg = build_attribution_masks(standard, dlmw, fg)

    target_xor = {
        "DLMW_vs_Wanda": float((dlmw != standard[TARGET]).float().mean()),
        "FG_vs_Wanda": float((fg != standard[TARGET]).float().mean()),
        "FG_vs_DLMW": float((fg != dlmw).float().mean()),
    }
    historical = {
        row["pair"]: row["global_mask_xor"]
        for row in json.loads((ROOT / "gate1_repository_exact.json").read_text())["comparisons"]
    }
    expected = {
        "DLMW_vs_Wanda": historical["DLMW_vs_WandaSequential"],
        "FG_vs_Wanda": historical["FG_vs_WandaSequential"],
        "FG_vs_DLMW": historical["FG_vs_DLMW"],
    }
    for name in target_xor:
        if abs(target_xor[name] - expected[name]) > 1e-8:
            raise RuntimeError(f"historical target XOR mismatch: {name}")

    entries = []
    for canonical_name, mask in sorted(ours.items()):
        block = int(canonical_name[6:8])
        module = canonical_name.split(".", 1)[1]
        payload = pack_mask(mask)
        path = ATTR_STORE / f"{canonical_name}.bin"
        _atomic_bytes(path, payload["bits"])
        entries.append(
            {
                "method": "Ours-DLMW",
                "block_index": block,
                "module": module,
                "canonical_name": canonical_name,
                "shape": list(mask.shape),
                "sha256": mask_sha256(payload),
                "raw_sha256": raw_mask_sha256(mask),
                "runtime_path": str(path),
                "pruned": int(mask.sum()),
                "weights": mask.numel(),
            }
        )
    overall = _overall_mask_hash(entries, "Ours-DLMW")
    pruned = sum(x["pruned"] for x in entries)
    weights = sum(x["weights"] for x in entries)
    if len(entries) != 224 or pruned * 2 != weights:
        raise RuntimeError("Ours-DLMW full mask count/sparsity mismatch")

    gsm = json.loads((ROOT / "gate3/gsm8k.json").read_text())
    wino = json.loads((ROOT / "gate3/winogrande.json").read_text())
    if gsm["model_revision"] != fixed["model_revision"] or wino["model_revision"] != fixed["model_revision"]:
        raise RuntimeError("downstream model revision mismatch")
    if gsm["mask"]["mask_hash"] != gate3_manifest["overall_sha256"]["Ours-FG"]:
        raise RuntimeError("GSM Ours-FG mask hash mismatch")
    if wino["methods"]["Wanda"]["mask"]["mask_hash"] != fixed["standard_wanda_mask_sha256"]:
        raise RuntimeError("Wino Standard Wanda mask hash mismatch")
    if wino["methods"]["Ours-FG"]["mask"]["mask_hash"] != gate3_manifest["overall_sha256"]["Ours-FG"]:
        raise RuntimeError("Wino Ours-FG mask hash mismatch")

    gsm_wanda = _records_audit(
        gsm["references"]["Wanda"]["source"],
        gsm["references"]["Wanda"]["source_sha256"],
        1319,
    )
    gsm_fg = _records_audit(
        ROOT / "gate3/gsm8k_ours_predictions.jsonl",
        gsm["ours"]["prediction_sha256"],
        1319,
    )
    _assert_same_examples(gsm_wanda, gsm_fg)
    wino_wanda = _records_audit(
        ROOT / "gate3/winogrande_wanda_predictions.jsonl",
        wino["methods"]["Wanda"]["prediction_sha256"],
        1267,
    )
    wino_fg = _records_audit(
        ROOT / "gate3/winogrande_ours_fg_predictions.jsonl",
        wino["methods"]["Ours-FG"]["prediction_sha256"],
        1267,
    )
    assert_same_wino(wino_wanda, wino_fg)

    compatibility = {
        "status": "pass",
        "model_revision": fixed["model_revision"],
        "geometry_manifest_sha256": sha(ROOT / "geometry_calibration_manifest.json"),
        "standard_wanda_mask_sha256": fixed["standard_wanda_mask_sha256"],
        "ours_fg_mask_sha256": gate3_manifest["overall_sha256"]["Ours-FG"],
        "gsm8k": {
            "status": "reuse Dense/Wanda/SparseGPT/Ours-FG; run only Ours-DLMW",
            "protocol_hash": gsm["protocol_hash"],
            "sample_count": len(gsm_wanda),
            "example_order_match": True,
            "prediction_hashes_verified": True,
        },
        "winogrande": {
            "status": "reuse pinned-revision Dense/Wanda/SparseGPT/Ours-FG; run only Ours-DLMW",
            "protocol": wino["protocol"],
            "protocol_hash": _protocol_hash(wino["protocol"]),
            "sample_count": len(wino_wanda),
            "example_order_match": True,
            "prediction_hashes_verified": True,
        },
    }
    ATTR_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ATTR_ROOT / "protocol_compatibility_audit.json", compatibility)
    write_json(
        ATTR_ROOT / "ours_dlmw_mask_manifest.json",
        {
            "status": "complete",
            "model_revision": fixed["model_revision"],
            "method": "Ours-DLMW",
            "matrix_count": len(entries),
            "target": TARGET,
            "changed_vs_standard_wanda": changed_w,
            "changed_vs_ours_fg": changed_fg,
            "target_mask_xor": target_xor,
            "target_raw_sha256": raw_mask_sha256(dlmw),
            "overall_sha256": overall,
            "total_pruned": pruned,
            "total_weights": weights,
            "sparsity": pruned / weights,
            "per_row_pruned_min": int(dlmw.sum(1).min()),
            "per_row_pruned_max": int(dlmw.sum(1).max()),
            "entries": entries,
        },
    )
    print(json.dumps({"event": "attribution_prepare_complete", "mask_sha256": overall, "xor": target_xor}), flush=True)


def _load_attribution_masks():
    manifest = json.loads((ATTR_ROOT / "ours_dlmw_mask_manifest.json").read_text())
    result = {}
    for row in manifest["entries"]:
        bits = Path(row["runtime_path"]).read_bytes()
        payload = {"shape": row["shape"], "bits": bits}
        if mask_sha256(payload) != row["sha256"]:
            raise RuntimeError("packed Ours-DLMW mask hash mismatch")
        mask = unpack_mask(payload)
        if raw_mask_sha256(mask) != row["raw_sha256"]:
            raise RuntimeError("raw Ours-DLMW mask hash mismatch")
        result[row["canonical_name"]] = mask
    if len(result) != 224 or _overall_mask_hash(manifest["entries"], "Ours-DLMW") != manifest["overall_sha256"]:
        raise RuntimeError("Ours-DLMW mask manifest is incomplete")
    return result, manifest


@torch.no_grad()
def _load_model_and_apply_dlmw():
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(config)
    before = model_sha(model)
    expected = json.loads((ROOT / "gate3_mask_manifest.json").read_text())["dense_model_sha256"]
    if before != expected:
        raise RuntimeError("dense model SHA differs before attribution evaluation")
    masks, manifest = _load_attribution_masks()
    started = time.monotonic()
    applied = apply_masks(modules(model), masks)
    pruning_seconds = time.monotonic() - started
    zero = _zero_mask_summary(model, "Ours-DLMW", require_exact_rowwise=True)
    if applied["sparsity"] != 0.5 or zero["sparsity"] != 0.5:
        raise RuntimeError("Ours-DLMW applied sparsity mismatch")
    if zero["mask_hash"] != manifest["overall_sha256"]:
        raise RuntimeError("applied Ours-DLMW hash mismatch")
    return model, before, zero, pruning_seconds


@torch.no_grad()
def run_gsm():
    ATTR_ROOT.mkdir(parents=True, exist_ok=True)
    fixed = json.loads(PREREG.read_text())
    config = load_exp002_config(EXP002 / "config.yaml")
    config_hash, protocol = _evaluation_config_hash(config)
    prior = json.loads((ROOT / "gate3/gsm8k.json").read_text())
    if config_hash != prior["protocol_hash"]:
        raise RuntimeError("GSM8K protocol differs from Prototype 1")
    references = _load_gsm_references(config_hash)
    fg = _records_audit(ROOT / "gate3/gsm8k_ours_predictions.jsonl", prior["ours"]["prediction_sha256"], 1319)
    _assert_same_examples(references["Wanda"]["records"], fg)

    model, dense_sha, mask, pruning_seconds = _load_model_and_apply_dlmw()
    pruned_before = model_sha(model)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    measured, records = _evaluate_gsm8k(model, tokenizer, config, "Ours-DLMW", None, config_hash)
    if measured["num_examples"] != 1319:
        raise RuntimeError("wrong GSM8K sample count")
    _assert_same_examples(references["Wanda"]["records"], records)
    pruned_after = model_sha(model)
    if pruned_before != pruned_after:
        raise RuntimeError("GSM8K evaluation changed Ours-DLMW weights")
    output = ATTR_ROOT / "gsm8k_ours_dlmw_predictions.jsonl"
    _write_jsonl(output, records)
    result = {
        "status": "complete",
        "method": "Ours-DLMW",
        "model_revision": fixed["model_revision"],
        "protocol": protocol,
        "protocol_hash": config_hash,
        "mask": mask,
        "dense_model_sha256": dense_sha,
        "pruned_model_sha256_before_eval": pruned_before,
        "pruned_model_sha256_after_eval": pruned_after,
        "pruning_seconds": pruning_seconds,
        "evaluation": {**measured, "correct": sum(bool(x["correct"]) for x in records)},
        "prediction_path": str(output),
        "prediction_sha256": sha(output),
        "paired": {
            "Wanda_vs_DLMW": paired_correctness(
                [x["correct"] for x in references["Wanda"]["records"]],
                [x["correct"] for x in records],
                "Wanda",
                "DLMW",
            ),
            "DLMW_vs_FG": paired_correctness(
                [x["correct"] for x in records],
                [x["correct"] for x in fg],
                "DLMW",
                "FG",
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lm_eval": importlib.metadata.version("lm_eval"),
        },
    }
    write_json(ATTR_ROOT / "gsm8k_ours_dlmw.json", result)
    print(json.dumps({"event": "attribution_gsm_complete", "evaluation": result["evaluation"], "paired": result["paired"]}), flush=True)


@torch.no_grad()
def run_wino():
    ATTR_ROOT.mkdir(parents=True, exist_ok=True)
    fixed = json.loads(PREREG.read_text())
    prior = json.loads((ROOT / "gate3/winogrande.json").read_text())
    reference = _records_audit(
        ROOT / "gate3/winogrande_wanda_predictions.jsonl",
        prior["methods"]["Wanda"]["prediction_sha256"],
        1267,
    )
    fg = _records_audit(
        ROOT / "gate3/winogrande_ours_fg_predictions.jsonl",
        prior["methods"]["Ours-FG"]["prediction_sha256"],
        1267,
    )
    assert_same_wino(reference, fg)

    base = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, dense_sha, mask, pruning_seconds = _load_model_and_apply_dlmw()
    pruned_before = model_sha(model)
    tokenizer = AutoTokenizer.from_pretrained(
        base["model"]["id"],
        revision=base["model"]["revision"],
        trust_remote_code=True,
    )
    measured, records = _evaluate_winogrande(model, tokenizer, torch.device("cuda:0"))
    if measured["sample_count"] != 1267:
        raise RuntimeError("wrong WinoGrande sample count")
    assert_same_wino(reference, records)
    pruned_after = model_sha(model)
    if pruned_before != pruned_after:
        raise RuntimeError("WinoGrande evaluation changed Ours-DLMW weights")
    protocol_hash = _protocol_hash(prior["protocol"])
    for record in records:
        record["method"] = "Ours-DLMW"
        record["evaluation_config_hash"] = protocol_hash
    output = ATTR_ROOT / "winogrande_ours_dlmw_predictions.jsonl"
    _write_jsonl(output, records)
    result = {
        "status": "complete",
        "method": "Ours-DLMW",
        "model_revision": fixed["model_revision"],
        "protocol": prior["protocol"],
        "protocol_hash": protocol_hash,
        "mask": mask,
        "dense_model_sha256": dense_sha,
        "pruned_model_sha256_before_eval": pruned_before,
        "pruned_model_sha256_after_eval": pruned_after,
        "pruning_seconds": pruning_seconds,
        "evaluation": measured,
        "prediction_path": str(output),
        "prediction_sha256": sha(output),
        "paired": {
            "Wanda_vs_DLMW": paired_correctness(
                [x["correct"] for x in reference], [x["correct"] for x in records], "Wanda", "DLMW"
            ),
            "DLMW_vs_FG": paired_correctness(
                [x["correct"] for x in records], [x["correct"] for x in fg], "DLMW", "FG"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lm_eval": importlib.metadata.version("lm_eval"),
        },
    }
    write_json(ATTR_ROOT / "winogrande_ours_dlmw.json", result)
    print(json.dumps({"event": "attribution_wino_complete", "evaluation": measured, "paired": result["paired"]}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _correctness(path):
    records = _read_jsonl(path)
    return records, [bool(x["correct"]) for x in records]


def _outcome(gsm_counts, wino_counts):
    w, d, f = (gsm_counts[x] for x in ("Wanda", "DLMW", "FG"))
    ww, dw, fw = (wino_counts[x] for x in ("Wanda", "DLMW", "FG"))
    if d > f and dw >= fw:
        return "D"
    if d > w and f > d and fw >= dw:
        return "A"
    if f > w and d <= w:
        return "C"
    if d > w and (f <= d or (f > d and fw < dw)):
        return "B"
    return "E"


def finalize():
    prior_gsm = json.loads((ROOT / "gate3/gsm8k.json").read_text())
    prior_wino = json.loads((ROOT / "gate3/winogrande.json").read_text())
    dlmw_gsm = json.loads((ATTR_ROOT / "gsm8k_ours_dlmw.json").read_text())
    dlmw_wino = json.loads((ATTR_ROOT / "winogrande_ours_dlmw.json").read_text())

    gsm_paths = {
        "Wanda": prior_gsm["references"]["Wanda"]["source"],
        "DLMW": dlmw_gsm["prediction_path"],
        "FG": str(ROOT / "gate3/gsm8k_ours_predictions.jsonl"),
    }
    wino_paths = {
        "Wanda": str(ROOT / "gate3/winogrande_wanda_predictions.jsonl"),
        "DLMW": dlmw_wino["prediction_path"],
        "FG": str(ROOT / "gate3/winogrande_ours_fg_predictions.jsonl"),
    }
    records = {"gsm8k": {}, "winogrande": {}}
    bits = {"gsm8k": {}, "winogrande": {}}
    for task, paths in (("gsm8k", gsm_paths), ("winogrande", wino_paths)):
        for method, path in paths.items():
            records[task][method], bits[task][method] = _correctness(path)
        if task == "gsm8k":
            _assert_same_examples(records[task]["Wanda"], records[task]["DLMW"])
            _assert_same_examples(records[task]["Wanda"], records[task]["FG"])
        else:
            assert_same_wino(records[task]["Wanda"], records[task]["DLMW"])
            assert_same_wino(records[task]["Wanda"], records[task]["FG"])

    summaries = {}
    for task, n in (("gsm8k", 1319), ("winogrande", 1267)):
        counts = {name: sum(values) for name, values in bits[task].items()}
        paired = {
            "Wanda_vs_DLMW": paired_correctness(bits[task]["Wanda"], bits[task]["DLMW"], "Wanda", "DLMW"),
            "DLMW_vs_FG": paired_correctness(bits[task]["DLMW"], bits[task]["FG"], "DLMW", "FG"),
            "Wanda_vs_FG": paired_correctness(bits[task]["Wanda"], bits[task]["FG"], "Wanda", "FG"),
        }
        summaries[task] = {
            "sample_count": n,
            "correct": counts,
            "accuracy": {name: value / n for name, value in counts.items()},
            "paired": paired,
            "three_way_correctness": three_way_correctness(
                bits[task]["Wanda"], bits[task]["DLMW"], bits[task]["FG"]
            ),
            "decomposition": attribution_decomposition(counts["Wanda"], counts["DLMW"], counts["FG"], n),
        }

    outcome = _outcome(summaries["gsm8k"]["correct"], summaries["winogrande"]["correct"])
    cost = {
        "source": "existing artifacts/logs only; no runtime benchmark rerun",
        "DLMW": {
            "dense_forward_states": 80,
            "masked_activation_accumulation": True,
            "readout_geometry_or_kappa": False,
            "persisted_score_bytes": (STORE / "score_DLMW.pt").stat().st_size,
            "wall_clock_scoring_seconds": None,
            "peak_gpu_memory_bytes": None,
        },
        "FG": {
            "dense_forward_states": 80,
            "masked_activation_accumulation": True,
            "readout_geometry_or_kappa": True,
            "fp64_G_accumulation_shape": [4096, 12288],
            "persisted_score_bytes": (STORE / "score_FG.pt").stat().st_size,
            "wall_clock_scoring_seconds": None,
            "peak_gpu_memory_bytes": None,
        },
        "note": "Existing run log records 80 completed states but has no trustworthy wall-clock or peak-memory instrumentation.",
    }
    result = {
        "status": "complete",
        "outcome": outcome,
        "primary_question": "Fisher incremental downstream value after holding DLM calibration fixed",
        "gsm8k": summaries["gsm8k"],
        "winogrande": summaries["winogrande"],
        "cost_accounting": cost,
        "protocol_compatibility_audit": str(ATTR_ROOT / "protocol_compatibility_audit.json"),
        "mask_manifest": str(ATTR_ROOT / "ours_dlmw_mask_manifest.json"),
        "no_tuning_or_rescue": True,
    }
    write_json(ATTR_ROOT / "attribution_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["prepare", "gsm", "wino", "finalize"])
    job = parser.parse_args().job
    {"prepare": prepare, "gsm": run_gsm, "wino": run_wino, "finalize": finalize}[job]()


if __name__ == "__main__":
    main()
