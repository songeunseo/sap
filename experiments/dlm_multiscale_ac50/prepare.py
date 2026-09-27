"""CPU preparation and frozen source validation. Never loads model weights."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata

from .artifacts import LEGACY, PLAN, REPO, checked, digest, freeze, mask_identity, read, sha
from .core import clean_sequences, make_bank
from .evaluation import freeze_requests, grade, validate_prediction

DEPENDENCIES = [
    "eval_llada.py", "generate.py", "experiments/dlm_context_response50/core.py", "experiments/dlm_loss_aggregation/config.yaml",
    "experiments/dlm_loss_aggregation/exp002/config.yaml", "experiments/dlm_loss_aggregation/run.py",
    "experiments/dlm_loss_aggregation/core.py", "experiments/dlm_owl65/core.py",
    "experiments/dlm_ppl50/sequential.py", "experiments/projection_capacity_allocation_65/run.py",
    "experiments/projection_capacity_followup_65/run_heldout.py",
    "experiments/wanda_failure_characterization/run_failure_map.py",
    "experiments/dlm_capacity_predictor/audit_existing.py",
]



def checkpoint_files(model):
    from huggingface_hub import try_to_load_from_cache
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json", "model.safetensors", "pytorch_model.bin"):
        path = try_to_load_from_cache(model["id"], name, revision=model["revision"])
        if not isinstance(path, str):
            continue
        if name.endswith(".json"):
            names = sorted(set(read(path)["weight_map"].values()))
            files = [try_to_load_from_cache(model["id"], n, revision=model["revision"]) for n in names]
        else:
            files = [path]
        if not all(isinstance(p, str) and Path(p).is_file() and Path(p).stat().st_size > 0 for p in files):
            raise RuntimeError("Incomplete local model checkpoint; no GPU was loaded")
        return {p:Path(p).stat().st_size for p in files}
    raise RuntimeError("Model checkpoint not cached locally; preparation does not download weights")


def prepare(root, plan_path=PLAN):
    root, plan_path = Path(root), Path(plan_path)
    if (root / "started.json").exists():
        raise RuntimeError("Preparation cannot modify a started experiment; use validate")
    plan = read(plan_path)
    if plan["gsm8k"]["full_gsm8k"] or plan["gsm8k"]["max_new_generations"] != 800:
        raise RuntimeError("Unexpected experiment budget")
    for source in plan["sources"].values():
        checked(source["path"], source["sha256"])
    checkpoint = checkpoint_files(plan["model"])
    legacy = read(LEGACY / "config.json")
    if legacy["model"] != plan["model"]:
        raise RuntimeError("Model revision/dtype changed")
    if any(legacy["evaluation"][k] != v for k, v in plan["evaluation"].items()):
        raise RuntimeError("Evaluation settings changed")
    banks, sources = {}, {str(plan_path.resolve()):sha(plan_path)}
    for value in plan["sources"].values():
        sources[value["path"]] = value["sha256"]
    clean_hashes = []
    for split, name in (("calibration", "calibration_source"), ("diagnostic", "diagnostic_source")):
        source = read(plan["sources"][name]["path"])
        clean_hashes.append({digest(ids) for ids in clean_sequences(source).values()})
        bank = make_bank(source, plan["bank"], split)
        path = root / f"bank_{split}.json"
        freeze(path, bank)
        banks[split] = dict(path=str(path.resolve()), sha256=sha(path), states=bank["states"])
    if clean_hashes[0].intersection(clean_hashes[1]):
        raise RuntimeError("Clean sequence overlap between banks")

    task, requests = freeze_requests(root, plan, legacy)
    manifests, cached = {}, {}
    expected_names = None
    for name in ("uniform", "A", "AC"):
        folder = LEGACY / name
        result, manifest = read(folder / "results.json"), read(folder / "mask_manifest.json")
        checked(folder / "predictions.jsonl", result["predictions_sha256"])
        checked(folder / "mask_manifest.json", result["manifest_sha256"])
        if result["config_sha256"] != sha(LEGACY / "config.json"):
            raise RuntimeError("Legacy result config mismatch")
        if result["total"] != 100 or result["protocol_hash"] != requests["protocol_hash"]:
            raise RuntimeError("Unexpected legacy evaluation")
        rows = [__import__("json").loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
        if len(rows) != 100 or result["correct"] != sum(r["correct"] for r in rows):
            raise RuntimeError("Invalid cached prediction count/score")
        for row, request in zip(rows, requests["development"], strict=True):
            validate_prediction(row, request, requests["protocol_hash"])
            scored = grade(task, request, row["generated_text"])
            if any(row[k] != scored[k] for k in scored):
                raise RuntimeError("Official task regrading differs from historical result")
        if manifest["pruned"] != plan["pruning"]["pruned"] or len(manifest["entries"]) != 224:
            raise RuntimeError("Invalid historical pruning budget")
        names = [entry["name"] for entry in manifest["entries"]]
        if expected_names is None:
            expected_names = names
        if names != expected_names or len(set(names)) != 224:
            raise RuntimeError("Legacy projection identity/order mismatch")
        if sum(e["weights"] for e in manifest["entries"]) != plan["pruning"]["total_prunable"]:
            raise RuntimeError("Legacy prunable-weight total changed")
        if sum(e["selected_mask"]["pruned"] for e in manifest["entries"]) != manifest["pruned"]:
            raise RuntimeError("Legacy per-projection budgets do not sum to global target")
        for entry in manifest["entries"]:
            meta = entry["selected_mask"]
            height, width = entry["shape"]
            if height*width != entry["weights"] or meta["pruned"] != height*meta["prune_per_row"]:
                raise RuntimeError("Invalid historical row-count metadata")
            checked(meta["path"], meta["file_sha256"])
            sources[meta["path"]] = meta["file_sha256"]
        for path in (folder / "mask_manifest.json", folder / "results.json", folder / "predictions.jsonl"):
            sources[str(path)] = sha(path)
        manifests[name] = dict(path=str(folder / "mask_manifest.json"), identity=mask_identity(manifest),
                               sparse_model_sha256=result["sparse_model_sha256"])
        cached[name] = rows
    freeze(root / "cached_development.json", cached)
    ranking_receipt = read(LEGACY / "collection_receipt.json")
    activations = []
    for block in range(32):
        path = LEGACY / "activations" / f"block{block:02d}.pt"
        expected = ranking_receipt["files"][str(path)]
        checked(path, expected)
        sources[str(path)] = expected
        activations.append(str(path))
    for relative in DEPENDENCIES:
        path = REPO / relative
        sources[str(path)] = sha(path)
    for path in sorted((REPO / "model").glob("*.py")):
        sources[str(path)] = sha(path)
    for pattern in ("*.py", "*.sh"):
        for path in sorted(Path(__file__).parent.glob(pattern)):
            sources[str(path)] = sha(path)
    config = dict(status="prepared_not_started", schema=1, plan=plan, model=plan["model"], checkpoint_files=checkpoint,
                  pruning=plan["pruning"], evaluation=legacy["evaluation"],
                  protocol_hash=requests["protocol_hash"], banks=banks,
                  legacy_config=str(LEGACY / "config.json"), legacy_manifests=manifests,
                  activations=activations, sources=sources,
                  package_versions={p:importlib.metadata.version(p) for p in
                      ("torch", "numpy", "scipy", "transformers", "datasets", "accelerate", "lm_eval")},
                  requests_sha256=sha(root / "requests.json"),
                  cached_development_sha256=sha(root / "cached_development.json"))
    freeze(root / "config.json", config)
    validate(root)
    return dict(status="prepared_not_started", root=str(root), config_sha256=sha(root / "config.json"),
                states_per_bank=128, development_questions=100, confirmation_questions=100,
                gpu_used=False, obsidian_sync="local_pending")


def validate(root):
    root = Path(root)
    config = read(root / "config.json")
    for path, size in config["checkpoint_files"].items():
        if not Path(path).is_file() or Path(path).stat().st_size != size:
            raise RuntimeError(f"Local checkpoint shard missing/size changed: {path}")
    for package, version in config["package_versions"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Package version changed: {package}")
    for path, expected in config["sources"].items():
        checked(path, expected)
    for bank in config["banks"].values():
        checked(bank["path"], bank["sha256"])
    checked(root / "requests.json", config["requests_sha256"])
    checked(root / "cached_development.json", config["cached_development_sha256"])
    return config
