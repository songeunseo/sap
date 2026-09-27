"""Model operations, imported only by explicitly launched tmux GPU workers."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .artifacts import (ARMS, Progress, ReadoutStore, checked, digest, freeze,
                        mask_identity, read, sha, write)
from .core import metrics, marginal_cost, nodes
from .evaluation import evaluate_requests, generator, grade, task_and_protocol
from .prepare import validate


def require_gpu_worker():
    if not os.environ.get("TMUX"):
        raise RuntimeError("GPU workers must run inside tmux; use launch")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise RuntimeError("Assign exactly one GPU through CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")


def save_tensor(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(tensor, tmp)
    tmp.replace(path)


def runtime(root, worker):
    require_gpu_worker()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    config = validate(root)
    from eval_llada import set_seed
    from experiments.projection_capacity_allocation_65.run import load_dense
    set_seed(config["evaluation"]["torch_seed"])
    progress = Progress(root, worker)
    progress("loading_dense")
    model, mapping = load_dense()
    model.eval()
    write(Path(root) / "worker_environment" / f"{worker}.json",
          dict(gpu=os.environ["CUDA_VISIBLE_DEVICES"], gpu_name=torch.cuda.get_device_name(0),
               torch_version=torch.__version__, cuda_version=torch.version.cuda,
               config_sha256=sha(Path(root) / "config.json")))
    return Runtime(root, config, model, mapping, progress)


class Runtime:
    def __init__(self, root, config, model, mapping, progress):
        self.root, self.config, self.model, self.mapping, self.progress = Path(root), config, model, mapping, progress
        self.config_hash = sha(self.root / "config.json")
        self.base = read(config["legacy_manifests"]["uniform"]["path"])
        self.refs = self.base["entries"]
        if list(mapping) != [r["name"] for r in self.refs]:
            raise RuntimeError("Model/reference module order mismatch")
        self.banks = {split:read(row["path"]) for split, row in config["banks"].items()}

    def memory_check(self):
        limit = self.config["evaluation"]["max_cuda_gib"] * 2**30
        if next(self.model.parameters()).device.type == "cuda" and torch.cuda.max_memory_allocated() > limit:
            raise RuntimeError("CUDA allocation exceeded frozen 30 GiB bound")

    @torch.inference_mode()
    def margin(self, chain, node):
        from experiments.dlm_context_response50.core import log_odds
        device = next(self.model.parameters()).device
        logits = self.model(torch.tensor([node["input_ids"]], device=device)).logits[0, chain["query"]]
        result = log_odds(logits, chain["gold"]).cpu().tolist()
        self.memory_check()
        return result

    def readouts(self, candidate, identity, split):
        bank = self.banks[split]
        path = self.root / "readouts" / candidate / f"{split}.json"
        fingerprint = dict(config_sha256=self.config_hash, mask_identity=identity,
                           bank_sha256=self.config["banks"][split]["sha256"])
        store = ReadoutStore(path, fingerprint, bank["states"], len(bank["chains"][0]["query"]))
        self.progress("readout", completed=len(store.values), total=bank["states"], candidate=candidate, split=split)
        for i, (chain, node) in enumerate(nodes(bank)):
            if i < len(store.values):
                continue
            store.append(self.margin(chain, node))
            self.progress("readout", completed=i+1, total=bank["states"], candidate=candidate, split=split)
        return store.complete()

    def distortion(self, candidate, identity, split):
        values = self.readouts(candidate, identity, split)
        teacher = self.teacher(split)
        result = metrics(values, teacher, self.banks[split])
        freeze(self.root / "diagnostics" / candidate / f"{split}.json",
               dict(config_sha256=self.config_hash, mask_identity=identity,
                    readout_sha256=sha(self.root / "readouts" / candidate / f"{split}.json"),
                    teacher_sha256=sha(self.root / "readouts/dense" / f"{split}.json"), **result))
        return result

    def teacher(self, split):
        from experiments.projection_capacity_allocation_65.run import DENSE_SHA
        path = self.root / "readouts/dense" / f"{split}.json"
        bank = self.banks[split]
        fp = dict(config_sha256=self.config_hash, mask_identity="dense:"+DENSE_SHA,
                  bank_sha256=self.config["banks"][split]["sha256"])
        return ReadoutStore(path, fp, bank["states"], len(bank["chains"][0]["query"])).complete()

    @torch.inference_mode()
    def apply_manifest(self, manifest):
        from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
        if apply_manifest(self.model, self.mapping, manifest) != self.config["pruning"]["pruned"]:
            raise RuntimeError("Wrong global pruning count")
        self.memory_check()

    def activation(self, block):
        path = self.config["activations"][block]
        checked(path, self.config["sources"][path])
        return torch.load(path, map_location="cpu", weights_only=False)

    @torch.inference_mode()
    def mask_block(self, block, counts, originals=None, save_folder=None, check_uniform=False):
        from experiments.dlm_ppl50.sequential import wanda_mask
        from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
        activation = self.activation(block)
        entries = []
        for i in range(block*7, (block+1)*7):
            ref = self.refs[i]
            name, count = ref["name"], counts[i]
            weight = self.mapping[name].weight
            if originals is not None:
                weight.copy_(originals[name])
            if list(weight.shape) != ref["shape"]:
                raise RuntimeError("Projection shape changed")
            mask = wanda_mask(weight, activation[name].to(weight.device), count)
            if not torch.all(mask.sum(1) == count):
                raise RuntimeError("Incorrect rowwise pruning")
            packed = pack_mask(mask.cpu())
            mask_hash = mask_sha256(packed)
            if check_uniform and mask_hash != ref["selected_mask"]["mask_sha256"]:
                raise RuntimeError(f"Frozen Wanda ranking failed to reproduce Uniform: {name}")
            entry = dict(name=name, shape=ref["shape"], weights=ref["weights"],
                         selected_mask=dict(mask_sha256=mask_hash, prune_per_row=count,
                                            pruned=count*weight.shape[0]))
            if save_folder is not None:
                path = save_folder / "masks" / f"{name}.pt"
                if path.exists():
                    # An interrupted build may already have written this exact mask.
                    if mask_sha256(torch.load(path, map_location="cpu", weights_only=False)) != mask_hash:
                        raise RuntimeError("Partial mask artifact differs from current ranking")
                else:
                    save_tensor(path, packed)
                entry["selected_mask"].update(path=str(path.resolve()), file_sha256=sha(path))
            weight.masked_fill_(mask, 0)
            entries.append(entry)
            del mask, packed
            self.memory_check()
        return entries

    def uniform(self, verify_ranking=False):
        if verify_ranking:
            counts = [r["shape"][1]//2 for r in self.refs]
            for b in range(32):
                self.mask_block(b, counts, check_uniform=True)
                self.progress("verify_wanda", completed=b+1, total=32)
        else:
            self.apply_manifest(self.base)
        from experiments.wanda_failure_characterization.run_failure_map import model_sha
        actual = model_sha(self.model)
        if actual != self.config["legacy_manifests"]["uniform"]["sparse_model_sha256"]:
            raise RuntimeError("Uniform physical model differs from historical model")

    @torch.inference_mode()
    def restore_block(self, block, originals):
        from experiments.projection_capacity_followup_65.run_heldout import selected_mask
        for ref in self.refs[block*7:(block+1)*7]:
            name = ref["name"]
            weight = self.mapping[name].weight
            expected = originals[name].to(weight.device).clone()
            expected.masked_fill_(selected_mask(ref, weight.device), 0)
            weight.copy_(expected)
            if not torch.equal(weight, expected):
                raise RuntimeError("Uniform block restore failed")

    def build(self, method):
        folder = self.root / "candidates" / method
        if method == "legacy_AC":
            manifest = read(self.config["legacy_manifests"]["AC"]["path"])
            self.apply_manifest(manifest)
            return manifest
        allocation = read(self.root / "allocation.json")
        if allocation["config_sha256"] != self.config_hash:
            raise RuntimeError("Allocation config mismatch")
        counts = allocation["allocations"][method]["row_counts"]
        path = folder / "mask_manifest.json"
        if path.exists():
            manifest = read(path)
            if manifest["config_sha256"] != self.config_hash or manifest["allocation_sha256"] != sha(self.root / "allocation.json"):
                raise RuntimeError("Candidate manifest fingerprint mismatch")
            if [r["selected_mask"]["prune_per_row"] for r in manifest["entries"]] != counts:
                raise RuntimeError("Candidate row counts differ from allocation")
            self.apply_manifest(manifest)
            return manifest
        entries = []
        for block in range(32):
            entries.extend(self.mask_block(block, counts, save_folder=folder))
            self.progress("build_mask", completed=block+1, total=32, candidate=method)
        total = sum(r["selected_mask"]["pruned"] for r in entries)
        if total != self.config["pruning"]["pruned"]:
            raise RuntimeError("Final candidate is not exactly 50% sparse")
        return freeze(path, dict(config_sha256=self.config_hash,
                                allocation_sha256=sha(self.root / "allocation.json"),
                                entries=entries, pruned=total))


def reference(root):
    rt = runtime(root, "reference")
    from experiments.projection_capacity_allocation_65.run import DENSE_SHA
    for split, bank in rt.banks.items():
        values = rt.readouts("dense", "dense:"+DENSE_SHA, split)
        chain, node = next(nodes(bank))
        if rt.margin(chain, node) != values[0]:
            raise RuntimeError("Dense repeat readout mismatch")
    rt.uniform(verify_ranking=True)
    for split in rt.banks:
        rt.distortion("uniform", rt.config["legacy_manifests"]["uniform"]["identity"], split)
    files = [root / "readouts" / candidate / f"{split}.json"
             for candidate in ("dense", "uniform") for split in rt.banks]
    freeze(root / "reference_receipt.json", dict(config_sha256=rt.config_hash,
           verified_uniform_masks=224, files={str(p):sha(p) for p in files}))
    rt.progress("complete")


def probes(root, blocks):
    rt = runtime(root, f"probes_{blocks[0]:02d}_{blocks[-1]:02d}")
    receipt = read(root / "reference_receipt.json")
    if receipt["config_sha256"] != rt.config_hash:
        raise RuntimeError("Reference config mismatch")
    for path, expected in receipt["files"].items():
        checked(path, expected)
    # Keep dense weights only for this worker's block shard; all other blocks stay Uniform50.
    originals = {r["name"]:rt.mapping[r["name"]].weight.detach().cpu().clone()
                 for b in blocks for r in rt.refs[b*7:(b+1)*7]}
    rt.uniform()
    for done, block in enumerate(blocks):
        output = root / "probes" / f"block{block:02d}.json"
        if output.exists():
            saved = read(output)
            if saved["config_sha256"] != rt.config_hash:
                raise RuntimeError("Probe config mismatch")
            for condition in saved["conditions"].values():
                checked(condition["readout_path"], condition["readout_sha256"])
            continue
        conditions = {}
        try:
            for rate in rt.config["pruning"]["probe_rates"]:
                counts = [int(r["shape"][1]*rate) for r in rt.refs]
                entries = rt.mask_block(block, counts, originals=originals)
                identity = digest(dict(uniform=mask_identity(rt.base), changed=mask_identity(dict(entries=entries))))
                label = f"probe_{block:02d}_{round(rate*100)}"
                result = rt.distortion(label, identity, "calibration")
                path = root / "readouts" / label / "calibration.json"
                conditions[str(rate)] = dict(pruned=sum(r["selected_mask"]["pruned"] for r in entries),
                                            metrics=result, mask_identity=identity,
                                            readout_path=str(path), readout_sha256=sha(path))
        finally:
            rt.restore_block(block, originals)
        low, high = [conditions[str(rate)] for rate in rt.config["pruning"]["probe_rates"]]
        costs = marginal_cost(low["metrics"], high["metrics"], low["pruned"], high["pruned"])
        freeze(output, dict(config_sha256=rt.config_hash, block=block, conditions=conditions,
                            costs=costs, restored_uniform=True))
        rt.progress("probe_blocks", completed=done+1, total=len(blocks), block=block)
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    if model_sha(rt.model) != rt.config["legacy_manifests"]["uniform"]["sparse_model_sha256"]:
        raise RuntimeError("Final Uniform weights changed during probes")
    rt.progress("complete", completed=len(blocks), total=len(blocks))


def candidate(root, method, split):
    rt = runtime(root, f"{split}_{method}")
    if split == "confirmation":
        selection = read(root / "confirmation_selection.json")
        if method not in selection["canonical_methods"]:
            raise RuntimeError("Method not in frozen confirmation selection")
    manifest = rt.build(method)
    identity = mask_identity(manifest)
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    sparse_sha = model_sha(rt.model)
    freeze(root / "candidates" / method / "model_identity.json",
           dict(config_sha256=rt.config_hash, mask_identity=identity, sparse_model_sha256=sparse_sha))
    if method == "legacy_AC" and sparse_sha != rt.config["legacy_manifests"]["AC"]["sparse_model_sha256"]:
        raise RuntimeError("Historical A+C weights differ")
    if split == "development":
        for bank_split in rt.banks:
            rt.distortion(method, identity, bank_split)
    requests_manifest = read(root / "requests.json")
    requests = requests_manifest[split]
    fingerprint = dict(config_sha256=rt.config_hash, requests_sha256=sha(root / "requests.json"),
                       mask_identity=identity, sparse_model_sha256=sparse_sha, split=split,
                       protocol_hash=rt.config["protocol_hash"])
    if split == "confirmation":
        fingerprint["selection_sha256"] = sha(root / "confirmation_selection.json")
    task, _, _ = task_and_protocol(read(rt.config["legacy_config"]))
    cached = None
    if split == "development":
        for name, row in rt.config["legacy_manifests"].items():
            if row["identity"] == identity:
                if row["sparse_model_sha256"] != sparse_sha:
                    raise RuntimeError("Same mask but different model weights")
                cached = {r["example_id"]:r for r in read(root / "cached_development.json")[name]}
                break
    if cached is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(rt.config["model"]["id"],
            revision=rt.config["model"]["revision"], trust_remote_code=True, local_files_only=True)
        generate_one = generator(rt.model, tokenizer, rt.config["evaluation"], rt.banks["calibration"]["mask_id"])
    else:
        generate_one = lambda request:cached[request["example_id"]]["generated_text"]
        freeze(root / "gsm8k" / split / method / "cache_reuse.json", dict(mask_identity=identity, source="legacy"))
    calls = [0]
    def tick(module, args, output):
        calls[0] += 1
        rt.memory_check()
    hook = rt.model.register_forward_hook(tick)
    try:
        result = evaluate_requests(root / "gsm8k" / split / method, requests, fingerprint,
            rt.config["protocol_hash"], method, generate_one, lambda req, text:grade(task, req, text), rt.progress)
    finally:
        hook.remove()
    if calls[0] % 256 or (cached is not None and calls[0]):
        raise RuntimeError("Unexpected generation forward count")
    if model_sha(rt.model) != sparse_sha:
        raise RuntimeError("Model weights changed during evaluation")
    # Per-attempt accounting stays separate from deterministic, resumable results.
    write(root / "gsm8k" / split / method / "last_attempt.json",
          dict(forward_calls=calls[0], reused_legacy=cached is not None))
    rt.progress("complete", completed=100, total=100, candidate=method, correct=result["correct"])
