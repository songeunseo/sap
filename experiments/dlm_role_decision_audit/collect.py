#!/usr/bin/env python3
"""Causal bundle audit with ordinary batch-one full forwards."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_role_decision_audit.core import CONDITIONS, sha256
from experiments.projection_capacity_allocation_65.run import load_dense, read_mask
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
from experiments.wanda_failure_characterization.run_failure_map import state_tensors

ROOT = Path("experiments/dlm_role_decision_audit")
RUNTIME = Path("/DATA/tmluser1/sap-dlm-role-decision-audit")
HELDOUT = Path("experiments/wanda_failure_characterization/heldout_state_manifest.json")
MANIFESTS = {
    "A": Path("experiments/dlm_dual_role_mini100/role65_mask_manifest.json"),
    "B": ROOT / "manifest_B_exact_local_max.json",
    "C": ROOT / "manifest_C_supported_global_minimax.json",
}


def event(kind, **kwargs):
    print(json.dumps({"event": kind, "time": time.time(), **kwargs}, sort_keys=True), flush=True)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp); tmp.replace(path)


def load_inputs():
    config = json.loads((ROOT / "config.json").read_text())
    bundles = json.loads((ROOT / "bundles.json").read_text())
    states = json.loads(HELDOUT.read_text())
    if states["historical_state_sha256"] != config["heldout_state_digest"]:
        raise RuntimeError("heldout digest mismatch")
    for key, path in MANIFESTS.items():
        if sha256(path) != bundles["manifest_sha256"][key]:
            raise RuntimeError(f"manifest {key} changed")
    return config, bundles, states, {k: json.loads(v.read_text()) for k, v in MANIFESTS.items()}


def metric(logits, dense_selected, clean, token_mask, p_mask):
    selected = logits[0, token_mask[0]].float()
    dense = dense_selected.float().to(selected.device)
    target = clean[0, token_mask[0]]
    lp, dp = F.log_softmax(selected, -1), F.log_softmax(dense, -1)
    kl_token = dp.exp() * (dp - lp)
    ce = F.cross_entropy(selected, target, reduction="none")
    dense_ce = F.cross_entropy(dense, target, reduction="none")
    return {
        "mean_kl": float(kl_token.sum(-1).mean()),
        "median_kl": float(kl_token.sum(-1).median()),
        "delta_loss": float((ce - dense_ce).sum() / float(p_mask) / 256),
        "top1_agreement": float((selected.argmax(-1) == dense.argmax(-1)).float().mean()),
        "confidence_mae": float((lp.exp().amax(-1) - dp.exp().amax(-1)).abs().mean()),
    }


@torch.inference_mode()
def dense_references(model, states, device, limit):
    worker = os.environ.get("CUDA_VISIBLE_DEVICES", "worker").replace(",", "_")
    path = RUNTIME / f"dense_refs_{limit or 40}_{worker}.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["heldout_digest"] == states["historical_state_sha256"]:
            return payload["refs"]
    refs = []
    for i, state in enumerate(states["states"][:limit]):
        noisy, _, mask = state_tensors(state, device)
        first = model(noisy).logits
        second = model(noisy).logits
        diff = float((first - second).abs().max())
        if diff != 0: raise RuntimeError(f"dense same-path sham drift: {diff}")
        refs.append(first[0, mask[0]].cpu())
        event("dense_reference", state=i + 1, total=len(states["states"][:limit]))
    atomic_torch(path, {"heldout_digest": states["historical_state_sha256"], "refs": refs})
    return refs


def condition_weights(condition, changes, base, new, token_mask):
    result = {}
    for j, change in enumerate(changes):
        name = change["name"]
        if condition in ("baseline", "sham"):
            result[name] = (base[name], None)
        elif condition in ("bundle_direct", "bundle_mixed"):
            result[name] = (new[name], None)
        elif condition == "masked_only":
            result[name] = (base[name], (new[name], token_mask))
        elif condition == "unmasked_only":
            result[name] = (base[name], (new[name], ~token_mask))
        elif condition.startswith("single_"):
            target = int(condition.split("_")[1])
            result[name] = (new[name] if j == target else base[name], None)
        else:
            raise ValueError(condition)
    return result


@torch.inference_mode()
def forward_condition(model, mapping, noisy, mask, changes, base, new, condition):
    specs = condition_weights(condition, changes, base, new, mask[0])
    handles = []
    for name, (weight, role_override) in specs.items():
        def hook(mod, inp, out, w=weight, override=role_override):
            if override is None:
                return F.linear(inp[0], w, mod.bias)
            alt, positions = override
            ordinary = F.linear(inp[0], w, mod.bias)
            changed = F.linear(inp[0], alt, mod.bias)
            return torch.where(positions.view(1, -1, 1), changed, ordinary)
        handles.append(mapping[name].register_forward_hook(hook))
    try:
        return model(noisy).logits
    finally:
        for handle in handles: handle.remove()


def build_weights(mapping, candidate_entries, bundle, dense_original):
    base, new = {}, {}
    for change in bundle["changes"]:
        name = change["name"]; module = mapping[name]; row = candidate_entries[change["module_index"]]
        original = dense_original[name] if dense_original is not None else module.weight.detach()
        bm = read_mask(row, int(change["base_level"]), module.weight.device)
        nm = read_mask(row, int(change["new_level"]), module.weight.device)
        base[name] = original.masked_fill(bm, 0)
        new[name] = original.masked_fill(nm, 0)
    return base, new


@torch.inference_mode()
def collect(background, state_limit=None, bundle_limit=None):
    config, bundle_doc, states_doc, manifests = load_inputs()
    bundles = bundle_doc["bundles"][:bundle_limit]
    states = states_doc["states"][:state_limit]
    model, mapping = load_dense(); device = next(model.parameters()).device
    refs = dense_references(model, states_doc, device, state_limit)
    union = sorted({x["name"] for b in bundles for x in b["changes"]})
    if background == "role_sparse" and state_limit is None and bundle_limit is None:
        union = sorted({manifests["A"]["entries"][i]["name"] for label in ("B", "C")
                        for i, (a, b) in enumerate(zip(manifests["A"]["allocation_levels"],
                                                       manifests[label]["allocation_levels"])) if a != b})
    dense_original = None
    if background == "role_sparse":
        dense_original = {name: mapping[name].weight.detach().clone() for name in union}
        apply_manifest(model, mapping, manifests["A"])
    output = RUNTIME / f"bundle_{background}{'_smoke' if state_limit or bundle_limit else ''}.pt"
    payload = torch.load(output, map_location="cpu", weights_only=False) if output.exists() else {
        "background": background, "heldout_digest": config["heldout_state_digest"],
        "bundle_digest": bundle_doc["bundle_selection_sha256"], "rows": []}
    done = {(r["bundle_id"], r["state_index"], r["condition"]) for r in payload["rows"]}
    started = time.monotonic()
    for bi, bundle in enumerate(bundles):
        base, new = build_weights(mapping, json.loads(Path("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json").read_text())["entries"], bundle, dense_original)
        for si, state in enumerate(states):
            noisy, clean, mask = state_tensors(state, device)
            cached = {}
            for condition in CONDITIONS:
                if condition.startswith("single_") and int(condition[-1]) >= len(bundle["changes"]):
                    continue
                key = (bundle["bundle_id"], si, condition)
                if key in done: continue
                logits = forward_condition(model, mapping, noisy, mask, bundle["changes"], base, new, condition)
                row = metric(logits, refs[si], clean, mask, state["p_mask"])
                row.update(bundle_id=bundle["bundle_id"], source=bundle["source"],
                           category=bundle["selection_category"], state_index=si,
                           sequence_index=int(state["sequence_index"]), timestep=float(state["timestep"]),
                           condition=condition)
                if condition in ("sham", "bundle_mixed", "bundle_direct"):
                    cached[condition] = logits.detach().cpu()
                payload["rows"].append(row); done.add(key)
                atomic_torch(output, payload)
                event("bundle_condition", background=background, bundle=bundle["bundle_id"],
                      bundle_index=bi + 1, bundles=len(bundles), state=si + 1, states=len(states),
                      condition=condition, completed=len(done), elapsed_seconds=time.monotonic()-started)
            if {"bundle_mixed", "bundle_direct", "sham"} <= set(cached):
                direct_diff = float((cached["bundle_mixed"] - cached["bundle_direct"]).abs().max())
                # Baseline may have been resumed, so sham equality is also checked via metrics in analysis.
                if direct_diff != 0: raise RuntimeError(f"mixed/direct mismatch {direct_diff}")
        del base, new; torch.cuda.empty_cache()

    # Full allocations are evaluated only once, in the role-sparse worker.
    if background == "role_sparse" and state_limit is None and bundle_limit is None:
        full_path = RUNTIME / "full_allocations.pt"
        full = torch.load(full_path, map_location="cpu", weights_only=False) if full_path.exists() else {"rows": []}
        completed = {(r["method"], r["state_index"]) for r in full["rows"]}
        # Recover all changed dense weights before A was applied.
        all_changed = sorted({manifests["A"]["entries"][i]["name"] for label in ("B", "C")
                              for i, (a, b) in enumerate(zip(manifests["A"]["allocation_levels"], manifests[label]["allocation_levels"])) if a != b})
        missing = [n for n in all_changed if n not in dense_original]
        if missing: raise RuntimeError(f"dense originals missing for full comparison: {missing[:2]}")
        for method in ("A", "B", "C"):
            changes = []
            for i, (a, b) in enumerate(zip(manifests["A"]["allocation_levels"], manifests[method]["allocation_levels"])):
                if a != b: changes.append({"name": manifests["A"]["entries"][i]["name"], "module_index": i, "base_level": a, "new_level": b})
            if method == "A": changes = []
            base = new = {}
            if changes:
                base, new = build_weights(mapping, json.loads(Path("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json").read_text())["entries"], {"changes": changes}, dense_original)
            for si, state in enumerate(states):
                if (method, si) in completed: continue
                noisy, clean, mask = state_tensors(state, device)
                logits = model(noisy).logits if method == "A" else forward_condition(model, mapping, noisy, mask, changes, base, new, "bundle_direct")
                row = metric(logits, refs[si], clean, mask, state["p_mask"])
                row.update(method=method, state_index=si, sequence_index=int(state["sequence_index"]), timestep=float(state["timestep"]))
                full["rows"].append(row); atomic_torch(full_path, full)
                event("full_allocation", method=method, state=si + 1, states=len(states))
    del model; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", choices=("dense", "role_sparse"), required=True)
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--bundle-limit", type=int)
    args = parser.parse_args()
    collect(args.background, args.state_limit, args.bundle_limit)


if __name__ == "__main__": main()
