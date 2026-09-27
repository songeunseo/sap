"""Shared-state single-reveal diagnostic for the frozen Dense32 trajectory."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_multiscale_ac50.artifacts import digest, freeze, read, sha, write


def trace_path(root, example_id):
    return Path(root) / "traces" / f"{example_id:04d}.json"


def teacher_paths(root, example_id):
    base = Path(root) / "diagnostic" / "dense" / f"{example_id:04d}"
    return base.with_suffix(".pt"), base.with_suffix(".json")


def sparse_path(root, example_id):
    return Path(root) / "diagnostic" / "A" / f"{example_id:04d}.json"


def check_trace(root, request, trace, config_identity):
    if trace.get("example_id") != request["example_id"] or trace.get("prompt_hash") != request["prompt_hash"]:
        raise RuntimeError("Trace request identity mismatch")
    if trace.get("config_identity") != config_identity or len(trace.get("states", [])) != 4:
        raise RuntimeError("Trace configuration mismatch")
    for state, step in zip(trace["states"], (0, 8, 16, 24)):
        ids = state["input_ids"]
        if (state["step"] != step or state["input_ids_sha256"] != digest(ids)
                or state["input_length"] != len(ids)
                or state["masked_count"] != sum(x == 126336 for x in ids)
                or abs(state["mask_fraction"] - state["masked_count"] / len(ids)) > 1e-8
                or abs(state["response_mask_fraction"] - state["masked_count"] / 256) > 1e-8):
            raise RuntimeError("Trace state mismatch")
        source = state["source"]
        targets = state["targets"]
        selected = [source, *targets]
        if (len(targets) != 7 or len({x["position"] for x in selected}) != 8
                or any(not isinstance(x["position"], int) or not 0 <= x["position"] < len(ids)
                       or not isinstance(x["token_id"], int) or x["token_id"] < 0
                       or x["token_id"] == 126336
                       or not math.isfinite(x["native_confidence"]) for x in selected)):
            raise RuntimeError("Trace top-eight positions invalid")
        if ids[source["position"]] != 126336 or any(ids[x["position"]] != 126336 for x in targets):
            raise RuntimeError("Trace selected a visible token")
        after = ids.copy()
        after[source["position"]] = source["token_id"]
        if digest(after) != state["after_input_ids_sha256"]:
            raise RuntimeError("Trace reveal hash mismatch")
    return trace


def distribution(model, ids, positions, counter):
    device = model.device
    counter["forwards"] += 1
    counter["forward_tokens"] += len(ids)
    logits = model(torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)).logits
    chosen = logits[0, positions].float()
    values = F.log_softmax(chosen, dim=-1).cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError("Nonfinite diagnostic distribution")
    return values


@torch.inference_mode()
def collect_dense(root, request, model, trace, config_identity, counter):
    """Persist FP32 full-vocabulary readouts one question at a time."""
    path, receipt = teacher_paths(root, request["example_id"])
    if receipt.exists():
        meta = read(receipt)
        if (meta["config_identity"] != config_identity
                or meta["trace_sha256"] != sha(trace_path(root, request["example_id"]))
                or meta["tensor_sha256"] != sha(path)):
            raise RuntimeError("Dense diagnostic receipt mismatch")
        return meta
    # A tensor without its receipt is an interrupted checkpoint; regenerate it.
    check_trace(root, request, trace, config_identity)
    before, after = [], []
    started = time.monotonic()
    for state in trace["states"]:
        ids = state["input_ids"]
        targets = [x["position"] for x in state["targets"]]
        before.append(distribution(model, ids, targets, counter))
        revealed = ids.copy()
        revealed[state["source"]["position"]] = state["source"]["token_id"]
        after.append(distribution(model, revealed, targets, counter))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save({"before": torch.stack(before), "after": torch.stack(after)}, temp)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    temp.replace(path)
    meta = dict(example_id=request["example_id"], config_identity=config_identity,
                trace_sha256=sha(trace_path(root, request["example_id"])),
                tensor_sha256=sha(path), seconds=time.monotonic()-started,
                shape=[4, 7, int(before[0].shape[-1])])
    freeze(receipt, meta)
    return meta


def metrics(d_before, d_after, a_before, a_after, state):
    p0, p1, q0, q1 = (x.exp() for x in (d_before, d_after, a_before, a_after))
    dense_delta, sparse_delta = p1-p0, q1-q0
    target_rows = []
    for j, target in enumerate(state["targets"]):
        db, da, ab, aa = (x[j] for x in (d_before, d_after, a_before, a_after))
        pb, pa, qb, qa = (x[j] for x in (p0, p1, q0, q1))
        target_rows.append(dict(position=target["position"],
            dense_response_l1=float(dense_delta[j].abs().sum()),
            response_error_l1=float((sparse_delta[j]-dense_delta[j]).abs().sum()),
            endpoint_kl_before=float((pb*(db-ab)).sum()),
            endpoint_kl_after=float((pa*(da-aa)).sum()),
            dense_argmax_before=int(db.argmax()), dense_argmax_after=int(da.argmax()),
            A_argmax_before=int(ab.argmax()), A_argmax_after=int(aa.argmax()),
            dense_flip=bool(db.argmax()!=da.argmax()), A_flip=bool(ab.argmax()!=aa.argmax()),
            disagreement_before=bool(db.argmax()!=ab.argmax()),
            disagreement_after=bool(da.argmax()!=aa.argmax()),
            dense_confidence_before=float(pb.max()),
            dense_mask_fraction=float(state["mask_fraction"]),
            response_mask_fraction=float(state["response_mask_fraction"])))
    return target_rows


@torch.inference_mode()
def collect_sparse(root, request, model, trace, config_identity, counter):
    path = sparse_path(root, request["example_id"])
    teacher, receipt = teacher_paths(root, request["example_id"])
    if not receipt.exists():
        return False
    meta = read(receipt)
    if meta["config_identity"] != config_identity or meta["trace_sha256"] != sha(trace_path(root, request["example_id"])) or meta["tensor_sha256"] != sha(teacher):
        raise RuntimeError("Dense readout identity mismatch")
    if path.exists():
        row = read(path)
        if (row["config_identity"] != config_identity
                or row["teacher_sha256"] != meta["tensor_sha256"]
                or row["trace_sha256"] != meta["trace_sha256"]):
            raise RuntimeError("Sparse diagnostic checkpoint mismatch")
        return True
    check_trace(root, request, trace, config_identity)
    saved = torch.load(teacher, map_location="cpu", weights_only=True)
    rows = []
    started = time.monotonic()
    for i, state in enumerate(trace["states"]):
        ids = state["input_ids"]
        positions = [x["position"] for x in state["targets"]]
        ab = distribution(model, ids, positions, counter)
        revealed = ids.copy()
        revealed[state["source"]["position"]] = state["source"]["token_id"]
        aa = distribution(model, revealed, positions, counter)
        rows.append(dict(step=state["step"], source=state["source"],
                         targets=metrics(saved["before"][i], saved["after"][i], ab, aa, state)))
    values = [t for s in rows for t in s["targets"]]
    means = {key: sum(t[key] for t in values)/len(values) for key in
             ("dense_response_l1", "response_error_l1", "endpoint_kl_before", "endpoint_kl_after",
              "dense_confidence_before", "dense_mask_fraction", "response_mask_fraction")}
    freeze(path, dict(example_id=request["example_id"], config_identity=config_identity,
                      trace_sha256=meta["trace_sha256"], teacher_sha256=meta["tensor_sha256"],
                      seconds=time.monotonic()-started, states=rows, question_means=means))
    return True
