"""Standard Wanda 50% with SAP-scale calibration (128 x 2048 clean WikiText-2), full GSM8K.

Only the calibration scale differs from EXP-002 Wanda (8 x 256). Same repository prune_wanda,
same pinned LLaDA-8B-Base, same frozen 1319 GSM8K requests and native strict-match protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent / "output"
REPO = Path(__file__).resolve().parents[2]
REQUESTS_SRC = REPO / "experiments/dlm_crosschain_control50/output/requests.json"
LEGACY_CONFIG = REPO / "experiments/dlm_context_response50/config.json"
EXP002_WANDA_HASH = "3273fbbe3aaeeb1dc66b0c436e6af7537e07d1bb461667354f3da00a879fc484"
PROTOCOL_HASH = "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add"
MASK_ID = 126336
EXPECTED_PRUNED = 3_489_660_928
CALIBRATIONS = {"exp002_gate": dict(nsamples=8, seqlen=256), "sap": dict(nsamples=128, seqlen=2048)}
SEED = 0
SHARDS = 2


def progress(stage, **values):
    from experiments.dlm_multiscale_ac50.artifacts import write
    write(ROOT / "progress" / f"{os.environ.get('WORKER_NAME', 'main')}.json",
          dict(stage=stage, time=time.time(), pid=os.getpid(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), **values))
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def load_dense(seqlen):
    from experiments.projection_capacity_allocation_65.run import load_dense as _load
    model, _ = _load()  # asserts the pinned dense SHA
    model.seqlen = seqlen
    return model


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", revision="0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                                         trust_remote_code=True, local_files_only=True)


def prune(name):
    import torch
    from lib.prune_llada import prune_wanda
    from experiments.dlm_loss_aggregation.exp002.run import _zero_mask_summary
    cal = CALIBRATIONS[name]
    started = time.time()
    model = load_dense(cal["seqlen"])
    progress("pruning", calibration=name, **cal)
    args = SimpleNamespace(nsamples=cal["nsamples"], seed=SEED, sparsity_ratio=0.5, use_variant=False)
    prune_wanda(args, model, tokenizer(), device=torch.device("cuda:0"))
    summary = _zero_mask_summary(model, "wanda")
    summary.update(calibration=name, seconds=time.time() - started, **cal)
    return model, summary


def cmd_prune():
    import torch
    from experiments.dlm_multiscale_ac50.artifacts import freeze, sha, write
    from experiments.dlm_loss_aggregation.core import pack_mask
    from experiments.dlm_loss_aggregation.exp002.run import _module_map
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "requests.json").exists():
        (ROOT / "requests.json").write_bytes(REQUESTS_SRC.read_bytes())
    if sha(REQUESTS_SRC) != sha(ROOT / "requests.json"):
        raise RuntimeError("frozen requests differ")
    # Gate: the same code must reproduce the historical EXP-002 Wanda mask.
    if not (ROOT / "gate.json").exists():
        model, gate = prune("exp002_gate")
        gate["matches_exp002"] = gate["mask_hash"] == EXP002_WANDA_HASH
        write(ROOT / "gate.json", gate)
        del model
        torch.cuda.empty_cache()
    gate = json.loads((ROOT / "gate.json").read_text())
    if not gate["matches_exp002"]:
        raise RuntimeError("pipeline does not reproduce EXP-002 Wanda mask; stop")
    if (ROOT / "pruning.json").exists():
        return
    model, summary = prune("sap")
    mods = _module_map(model)
    masks = {f"block_{b:02d}.{n}": pack_mask(m.weight.eq(0).detach().cpu()) for (b, n), m in sorted(mods.items())}
    pruned = sum(int(m.weight.eq(0).sum()) for m in mods.values())
    if pruned != EXPECTED_PRUNED:
        raise RuntimeError(f"pruned {pruned} != {EXPECTED_PRUNED}")
    torch.save(masks, ROOT / "masks.pt")
    summary.update(pruned=pruned, sparse_model_sha256=model_sha(model), masks_sha256=sha(ROOT / "masks.pt"),
                   requests_sha256=sha(ROOT / "requests.json"))
    freeze(ROOT / "pruning.json", summary)
    progress("pruned", **summary)


def cmd_eval(shard):
    import torch
    from experiments.dlm_multiscale_ac50.artifacts import read, sha, write
    from experiments.dlm_multiscale_ac50.evaluation import evaluate_requests, generator, grade, task_and_protocol
    from experiments.dlm_loss_aggregation.core import unpack_mask
    from experiments.dlm_loss_aggregation.exp002.run import _module_map, _zero_mask_summary
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    os.environ["WORKER_NAME"] = f"shard{shard}"
    meta = read(ROOT / "pruning.json")
    if sha(ROOT / "masks.pt") != meta["masks_sha256"] or sha(ROOT / "requests.json") != meta["requests_sha256"]:
        raise RuntimeError("masks/requests changed")
    model = load_dense(2048)
    masks = torch.load(ROOT / "masks.pt", weights_only=False)
    with torch.no_grad():
        for (b, n), m in _module_map(model).items():
            m.weight[unpack_mask(masks[f"block_{b:02d}.{n}"]).to(m.weight.device)] = 0
    if _zero_mask_summary(model, "wanda")["mask_hash"] != meta["mask_hash"] or model_sha(model) != meta["sparse_model_sha256"]:
        raise RuntimeError("applied sparse model differs from pruning receipt")
    progress("model_verified", shard=shard)
    task, _, protocol_hash = task_and_protocol(read(LEGACY_CONFIG))
    if protocol_hash != PROTOCOL_HASH:
        raise RuntimeError("protocol differs")
    requests = read(ROOT / "requests.json")["development"][shard::SHARDS]
    evaluation = read(LEGACY_CONFIG)["evaluation"]
    fingerprint = dict(experiment="dlm_wanda_sapcalib50_full", mask_hash=meta["mask_hash"],
                       sparse_model_sha256=meta["sparse_model_sha256"], requests_sha256=meta["requests_sha256"], shard=shard)
    result = evaluate_requests(ROOT / "gsm8k" / f"shard{shard}", requests, fingerprint, protocol_hash, "Wanda-SAPcal",
                               generator(model, tokenizer(), evaluation, MASK_ID),
                               lambda req, text: grade(task, req, text),
                               lambda stage, **v: progress(stage, **v))
    write(ROOT / f"complete_shard{shard}.json", result)


def cmd_report():
    """Merge shards; paired comparisons against EXP-002 Wanda (full) and crosschain A/Multi (primary 1119)."""
    import numpy as np
    from experiments.dlm_multiscale_ac50.artifacts import read, write
    from experiments.dlm_crosschain_control50.prepare import EXPOSED
    rows = {}
    for s in range(SHARDS):
        for r in read(ROOT / "gsm8k" / f"shard{s}" / "predictions.json"):
            rows[r["example_id"]] = r["correct"]
    if sorted(rows) != list(range(1319)):
        raise RuntimeError("incomplete")
    refs = {"Wanda-8x256 (EXP-002)": {}}
    for line in (REPO / "experiments/dlm_loss_aggregation/exp002/results/predictions/wanda.jsonl").read_text().splitlines():
        r = json.loads(line); refs["Wanda-8x256 (EXP-002)"][r["example_id"]] = r["correct"]
    for line in (REPO / "experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl").read_text().splitlines():
        r = json.loads(line); refs.setdefault("Dense (EXP-002)", {})[r["example_id"]] = r["correct"]
    cc = REPO / "experiments/dlm_crosschain_control50/output/gsm8k"
    for arm in ("A", "Multi"):
        refs[f"{arm} (crosschain)"] = {r["example_id"]: r["correct"] for r in read(cc / arm / "predictions.json")}
    primary = [i for i in range(1319) if i not in set(EXPOSED)]
    rng = np.random.default_rng(20260927)
    out = dict(total=len(rows), correct=sum(rows.values()), primary_n=len(primary),
               primary_correct=sum(rows[i] for i in primary), comparisons={})
    from math import comb

    def exact_p(g, n):
        k = min(g, n - g)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    for name, ref in refs.items():
        ids = [i for i in (primary if "crosschain" in name else range(1319)) if i in ref]
        a = np.array([rows[i] for i in ids], int); b = np.array([ref[i] for i in ids], int)
        gain, loss = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        diff = a - b
        boot = [diff[rng.integers(0, len(ids), len(ids))].mean() for _ in range(10000)]
        out["comparisons"][name] = dict(n=len(ids), wanda_sap=int(a.sum()), reference=int(b.sum()), gain=gain, loss=loss,
            diff_pp=100 * diff.mean(), ci95_pp=[100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))],
            exact_p=exact_p(gain, gain + loss) if gain + loss else 1.0)
    write(ROOT / "report.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prune", "eval", "report"])
    p.add_argument("--shard", type=int)
    a = p.parse_args()
    {"prune": cmd_prune, "report": cmd_report}.get(a.cmd, lambda: cmd_eval(a.shard))()
