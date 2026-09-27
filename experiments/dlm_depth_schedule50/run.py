"""Depth direction / shape causal contrast at exact 50% (layer = block unit), frozen WikiText NELBO.

Only the per-block sparsity vector differs between arms. Everything else is shared with the existing
Uniform / A / Multi / Cross WikiText results: LLaDA-8B-Base rev 0f2787f, frozen native sparse-prefix Wanda
activation vectors (dlm_context_response50/activations), row-wise Wanda inside each projection, exact
3,489,660,928 pruned via the historical (block, input-width) row-count DP, 551 validation chunks, MC128 shared draws.

Arms (all 7 projections of a block share the block rate):
  DIS      rate_t = .5 - .08 (1 - 2t/31)            later blocks sparser (AR-statistics direction)
  EIS      rate_t = .5 + .08 (1 - 2t/31)            earlier blocks sparser (Layer Collapse 2605.06366)
  AmShape  rate_b = .5 - k (c_b - mean c), k = .16 / (max c - min c)
           c_b = pilot A_m cost, delta 10pp (mean over 32 spans of A_m(60%) - A_m(40%)); range matched to EIS
  AmPerm1..3  AmShape rate vector permuted across blocks (numpy default_rng(seed 1/2/3))
Commands: prepare (CPU), worker --arm X (tmux, one GPU), report (CPU).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from experiments.dlm_wikitext_ppl import run as protocol
from experiments.dlm_wikitext_ppl import evaluate as evaluator
from experiments.dlm_wikitext_ppl.core import digest, summarize

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROOT = HERE / "output"
MASKS = Path("/DATA/tmluser1/dlm-depth50")
UNIFORM_MANIFEST = REPO / "experiments/dlm_context_response50/uniform/mask_manifest.json"
ACT_CONFIG = REPO / "experiments/dlm_multiscale_ac50/output/config.json"
PILOT_CACHE = REPO / "research/pilot_controls/levels_cache.npz"
PILOT_CONFIG = REPO / "experiments/dlm_probe_reliability_pilot/output/config.json"
UNIFORM_ROWS = REPO / "experiments/dlm_ppl50/uniform/validation/blocks"
EXISTING = {"A": REPO / "experiments/dlm_crosschain_wikitext50/output/A/blocks",
            "Multi": REPO / "experiments/dlm_crosschain_wikitext50/output/Multi/blocks",
            "Cross": REPO / "experiments/dlm_crosschain_wikitext50/output/Cross/blocks"}
ARMS = ("DIS", "EIS", "AmShape", "AmPerm1", "AmPerm2", "AmPerm3")
TARGET = 3489660928
EPS, WIDTH = 0.08, 0.16
PRIMARY = [("EIS", "Uniform"), ("DIS", "Uniform"), ("AmShape", "EIS"),
           ("AmShape", "AmPerm1"), ("AmShape", "AmPerm2"), ("AmShape", "AmPerm3")]
SESOI = 0.01
BOOT_SEED, BOOT_DRAWS = 20260927, 10000


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return protocol.sha(path)


def schedules():
    t = np.arange(32)
    lin = EPS * (1 - 2 * t / 31)
    z = np.load(PILOT_CACHE)
    order = [round(float(x), 2) for x in z["order"]]
    cost = (z["A_m"][:, order.index(0.60)] - z["A_m"][:, order.index(0.40)]).mean(1)
    k = WIDTH / (cost.max() - cost.min())
    shape = 0.5 - k * (cost - cost.mean())
    rates = {"DIS": 0.5 - lin, "EIS": 0.5 + lin, "AmShape": shape}
    for s in (1, 2, 3):
        rates[f"AmPerm{s}"] = shape[np.random.default_rng(s).permutation(32)]
    return rates, cost


def prepare():
    from experiments.dlm_owl65.core import exact_row_counts
    if (ROOT / "config.json").exists():
        raise RuntimeError("already prepared")
    refs = read(UNIFORM_MANIFEST)["entries"]
    rates, cost = schedules()
    arms = {}
    for arm, r in rates.items():
        if not np.isclose(r.mean(), 0.5, atol=1e-12) or r.min() < 0.3 or r.max() > 0.7:
            raise RuntimeError(f"bad schedule {arm}")
        counts, budget = exact_row_counts(refs, r.tolist(), TARGET)
        arms[arm] = dict(rates=r.tolist(), row_counts=counts, budget=budget,
                         early8=float(100 * r[:8].mean()), late8=float(100 * r[-8:].mean()))
    activations = read(ACT_CONFIG)["activations"]
    pcfg = protocol.validate()
    sources = [Path(__file__), HERE / "run.sh", UNIFORM_MANIFEST, PILOT_CACHE, PILOT_CONFIG,
               protocol.ROOT / "config.json", protocol.ROOT / "corpus_manifest.json", Path(evaluator.__file__),
               REPO / "experiments/dlm_owl65/core.py", REPO / "experiments/dlm_ppl50/sequential.py"]
    cfg = dict(status="frozen", purpose="depth direction/shape causal contrast (exploratory, single calibration)",
               unit="block (all 7 projections share the block rate); row-wise Wanda inside projections",
               arms=list(ARMS), eps=EPS, width=WIDTH, pilot_cost_Am_d10=cost.tolist(), schedules=arms,
               primary=PRIMARY, sesoi=SESOI, multiple_comparison="Holm over primary",
               bootstrap=dict(unit="article", seed=BOOT_SEED, draws=BOOT_DRAWS),
               reference_uniform=str(UNIFORM_ROWS), existing=dict((k, str(v)) for k, v in EXISTING.items()),
               model=pcfg["model"], mc_samples=pcfg["mc_samples"], target=TARGET,
               activations={p: sha(p) for p in activations},
               sources={str(p): sha(p) for p in sources},
               deviations=["random-permutation null uses 3 permutations (MTH-006 asks >=5) to fit the approved GPU budget",
                           "single calibration (historical 80 corrupted states), as the existing Uniform/A/Multi arms"])
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol.frozen(ROOT / "config.json", cfg)
    for arm, v in arms.items():
        print(f"{arm:8s} early8={v['early8']:.2f} late8={v['late8']:.2f} min={min(v['rates']):.3f} max={max(v['rates']):.3f}")
    return cfg


def validate():
    cfg = read(ROOT / "config.json")
    for path, value in list(cfg["sources"].items()) + list(cfg["activations"].items()):
        if sha(path) != value:
            raise RuntimeError("frozen source changed: " + path)
    protocol.validate()
    return cfg


def block_path(arm, source):
    return ROOT / arm / "blocks" / (digest(source["block_id"])[:20] + ".json")


def progress(arm, stage, **kw):
    protocol.write(ROOT / arm / "progress.json", dict(arm=arm, stage=stage, updated=time.time(),
                                                      gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), **kw))


def worker(arm):
    import torch
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.dlm_ppl50.sequential import wanda_mask
    from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    if arm not in ARMS or not os.environ.get("TMUX") or torch.cuda.device_count() != 1:
        raise RuntimeError("worker needs a known arm, tmux and exactly one GPU")
    cfg = validate()
    pcfg = protocol.validate()
    folder = ROOT / arm
    folder.mkdir(parents=True, exist_ok=True)
    corpus = read(protocol.ROOT / "corpus_manifest.json")["splits"]["validation"]
    refs = read(UNIFORM_MANIFEST)["entries"]
    counts = cfg["schedules"][arm]["row_counts"]
    progress(arm, "loading_model")
    model, mapping = load_dense()
    if list(mapping) != [r["name"] for r in refs]:
        raise RuntimeError("module order mismatch")
    acts = [torch.load(p, map_location="cpu", weights_only=False) for p in cfg["activations"]]
    entries, pruned = [], 0
    with torch.inference_mode():
        for i, ref in enumerate(refs):
            b, name = i // 7, ref["name"]
            weight = mapping[name].weight
            act = acts[b][name].to(weight.device)
            half = wanda_mask(weight, act, ref["selected_mask"]["prune_per_row"])
            if mask_sha256(pack_mask(half.cpu())) != ref["selected_mask"]["mask_sha256"]:
                raise RuntimeError("frozen ranking does not reproduce Uniform: " + name)
            mask = wanda_mask(weight, act, counts[i])
            if not torch.all(mask.sum(1) == counts[i]):
                raise RuntimeError("row budget violated")
            packed = pack_mask(mask.cpu())
            path = MASKS / arm / f"{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(packed, path)
            entries.append(dict(name=name, shape=ref["shape"], weights=ref["weights"],
                                selected_mask=dict(path=str(path), file_sha256=sha(path), mask_sha256=mask_sha256(packed),
                                                   prune_per_row=counts[i], pruned=counts[i] * ref["shape"][0])))
            weight.masked_fill_(mask, 0)
            pruned += int(mask.sum())
    if pruned != TARGET:
        raise RuntimeError(f"pruned {pruned} != {TARGET}")
    manifest = dict(arm=arm, config_sha256=sha(ROOT / "config.json"), entries=entries, pruned=pruned,
                    sparse_model_sha256=model_sha(model))
    protocol.frozen(folder / "mask_manifest.json", manifest)
    run_cfg = dict(arm=arm, split="validation", config_sha256=sha(ROOT / "config.json"),
                   mask_manifest_sha256=sha(folder / "mask_manifest.json"),
                   sparse_model_sha256=manifest["sparse_model_sha256"],
                   protocol_config_sha256=sha(protocol.ROOT / "config.json"), mc_samples=128)
    protocol.frozen(folder / "config.json", run_cfg)
    ch = sha(folder / "config.json")
    for i, source in enumerate(corpus):
        path = block_path(arm, source)
        if path.exists():
            evaluator.verify_row(read(path), source, pcfg, ch)
            continue
        row = evaluator.seal(dict(evaluator.evaluate_block(model, source, pcfg), config_sha256=ch))
        evaluator.verify_row(row, source, pcfg, ch)
        protocol.frozen(path, row)
        progress(arm, "evaluating", completed=i + 1, total=len(corpus))
    rows = [read(block_path(arm, s)) for s in corpus]
    result = dict(status="complete", arm=arm, config_sha256=ch, summary=summarize(rows),
                  sparse_model_sha256=manifest["sparse_model_sha256"])
    protocol.frozen(folder / "results.json", result)
    progress(arm, "complete", completed=len(corpus), total=len(corpus), summary=result["summary"])


def _rows(folder):
    out = {}
    for f in Path(folder).glob("*.json"):
        r = read(f)
        out[r["block_id"]] = r
    return out


def report():
    cfg = validate()
    data = {"Uniform": _rows(UNIFORM_ROWS)}
    for k, v in EXISTING.items():
        data[k] = _rows(v)
    for arm in ARMS:
        if not (ROOT / arm / "results.json").exists():
            raise RuntimeError("incomplete arm " + arm)
        data[arm] = _rows(ROOT / arm / "blocks")
    ids = sorted(data["Uniform"])
    for k, rows in data.items():
        if sorted(rows) != ids or any(rows[i]["mask_sha256"] != data["Uniform"][i]["mask_sha256"] for i in ids):
            raise RuntimeError("draw/coverage mismatch: " + k)
    tok = np.array([data["Uniform"][i]["tokens"] for i in ids], float)
    art = [re.match(r"validation:article(\d+):", i).group(1) for i in ids]
    arts = sorted(set(art), key=int)
    idx = {a: [k for k, x in enumerate(art) if x == a] for a in arts}
    a_tok = np.array([tok[idx[a]].sum() for a in arts])
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(arts), size=(BOOT_DRAWS, len(arts)))
    nelbo = {k: float((tok * np.array([rows[i]["token_nelbo"] for i in ids])).sum() / tok.sum()) for k, rows in data.items()}

    def contrast(a, b):
        d = np.array([data[a][i]["token_nelbo"] - data[b][i]["token_nelbo"] for i in ids])
        s = np.array([(tok[idx[x]] * d[idx[x]]).sum() for x in arts])
        boot = s[draws].sum(1) / a_tok[draws].sum(1)
        return dict(delta=float(s.sum() / a_tok.sum()), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                    p=float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
                    articles_better=int((s < 0).sum()), articles_worse=int((s > 0).sum()))
    primary = {f"{a}-{b}": contrast(a, b) for a, b in PRIMARY}
    order = sorted(primary, key=lambda k: primary[k]["p"])
    run = 0.0
    for j, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - j) * primary[k]["p"]))
        primary[k]["holm_p"] = run
        primary[k]["exceeds_sesoi"] = abs(primary[k]["delta"]) >= SESOI
    secondary = {f"{a}-{b}": contrast(a, b) for a, b in
                 [("AmShape", "Uniform"), ("AmShape", "A"), ("AmShape", "Multi"), ("EIS", "A"), ("EIS", "Multi"),
                  ("Multi", "A"), ("DIS", "EIS")]}
    out = dict(config_sha256=sha(ROOT / "config.json"), nelbo=nelbo, primary=primary, secondary=secondary,
               note="exploratory: single calibration, development validation reused; not causal PPL")
    protocol.write(ROOT / "report.json", out)
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "worker", "report"])
    p.add_argument("--arm")
    a = p.parse_args()
    if a.cmd != "worker":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    {"prepare": prepare, "report": report}.get(a.cmd, lambda: worker(a.arm))()


if __name__ == "__main__":
    main()
