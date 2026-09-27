"""Redesigned C (R_share) allocations at exact 50%, block unit, frozen WikiText NELBO.

Identical backend to experiments/dlm_depth_schedule50 (same frozen Wanda activations, row-wise Wanda,
row-count DP, 551 chunks, MC128 shared draws); only the per-block rate vector differs.
  RShape: rate_b = .5 - k (c_b - mean c), c = pilot R_share cost (delta 10pp: R(60%) - R(40%), mean of 32 spans)
  AplusR: same mapping with c = z(A_m cost) + z(R_share cost), z = across-block standardization
k sets max-min rate = 16pp (as EIS / AmShape). Pre-registration: Obsidian Hypotheses/2026-09-27-C-Formula-Redesign.
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
MASKS = Path("/DATA/tmluser1/dlm-c-rshape50")
UNIFORM_MANIFEST = REPO / "experiments/dlm_context_response50/uniform/mask_manifest.json"
ACT_CONFIG = REPO / "experiments/dlm_multiscale_ac50/output/config.json"
VARIANT_LEVELS = REPO / "research/c_variants_2026-09-27/levels.npz"
DEPTH = REPO / "experiments/dlm_depth_schedule50/output"
UNIFORM_ROWS = REPO / "experiments/dlm_ppl50/uniform/validation/blocks"
XC = REPO / "experiments/dlm_crosschain_wikitext50/output"
ARMS = ("RShape", "AplusR")
TARGET, WIDTH = 3489660928, 0.16
PRIMARY = [("AplusR", "AmShape"), ("RShape", "Uniform"), ("AplusR", "Uniform")]
BOOT_SEED, BOOT_DRAWS, SESOI = 20260927, 10000, 0.01


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return protocol.sha(path)


def schedules():
    z = np.load(VARIANT_LEVELS)          # [block, rate(40,45,55,60), span]
    ca = (z["A_m"][:, 3] - z["A_m"][:, 0]).mean(1)
    cr = (z["R_share"][:, 3] - z["R_share"][:, 0]).mean(1)
    zs = lambda c: (c - c.mean()) / c.std()
    shape = lambda c: 0.5 - WIDTH / (c.max() - c.min()) * (c - c.mean())
    return {"RShape": shape(cr), "AplusR": shape(zs(ca) + zs(cr))}, ca, cr


def prepare():
    from experiments.dlm_owl65.core import exact_row_counts
    if (ROOT / "config.json").exists():
        raise RuntimeError("already prepared")
    refs = read(UNIFORM_MANIFEST)["entries"]
    rates, ca, cr = schedules()
    arms = {}
    for arm, r in rates.items():
        counts, budget = exact_row_counts(refs, r.tolist(), TARGET)
        arms[arm] = dict(rates=r.tolist(), row_counts=counts, budget=budget,
                         early8=float(100 * r[:8].mean()), late10=float(100 * r[22:].mean()))
    pcfg = protocol.validate()
    sources = [Path(__file__), HERE / "run.sh", UNIFORM_MANIFEST, VARIANT_LEVELS,
               REPO / "research/c_variants_2026-09-27/analyze.py",
               protocol.ROOT / "config.json", protocol.ROOT / "corpus_manifest.json", Path(evaluator.__file__),
               REPO / "experiments/dlm_owl65/core.py", REPO / "experiments/dlm_ppl50/sequential.py"]
    cfg = dict(status="frozen", purpose="redesigned C (R_share) allocations, exploratory single calibration",
               arms=list(ARMS), width=WIDTH, cost_A_m_d10=ca.tolist(), cost_R_share_d10=cr.tolist(), schedules=arms,
               primary=PRIMARY, sesoi=SESOI, bootstrap=dict(unit="article", seed=BOOT_SEED, draws=BOOT_DRAWS),
               model=pcfg["model"], target=TARGET,
               activations={p: sha(p) for p in read(ACT_CONFIG)["activations"]},
               sources={str(p): sha(p) for p in sources})
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol.frozen(ROOT / "config.json", cfg)
    for arm, v in arms.items():
        print(f"{arm:7s} early8={v['early8']:.2f} late10={v['late10']:.2f} min={min(v['rates']):.3f} max={max(v['rates']):.3f}")


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
            name = ref["name"]
            weight = mapping[name].weight
            act = acts[i // 7][name].to(weight.device)
            if mask_sha256(pack_mask(wanda_mask(weight, act, ref["selected_mask"]["prune_per_row"]).cpu())) != ref["selected_mask"]["mask_sha256"]:
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
    return {read(f)["block_id"]: read(f) for f in Path(folder).glob("*.json")}


def report():
    validate()
    data = {"Uniform": _rows(UNIFORM_ROWS)}
    for k in ("A", "Multi", "Cross"):
        data[k] = _rows(XC / k / "blocks")
    for k in ("DIS", "EIS", "AmShape", "AmPerm1", "AmPerm2", "AmPerm3"):
        if (DEPTH / k / "results.json").exists():
            data[k] = _rows(DEPTH / k / "blocks")
    for arm in ARMS:
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
    nelbo = {k: float((tok * np.array([r[i]["token_nelbo"] for i in ids])).sum() / tok.sum()) for k, r in data.items()}

    def contrast(a, b):
        d = np.array([data[a][i]["token_nelbo"] - data[b][i]["token_nelbo"] for i in ids])
        s = np.array([(tok[idx[x]] * d[idx[x]]).sum() for x in arts])
        boot = s[draws].sum(1) / a_tok[draws].sum(1)
        return dict(delta=float(s.sum() / a_tok.sum()), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                    p=float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
                    articles_better=int((s < 0).sum()), articles_worse=int((s > 0).sum()))
    out = dict(nelbo=nelbo, primary={}, secondary={}, note="exploratory; AmShape contrasts need depth_schedule50 AmShape")
    prim = [(a, b) for a, b in PRIMARY if a in data and b in data]
    res = {f"{a}-{b}": contrast(a, b) for a, b in prim}
    run = 0.0
    for j, k in enumerate(sorted(res, key=lambda k: res[k]["p"])):
        run = max(run, min(1.0, (len(res) - j) * res[k]["p"]))
        res[k]["holm_p"] = run
        res[k]["exceeds_sesoi"] = abs(res[k]["delta"]) >= SESOI
    out["primary"] = res
    for a, b in [("AplusR", "Multi"), ("AplusR", "A"), ("AplusR", "EIS"), ("RShape", "AmShape"), ("RShape", "DIS")]:
        if a in data and b in data:
            out["secondary"][f"{a}-{b}"] = contrast(a, b)
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
