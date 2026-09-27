"""Probe reliability pilot: can finite block probes rank the 32 LLaDA blocks reproducibly?

Same background/ranking as the multiscale A+C probes (historical native sparse-prefix Uniform50,
frozen Wanda activations); changes only (i) 32 new clean WikiText-2 train spans, (ii) perturbation
width 2/5/10pp, (iii) readouts stored for every position so masked-position metrics can be compared
with the original 8-query readout. Commands: prepare (CPU), gpu (tmux, one GPU), analyze (CPU).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
MS = REPO / "experiments/dlm_multiscale_ac50/output"
UNIFORM_MANIFEST = REPO / "experiments/dlm_context_response50/uniform/mask_manifest.json"
UNIFORM_SHA = "e0c40d9d32d5b87b7bb43d7898d1baae6e0d9fda84b7dfa29f897807a646598f"
EXISTING_SPANS = [REPO / "experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json",
                  REPO / "experiments/wanda_failure_characterization/heldout_state_manifest.json"]
MASK_ID = 126336
SEED = 20260927
N_SPANS, SEQLEN, NGRAM = 32, 256, 32
RATES = [0.40, 0.45, 0.48, 0.52, 0.55, 0.60]
DELTAS = {2: (0.48, 0.52), 5: (0.45, 0.55), 10: (0.40, 0.60)}
READOUTS = ("A_q", "Multi_q", "A_m", "CE_m")
DECISION = dict(
    usable="two-way reliability at 32 spans >= 0.8 AND mean 16/16 split-half rho >= 0.6",
    marginal="0.5 <= reliability < 0.8: report spans needed for 0.8 (extrapolation, not a guarantee)",
    unusable="all readout x delta reliability < 0.5: block-level probe allocation dropped from experiment 3",
    consistency="A_q at delta 2 with 8 spans is expected to be near 0, matching the CPU audit",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def progress(stage, **values):
    row = dict(stage=stage, time=time.time(), pid=os.getpid(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), **values)
    write(OUT / "progress.json", row)
    print(json.dumps(row), flush=True)


# ---------------------------------------------------------------- CPU preparation
def ngrams(ids):
    return {tuple(ids[i:i + NGRAM]) for i in range(len(ids) - NGRAM + 1)}


def cmd_prepare():
    from transformers import AutoTokenizer
    from lib.data import get_wikitext2
    from experiments.dlm_multiscale_ac50.core import clean_sequences, make_bank
    if (OUT / "config.json").exists():
        raise RuntimeError("already prepared; use a new output directory to change inputs")
    existing = set()
    for path in EXISTING_SPANS:
        for ids in clean_sequences(read(path)).values():
            existing |= ngrams(ids)
    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", revision="0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                                        trust_remote_code=True, local_files_only=True)
    pool, _ = get_wikitext2(4 * N_SPANS, SEED, SEQLEN, tok)
    spans, used, rejected = [], set(existing), 0
    for inp, _ in pool:
        ids = inp[0].tolist()
        grams = ngrams(ids)
        if len(ids) != SEQLEN or MASK_ID in ids or grams & used:
            rejected += 1
            continue
        spans.append(ids)
        used |= grams
        if len(spans) == N_SPANS:
            break
    if len(spans) != N_SPANS:
        raise RuntimeError("not enough non-overlapping spans")
    source = dict(mask_id=MASK_ID, states=[dict(sequence_index=100 + k, clean_ids=ids) for k, ids in enumerate(spans)])
    settings = copy.deepcopy(read(MS / "config.json")["plan"]["bank"])
    settings.update(sequences_per_split=N_SPANS, states_per_split=N_SPANS * 16, calibration_seed=SEED)
    bank = make_bank(source, settings, "calibration")
    write(OUT / "spans.json", dict(seed=SEED, rejected_before_fill=rejected, ngram=NGRAM,
                                   spans={str(100 + k): ids for k, ids in enumerate(spans)}))
    write(OUT / "bank.json", bank)
    activations = read(MS / "config.json")["activations"]
    config = dict(
        purpose="probe reliability pilot (C gate follow-up)", seed=SEED, rates=RATES,
        deltas={str(k): v for k, v in DELTAS.items()}, readouts=READOUTS, decision=DECISION,
        background=dict(manifest=str(UNIFORM_MANIFEST), manifest_sha256=sha(UNIFORM_MANIFEST),
                        sparse_model_sha256=UNIFORM_SHA),
        activations={p: sha(p) for p in activations},
        bank=dict(path=str(OUT / "bank.json"), sha256=sha(OUT / "bank.json"), states=bank["states"]),
        spans=dict(path=str(OUT / "spans.json"), sha256=sha(OUT / "spans.json")),
        code_sha256=sha(__file__), forward_batch=1,
        planned_forwards=bank["states"] * (2 + 32 * len(RATES)))
    write(OUT / "config.json", config)
    print(json.dumps(dict(states=bank["states"], rejected=rejected, planned_forwards=config["planned_forwards"])))


def validate(check_code=True):
    config = read(OUT / "config.json")
    for key in ("bank", "spans"):
        if sha(config[key]["path"]) != config[key]["sha256"]:
            raise RuntimeError(f"{key} changed")
    if sha(config["background"]["manifest"]) != config["background"]["manifest_sha256"]:
        raise RuntimeError("background manifest changed")
    for path, digest in config["activations"].items():
        if sha(path) != digest:
            raise RuntimeError(f"activation changed: {path}")
    if check_code and sha(__file__) != config["code_sha256"]:
        raise RuntimeError("code changed after freeze")
    return config


# ---------------------------------------------------------------- GPU worker
def cmd_gpu():
    import torch
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.dlm_ppl50.sequential import wanda_mask
    from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
    from experiments.dlm_context_response50.core import log_odds
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    if not os.environ.get("TMUX"):
        raise RuntimeError("run inside tmux")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU required")
    config = validate()
    bank, spans = read(config["bank"]["path"]), read(config["spans"]["path"])["spans"]
    states = [(node["input_ids"], spans[str(chain["sequence_index"])]) for chain in bank["chains"] for node in chain["nodes"]]
    refs = read(UNIFORM_MANIFEST)["entries"]
    progress("loading_dense")
    model, mapping = load_dense()
    model.eval()
    device = next(model.parameters()).device
    if list(mapping) != [r["name"] for r in refs]:
        raise RuntimeError("module order mismatch")
    activations = {b: torch.load(p, map_location="cpu", weights_only=False)
                   for b, p in enumerate(config["activations"])}

    @torch.inference_mode()
    def readout(label):
        path = OUT / "readouts" / f"{label}.pt"
        receipt = OUT / "readouts" / f"{label}.json"
        if path.exists() and receipt.exists() and read(receipt)["sha256"] == sha(path):
            return torch.load(path, weights_only=False)
        values = torch.empty((len(states), SEQLEN), dtype=torch.float32)
        started = time.time()
        for i, (ids, clean) in enumerate(states):
            logits = model(torch.tensor([ids], device=device)).logits[0]
            values[i] = log_odds(logits, clean).cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, path)
        write(receipt, dict(label=label, sha256=sha(path), seconds=time.time() - started, forwards=len(states)))
        return values

    dense = readout("dense")
    with torch.inference_mode():
        repeat = log_odds(model(torch.tensor([states[0][0]], device=device)).logits[0], states[0][1]).cpu()
    if not torch.equal(repeat, dense[0]):
        raise RuntimeError("dense repeat readout mismatch")
    progress("dense_done")

    originals = {n: m.weight.detach().cpu().clone() for n, m in mapping.items()}

    @torch.inference_mode()
    def set_block(block, rate=None):
        """rate None -> the frozen Uniform50 Wanda mask (hash-checked); else Wanda at int(in*rate) per row."""
        for ref in refs[block * 7:(block + 1) * 7]:
            name, weight = ref["name"], mapping[ref["name"]].weight
            weight.copy_(originals[name].to(weight.device))
            count = ref["selected_mask"]["prune_per_row"] if rate is None else int(ref["shape"][1] * rate)
            mask = wanda_mask(weight, activations[block][name].to(weight.device), count)
            if not torch.all(mask.sum(1) == count):
                raise RuntimeError("row budget violated")
            if rate is None and mask_sha256(pack_mask(mask.cpu())) != ref["selected_mask"]["mask_sha256"]:
                raise RuntimeError(f"frozen Wanda ranking does not reproduce Uniform: {name}")
            weight.masked_fill_(mask, 0)

    for block in range(32):
        set_block(block)
    if model_sha(model) != UNIFORM_SHA:
        raise RuntimeError("background differs from historical Uniform50")
    readout("uniform50")
    progress("background_verified")
    done = 0
    for block in range(32):
        for rate in RATES:
            label = f"b{block:02d}_r{round(rate * 100)}"
            if not (OUT / "readouts" / f"{label}.json").exists():
                set_block(block, rate)
            readout(label)
            done += 1
            progress("probes", completed=done, total=32 * len(RATES), block=block, rate=rate)
        set_block(block)
    if model_sha(model) != UNIFORM_SHA:
        raise RuntimeError("background changed during probes")
    write(OUT / "gpu_complete.json", dict(config_sha256=sha(OUT / "config.json"), conditions=2 + 32 * len(RATES),
                                          forwards=len(states) * (2 + 32 * len(RATES)) + 1))
    progress("complete")


# ---------------------------------------------------------------- CPU analysis
def sequence_levels(values, dense, bank):
    """Per-sequence level of each readout for one condition. values/dense: [states, 256] numpy."""
    from experiments.dlm_multiscale_ac50.core import metrics
    q = np.asarray([c["query"] for c in bank["chains"] for _ in c["nodes"]])
    rows = np.arange(len(q))[:, None]
    ms = metrics(values[rows, q].tolist(), dense[rows, q].tolist(), bank)
    out = {"A_q": [s["metrics"]["A"] for s in ms["sequences"]], "Multi_q": [s["metrics"]["Multi"] for s in ms["sequences"]]}
    masked = np.asarray([[t == MASK_ID for t in n["input_ids"]] for c in bank["chains"] for n in c["nodes"]])
    per_state = {"A_m": ((values - dense) ** 2 * masked).sum(1) / masked.sum(1),
                 "CE_m": (np.logaddexp(0, -values) * masked).sum(1) / masked.sum(1)}
    seq = np.asarray([c["sequence_index"] for c in bank["chains"] for _ in c["nodes"]])
    for key, v in per_state.items():
        out[key] = [float(v[seq == s].mean()) for s in sorted(set(seq.tolist()))]
    return out


def reliability(cost, rng, splits=500):
    """cost [blocks, seqs]. Two-way block x sequence variance components and random half-split Spearman."""
    from scipy.stats import spearmanr
    nb, ns = cost.shape
    resid = cost - cost.mean(1, keepdims=True) - cost.mean(0, keepdims=True) + cost.mean()
    within = float((resid ** 2).sum() / ((nb - 1) * (ns - 1)))
    obs = float(cost.mean(1).var(ddof=1))
    true = obs - within / ns
    rel = lambda n: max(true, 0.0) / (max(true, 0.0) + within / n) if true > 0 else 0.0
    rhos = []
    for _ in range(splits):
        perm = rng.permutation(ns)
        a, b = perm[: ns // 2], perm[ns // 2:]
        rhos.append(spearmanr(cost[:, a].mean(1), cost[:, b].mean(1))[0])
    r = float(np.mean(rhos))
    return dict(reliability=rel(ns), reliability_8=rel(8), est_true_var=true, residual_var=within,
                spans_for_0_8=(4 * within / true) if true > 0 else None,
                half_rho_mean=r, half_rho_q05=float(np.quantile(rhos, .05)), half_rho_q95=float(np.quantile(rhos, .95)),
                spearman_brown=2 * r / (1 + r) if r > -1 else None)


def cmd_analyze():
    import torch
    from scipy.stats import spearmanr
    config = validate(check_code=False)  # analysis may be revised after GPU; its hash is recorded in result.json
    if not (OUT / "gpu_complete.json").exists():
        raise RuntimeError("GPU phase incomplete")
    bank, refs = read(config["bank"]["path"]), read(UNIFORM_MANIFEST)["entries"]
    load = lambda label: torch.load(OUT / "readouts" / f"{label}.pt", weights_only=False).double().numpy()
    for label in ["dense", "uniform50"] + [f"b{b:02d}_r{round(r * 100)}" for b in range(32) for r in RATES]:
        if read(OUT / "readouts" / f"{label}.json")["sha256"] != sha(OUT / "readouts" / f"{label}.pt"):
            raise RuntimeError(f"readout changed: {label}")
    dense = load("dense")
    levels = {}
    for block in range(32):
        for rate in RATES:
            levels[(block, rate)] = sequence_levels(load(f"b{block:02d}_r{round(rate * 100)}"), dense, bank)
    uniform = sequence_levels(load("uniform50"), dense, bank)
    pruned = lambda block, rate: sum(int(r["shape"][1] * rate) * r["shape"][0] for r in refs[block * 7:(block + 1) * 7])
    rng = np.random.default_rng(SEED)
    result = dict(config_sha256=sha(OUT / "config.json"), analysis_code_sha256=sha(__file__), decision_rule=DECISION, uniform_levels=uniform, cells={})
    means = {}
    for key in READOUTS:
        for delta, (lo, hi) in DELTAS.items():
            cost = np.asarray([(np.asarray(levels[(b, hi)][key]) - np.asarray(levels[(b, lo)][key]))
                               / (pruned(b, hi) - pruned(b, lo)) for b in range(32)])
            cell = reliability(cost, rng)
            cell.update(negative_costs=int((cost.mean(1) < 0).sum()))
            # One-sided costs against the measured Uniform50 background (linearity check).
            up = np.asarray([np.mean(levels[(b, hi)][key]) - np.mean(uniform[key]) for b in range(32)])
            down = np.asarray([np.mean(uniform[key]) - np.mean(levels[(b, lo)][key]) for b in range(32)])
            cell["up_down_spearman"] = float(spearmanr(up, down)[0])
            result["cells"][f"{key}|d{delta}"] = cell
            means[(key, delta)] = cost.mean(1)
    result["cross_delta_spearman"] = {f"{k}|d{a}~d{b}": float(spearmanr(means[(k, a)], means[(k, b)])[0])
                                      for k in READOUTS for a, b in ((2, 5), (5, 10), (2, 10))}
    usable = [c for c, v in result["cells"].items() if v["reliability"] >= 0.8 and v["half_rho_mean"] >= 0.6]
    marginal = [c for c, v in result["cells"].items() if c not in usable and v["reliability"] >= 0.5]
    result["decision"] = dict(usable=usable, marginal=marginal,
                              verdict="usable" if usable else "marginal" if marginal else "unusable",
                              consistency_A_q_d2_reliability_8=result["cells"]["A_q|d2"]["reliability_8"])
    write(OUT / "result.json", result)
    print(json.dumps(result["decision"], indent=1))
    for c, v in result["cells"].items():
        print(f"{c:12s} rel32={v['reliability']:.3f} rel8={v['reliability_8']:.3f} half_rho={v['half_rho_mean']:+.3f} "
              f"neg={v['negative_costs']:2d} up~down={v['up_down_spearman']:+.3f} n80={v['spans_for_0_8']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "gpu", "analyze"])
    {"prepare": cmd_prepare, "gpu": cmd_gpu, "analyze": cmd_analyze}[p.parse_args().cmd]()
