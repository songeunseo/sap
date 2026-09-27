"""DSA graph search adaptation: frozen DLM masked CE, never GSM8K fitness.

Upstream publishes graph execution/random generation, not its paper's full
performance-guided graph evolution controller. This controller is our explicit
bounded adaptation, NOT an official-search reproduction.
"""
import argparse
import fcntl
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.dlm_owl65 import run as base
from experiments.dlm_owl65.core import exact_row_counts
from experiments.dlm_allocation_baselines65.core import dsa_rates
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write, file_sha256 as sha

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path('/DATA/tmluser1/dlm_allocation_baselines65/sap-dsa-reference/lib/autolayer.py')
CAL = Path('experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json')
HELD = Path('experiments/wanda_failure_characterization/heldout_state_manifest.json')
VERIFY = Path('experiments/projection_capacity_allocation_65/state_verification.json')
FIXED = 'W:(ABSLOG)-(VAR)-(ATAN,ASIN)-(7)'
# These matrix operations cannot safely run on the public pooled 1-D block
# input (e.g. DIAGONAL would allocate an N*N tensor). No tensor subsampling.
UNSAFE = {'GRAM', 'CORREF', 'DIAGONAL', 'DETERMINANT', 'RANK', 'SLOGDET'}


def read(p):
    return json.loads(Path(p).read_text())


def event(stage, **kw):
    value = dict(stage=stage, time=time.time(), **kw)
    write(ROOT/'progress.json', value)
    print(json.dumps(value), flush=True)


def engine_class():
    spec = importlib.util.spec_from_file_location('dsa_public_graph', UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LayerEngine


def graph_parts(graph):
    return graph.split('-')


def feasible(graph):
    return not any(op in graph for op in UNSAFE)


def random_graph(rng, Engine):
    # Public operation vocabulary and public stage cardinalities, with only
    # physically unsafe matrix operations on pooled vectors excluded.
    e = Engine(FIXED)
    stick = [rng.choice(list(e._STICK_OPS)) for _ in range(rng.randint(0, 2))] or ['NO_OP']
    reduce = rng.choice([k for k in e._DEFORM_OPS if k not in UNSAFE])
    post = [rng.choice(list(e._POST_OPS)) for _ in range(rng.randint(1, 2))]
    return f'W:({",".join(stick)})-({reduce})-({",".join(post)})-({rng.randint(7,10)})'


def offspring(rng, parents, seen, Engine, count):
    result = []
    for _ in range(10000):
        a, b = [graph_parts(rng.choice(parents)) for _ in range(2)]
        parts = [rng.choice([x, y]) for x, y in zip(a, b)]
        if rng.random() < .5:
            j = rng.randrange(3)
            parts[j] = graph_parts(random_graph(rng, Engine))[j]
        g = '-'.join(parts)
        if g not in seen and g not in result and feasible(g):
            result.append(g)
        if len(result) == count:
            return result
    raise RuntimeError('Cannot generate distinct offspring')


def freeze():
    prior = base.validate()
    verified = read(VERIFY)
    if verified['status'] != 'verified' or not verified['disjoint']:
        raise RuntimeError('Historical calibration/heldout separation unverified')
    paths = [UPSTREAM, Path(__file__), ROOT/'README.md', ROOT/'test_search.py', ROOT/'run.sh',
             Path('experiments/dlm_allocation_baselines65/core.py'), Path('experiments/dlm_owl65/core.py'),
             base.STATS, CAL, HELD, VERIFY,
             base.SOURCE/'candidate_mask_manifest.json', base.EVAL]
    cfg = dict(model=prior['model'], target=.65, pruned=prior['pruned'],
               weights=prior['weights'], seed=0, population=8, generations=4,
               elites=2, parents=4, mutation_probability=.5, lam=.08,
               fitness='Equal-state mean masked-token CE on frozen 80 calibration states; lower is better',
               not_ppl='exp(CE) is descriptive masked perplexity, NOT AR PPL or unbiased DLM likelihood',
               adaptation='Public DSA graph operators/mapping + our graph-evolution controller; no official controller published in pinned source',
               search_space_exclusion=sorted(UNSAFE),
               exclusion_reason='Unsafe/incompatible matrix operations on full pooled 1-D block scores; no approximate tensors',
               max_new_evaluations=26, protocol_hash=prior['protocol_hash'],
               state_digest=prior['state_digest'], sources={str(p):sha(p) for p in paths})
    base.frozen(ROOT/'config.json', cfg)
    return cfg


def validate():
    c = read(ROOT/'config.json')
    for p, h in c['sources'].items():
        if sha(p) != h:
            raise RuntimeError(f'Frozen source changed: {p}')
    return c


@torch.inference_mode()
def fitness(model, states, candidate, stage='search'):
    device = next(model.parameters()).device
    values = []
    for i, s in enumerate(states):
        x = torch.tensor(s['noisy_ids'], device=device)
        y = torch.tensor(s['clean_ids'], device=device)
        mask = torch.tensor(s['mask'], dtype=torch.bool, device=device)
        if not mask.any():
            raise RuntimeError('Empty masked state')
        logits = model(x).logits
        ce = F.cross_entropy(logits[mask].float(), y[mask]).item()
        if not np.isfinite(ce):
            raise RuntimeError('Nonfinite masked CE')
        values.append(ce)
        event(stage, candidate=candidate, state=i+1, states=len(states), mean_ce=float(np.mean(values)))
    return dict(mean_ce=float(np.mean(values)), state_ce=values,
                descriptive_exp_ce=float(np.exp(np.mean(values))))


@torch.inference_mode()
def run():
    c = validate()
    if (ROOT/'results.json').exists():
        event('complete', correct=read(ROOT/'results.json')['correct'])
        return
    from experiments.projection_capacity_allocation_65.run import load_dense, save_tensor
    from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
    Engine = engine_class()
    event('loading_dense')
    model, mapping = load_dense()
    refs = read(base.SOURCE/'candidate_mask_manifest.json')['entries']
    if list(mapping) != [r['name'] for r in refs]:
        raise RuntimeError('Projection order mismatch')
    stats = torch.load(base.STATS, map_location='cpu', weights_only=False)['statistics']
    dense, orders = {}, {}
    # Keep exact dense weights and int32 row rankings resident on GPU1. ~60 GiB
    # including the model, instead of writing multi-GB checkpoints per candidate.
    for i, ref in enumerate(refs):
        name = ref['name']; w = mapping[name].weight
        dense[name] = w.detach().clone()
        metric = w.float().abs()*stats[name]['overall_uniform'].float().to(w.device).sqrt()[None,:]
        order = torch.argsort(metric, dim=1, stable=True)
        mask = torch.zeros_like(w, dtype=torch.bool).scatter_(1, order[:,:int(w.shape[1]*.65)], True)
        if mask_sha256(pack_mask(mask.cpu())) != ref['masks'][3]['mask_sha256']:
            raise RuntimeError('Historical Uniform65 ranking mismatch')
        orders[name] = order.to(torch.int32)
        del metric, order, mask
        event('ranking', module=i+1, total=224)

    def restore():
        for name, mod in mapping.items():
            mod.weight.copy_(dense[name])

    def apply(counts):
        total = 0
        for ref, k in zip(refs, counts):
            name = ref['name']; w = mapping[name].weight
            w.copy_(dense[name])
            w.scatter_(1, orders[name][:,:k].long(), 0)
            total += k*w.shape[0]
        if total != c['pruned']:
            raise RuntimeError('Exact budget mismatch')

    rng = random.Random(c['seed'])
    history = read(ROOT/'search.json') if (ROOT/'search.json').exists() else dict(candidates={}, generations=[])
    initial = [FIXED]
    while len(initial) < c['population']:
        g = random_graph(rng, Engine)
        if g not in initial:
            initial.append(g)
    population = initial
    states = read(CAL)['states']
    if len(states) != 80:
        raise RuntimeError('Wrong search state count')
    for generation in range(c['generations']):
        if generation:
            ranked = sorted((g for g in population if history['candidates'][g]['status']=='valid'),
                            key=lambda g:history['candidates'][g]['fitness']['mean_ce'])
            if len(ranked) < c['elites']:
                raise RuntimeError('Insufficient valid elites; stop rather than change search budget')
            elites = ranked[:c['elites']]
            parents = ranked[:c['parents']]
            population = elites + offspring(rng, parents, set(history['candidates']), Engine,
                                            c['population']-len(elites))
        pending = [g for g in population if g not in history['candidates']]
        restore()
        scores = {g:[] for g in pending}
        errors = {}
        for b in range(32):
            pooled = torch.cat([(dense[r['name']].float().abs()*stats[r['name']]['overall_uniform'].float().to(next(model.parameters()).device).sqrt()[None,:]).flatten()
                                for r in refs[b*7:b*7+7]])
            for g in pending:
                if g in errors:
                    continue
                try:
                    value = Engine.from_string(g).compute_importance(pooled.clone())
                    if value == -1 or not np.isfinite(value):
                        raise ValueError('Public graph invalid sentinel')
                    scores[g].append(value)
                except (ValueError, RuntimeError, TypeError, IndexError) as exc:
                    errors[g] = str(exc)
            del pooled
            event('graph_statistics', generation=generation+1, block=b+1, total=32)
        for g in pending:
            try:
                if g in errors:
                    raise ValueError(errors[g])
                rates = dsa_rates(scores[g])
                counts, budget = exact_row_counts(refs, rates, c['pruned'])
            except ValueError as exc:
                history['candidates'][g] = dict(status='invalid', error=str(exc), scores=scores[g])
                write(ROOT/'search.json', history)
                continue
            apply(counts)
            f = fitness(model, states, g)
            history['candidates'][g] = dict(status='valid', scores=scores[g], rates=rates.tolist(),
                                           counts=counts, budget=budget, fitness=f)
            write(ROOT/'search.json', history)
        record = dict(generation=generation+1, population=population)
        if generation >= len(history['generations']):
            history['generations'].append(record)
        elif history['generations'][generation] != record:
            raise RuntimeError('Resume population mismatch')
        write(ROOT/'search.json', history)

    winner = min((g for g,r in history['candidates'].items() if r['status']=='valid'),
                 key=lambda g:history['candidates'][g]['fitness']['mean_ce'])
    selected = history['candidates'][winner]
    write(ROOT/'selection.json', dict(graph=winner, selection_metric=c['fitness'], **selected))
    apply(selected['counts'])
    entries = []
    for ref, k in zip(refs, selected['counts']):
        name=ref['name']; mask=torch.zeros_like(mapping[name].weight,dtype=torch.bool)
        mask.scatter_(1, orders[name][:,:k].long(), True)
        packed=pack_mask(mask.cpu()); path=ROOT/'masks'/f'{name}.pt'
        save_tensor(path, packed)
        entries.append(dict(name=name,shape=ref['shape'],weights=ref['weights'],assigned_sparsity=k/ref['shape'][1],
                            selected_mask=dict(path=str(path),file_sha256=sha(path),mask_sha256=mask_sha256(packed),prune_per_row=k,pruned=k*ref['shape'][0])))
    manifest=dict(entries=entries,pruned=c['pruned'],weights=c['weights'])
    write(ROOT/'mask_manifest.json', manifest)
    if not (ROOT/'heldout.json').exists():
        write(ROOT/'heldout.json', fitness(model, read(HELD)['states'], winner, 'heldout_selected_only'))
    # Selection is irrevocably frozen before looking at GSM8K outcomes.
    del dense, orders, stats
    torch.cuda.empty_cache()
    from transformers import AutoTokenizer
    from experiments.dlm_loss_aggregation.exp002.run import load_config, _evaluate_gsm8k, _evaluation_config_hash, _write_jsonl
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    cfg=load_config(base.EVAL); ph,_=_evaluation_config_hash(cfg)
    if ph != c['protocol_hash']:
        raise RuntimeError('GSM8K protocol mismatch')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model']['id'],revision=cfg['model']['revision'],trust_remote_code=True)
    event('gsm8k',completed=0,total=100)
    metrics,rows=_evaluate_gsm8k(model,tokenizer,cfg,'dsa_dlm_search65',100,ph)
    comparisons={}
    for name,path in base.BASELINES.items():
        baseline=base.read_jsonl(path); _validate_rows(rows,baseline,ph)
        comparisons[name]=paired_binary_comparison([x['correct'] for x in baseline],[x['correct'] for x in rows])
    _write_jsonl(ROOT/'predictions.jsonl',rows)
    write(ROOT/'results.json',dict(status='complete',graph=winner,correct=sum(r['correct'] for r in rows),total=100,
                                  metrics=metrics,comparisons=comparisons,config_sha256=sha(ROOT/'config.json'),
                                  predictions_sha256=sha(ROOT/'predictions.jsonl'),manifest_sha256=sha(ROOT/'mask_manifest.json')))
    event('complete',correct=sum(r['correct'] for r in rows))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['freeze','run']); args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            freeze() if args.phase=='freeze' else run()
        except BaseException as exc:
            event('failed',error=f'{type(exc).__name__}: {exc}')
            raise
