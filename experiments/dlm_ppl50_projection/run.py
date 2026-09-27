"""Projection-level OWL/DSA allocation under the frozen DLM PPL50 protocol."""
import argparse
import fcntl
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_ppl50 import run as prior, sequential as seq
from experiments.dlm_owl65 import run as owl
from experiments.dlm_dsa_search65 import run as graphlib
from experiments.dlm_lsa_projection65.core import exact_projection_rows
from experiments.dlm_wikitext_ppl import run as protocol

ROOT = Path(__file__).resolve().parent
TARGET = prior.TARGET
METHODS = ('owl_projection', 'dsa_projection')
read, write, sha = seq.read, seq.write, seq.sha


def event(method, stage, **kw):
    value = dict(method=method, stage=stage, time=time.time(), pid=os.getpid(), **kw)
    write(ROOT / method / 'progress.json', value)
    print(value, flush=True)


def weighted_rates(values, weights, target=.5, lam=.08):
    """Public minmax OWL/DSA mapping with a parameter-weighted center.

    The 32 original blocks have equal size; 224 projections do not. Weighted
    centering is the necessary budget-preserving extension of mean centering.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim != 1 or v.shape != w.shape or not np.isfinite(v).all() or not np.isfinite(w).all() or (w <= 0).any() or np.ptp(v) == 0:
        raise ValueError('Invalid projection importance')
    z = (v - v.min()) / np.ptp(v) * lam * 2
    rates = target + np.average(z, weights=w) - z
    if (rates <= 0).any() or (rates >= 1).any():
        raise ValueError('Out-of-range projection sparsity')
    return rates


def dense_scores(mapping, refs, stats):
    for ref in refs:
        name = ref['name']
        weight = mapping[name].weight
        a = stats[name]['overall_uniform'].float().to(weight.device)
        if a.shape != (weight.shape[1],) or not torch.isfinite(a).all() or (a < 0).any():
            raise RuntimeError('Invalid frozen dense Wanda activation')
        yield weight.float().abs() * a.sqrt()[None, :]


def owl_allocation(mapping, refs, stats):
    ratios = []
    for score in dense_scores(mapping, refs, stats):
        ratios.append(float((score > score.mean() * 5).sum()) / score.numel() * 100)
    rates = weighted_rates(ratios, [r['weights'] for r in refs])
    counts, budget = exact_projection_rows(refs, rates, TARGET)
    return counts, dict(outlier_percent=ratios, ideal_projection_sparsities=rates.tolist(), budget=budget,
                        M=5, lam=.08, mapping='projection-local threshold; parameter-weighted minmax center')


@torch.inference_mode()
def dsa_search(model, mapping, refs, states, cfg, stats):
    settings = cfg['dsa']
    rng = random.Random(settings['seed'])
    Engine = graphlib.engine_class()
    path = ROOT / 'dsa_projection' / 'search.json'
    history = read(path) if path.exists() else dict(candidates={}, generations=[])
    dense = {n: m.weight.detach().cpu().clone() for n, m in mapping.items()}

    def restore():
        for n, m in mapping.items():
            m.weight.copy_(dense[n])

    def sample():
        return seq.canonical(graphlib.random_graph(rng, Engine))

    for generation in range(settings['generations']):
        if generation < len(history['generations']):
            record = history['generations'][generation]
            rng.setstate(seq.tuples(record['rng_after_population']))
            population = record['population']
        else:
            if generation == 0:
                population = [seq.canonical(graphlib.FIXED)]
                while len(population) < settings['population']:
                    graph = sample()
                    if graph not in population:
                        population.append(graph)
            else:
                previous = history['generations'][-1]['population']
                ranked = sorted((g for g in previous if history['candidates'][g]['status'] == 'valid'),
                                key=lambda g: history['candidates'][g]['mean_ce'])
                if len(ranked) < settings['elites']:
                    raise RuntimeError('Insufficient valid parents')
                population = ranked[:settings['elites']]
                for _ in range(10000):
                    parents = [rng.choice(ranked[:settings['parents']]).split('-') for _ in range(2)]
                    parts = [rng.choice(pair) for pair in zip(*parents)]
                    if rng.random() < settings['mutation_probability']:
                        j = rng.randrange(3)
                        parts[j] = sample().split('-')[j]
                    graph = seq.canonical('-'.join(parts))
                    if graphlib.feasible(graph) and graph not in history['candidates'] and graph not in population:
                        population.append(graph)
                    if len(population) == settings['population']:
                        break
                else:
                    raise RuntimeError('Cannot generate distinct offspring')
            history['generations'].append(dict(population=population, rng_after_population=rng.getstate()))
            write(path, history)
        for graph in population:
            if graph in history['candidates']:
                continue
            restore()
            try:
                engine = Engine.from_string(graph)
                values = []
                for score in dense_scores(mapping, refs, stats):
                    value = engine.compute_importance(score.flatten())
                    if value == -1 or not np.isfinite(value):
                        raise ValueError('Invalid public graph')
                    values.append(value)
                rates = weighted_rates([-value for value in values], [r['weights'] for r in refs], lam=settings['lam'])
                counts, budget = exact_projection_rows(refs, rates, TARGET)
            except (ValueError, RuntimeError, TypeError, IndexError) as exc:
                history['candidates'][graph] = dict(status='invalid', error=str(exc))
                write(path, history)
                continue
            seq.sequential(model, mapping, refs, states, counts, 'dsa_projection')
            scores = seq.masked_ce(model, read(seq.DEV)['states'], 'dsa_projection', 'development', graph=graph)
            history['candidates'][graph] = dict(status='valid', mean_ce=scores['mean_ce'],
                state_ce=scores['state_ce'], counts=counts, rates=rates.tolist(), budget=budget)
            write(path, history)
        event('dsa_projection', 'search_generation', completed=generation+1, total=settings['generations'])
    winner = min((g for g, row in history['candidates'].items() if row['status'] == 'valid'),
                 key=lambda g: history['candidates'][g]['mean_ce'])
    owl.frozen(ROOT / 'dsa_projection' / 'selection.json', dict(graph=winner, **history['candidates'][winner]))
    restore()
    return history['candidates'][winner]['counts']


def freeze():
    path = ROOT / 'config.json'
    if path.exists():
        cfg = read(path)
        for name, digest in cfg['sources'].items():
            if sha(name) != digest:
                raise RuntimeError('Frozen source changed: ' + name)
        return cfg
    base = prior.validate()
    files = [Path(__file__), seq.CAL, seq.DEV, seq.VER, seq.old.STATS,
             seq.old.SOURCE / 'candidate_mask_manifest.json', prior.ROOT / 'config.json',
             protocol.ROOT / 'config.json', protocol.ROOT / 'corpus_manifest.json',
             Path(seq.__file__), Path(prior.__file__), Path(graphlib.__file__),
             Path(owl.__file__), ROOT / 'run.sh', ROOT / 'status.py']
    cfg = dict(model=base['model'], target=.5, pruned=TARGET, weights=TARGET*2,
               methods=list(METHODS), calibration=base['calibration'], evaluation=base['evaluation'],
               dsa=base['dsa'], development=base['dsa']['fitness'],
               owl=dict(M=5, lam=.08),
               adaptation='224 projection-local statistics and parameter-weighted minmax centering; independent row count per projection; exact50 DP',
               test_policy='Validation only; same frozen 551 chunks/MC128, no test or GSM8K',
               sources={str(p):sha(p) for p in files})
    owl.frozen(path, cfg)
    return cfg


@torch.inference_mode()
def run(method):
    folder = ROOT / method
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = freeze()
        try:
            if (folder / 'validation/results.json').exists():
                event(method, 'complete', summary=read(folder / 'validation/results.json')['summary'])
                return
            from experiments.projection_capacity_allocation_65.run import load_dense
            from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
            event(method, 'loading_dense')
            model, mapping = load_dense()
            refs = read(seq.old.SOURCE / 'candidate_mask_manifest.json')['entries']
            if list(mapping) != [r['name'] for r in refs]:
                raise RuntimeError('Projection order mismatch')
            seq.ROOT = ROOT
            seq.STORE = Path('/DATA/tmluser1/dlm-ppl50-projection')
            seq.event = event
            mp = folder / 'mask_manifest.json'
            if mp.exists():
                manifest = read(mp)
                if manifest['config_sha256'] != sha(ROOT / 'config.json'):
                    raise RuntimeError('Manifest config mismatch')
                apply_manifest(model, mapping, manifest)
            else:
                states = read(seq.CAL)['states']
                stats = torch.load(seq.old.STATS, map_location='cpu', weights_only=False)['statistics']
                if method == 'owl_projection':
                    counts, allocation = owl_allocation(mapping, refs, stats)
                    owl.frozen(folder / 'allocation.json', dict(row_counts=counts, **allocation))
                else:
                    counts = dsa_search(model, mapping, refs, states, cfg, stats)
                    owl.frozen(folder / 'allocation.json', dict(row_counts=counts,
                              selection_sha256=sha(folder / 'selection.json')))
                rows = seq.sequential(model, mapping, refs, states, counts, method, folder)
                manifest = dict(method=method+'_sequential_wanda50', config_sha256=sha(ROOT/'config.json'),
                                entries=rows, pruned=TARGET, weights=TARGET*2)
                owl.frozen(mp, manifest)
                prior.verify_masks(manifest)
            prior.ROOT = ROOT
            prior.validate = freeze
            result = prior.score(model, method, cfg, mp)
            event(method, 'complete', summary=result['summary'])
        except BaseException as exc:
            event(method, 'failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'run'))
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    freeze() if args.phase == 'freeze' else run(args.method)
