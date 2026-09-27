"""Native sequential-Wanda50 and bounded DSA; historical block0 rank control remains65."""
import argparse
import fcntl
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.dlm_owl65 import run as old
from experiments.dlm_dsa_search65 import run as graphlib
from experiments.dlm_owl65.core import exact_row_counts
from experiments.dlm_allocation_baselines65.core import dsa_rates
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write, file_sha256 as sha
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CAL = REPO / graphlib.CAL
DEV = REPO / 'experiments/projection_capacity_followup_65/new_heldout_state_manifest.json'
HELD = REPO / graphlib.HELD
VER = REPO / 'experiments/projection_capacity_followup_65/new_state_verification.json'
METHODS = ('uniform', 'owl', 'dlp', 'alpha', 'lsa', 'dsa')
STORE = Path('/DATA/tmluser1/dlm-ppl50')

def read(path):
    return json.loads(Path(path).read_text())

def event(method, stage, **kw):
    value = dict(method=method, stage=stage, time=time.time(), pid=os.getpid(), **kw)
    write(ROOT / method / 'progress.json', value)
    print(json.dumps(value), flush=True)

class PrefixDone(Exception):
    pass

@torch.inference_mode()
def collect_block(model, mapping, refs, states, block_index, progress=None):
    """Run native model prefix, stopping after this block, without AR masks/kwargs."""
    names = [r['name'] for r in refs[block_index * 7:(block_index + 1) * 7]]
    observations = {n: [] for n in names}
    calls = dict.fromkeys(names, 0)
    handles = []
    for name in names:

        def hook(mod, inp, out, key=name):
            x = inp[0]
            if x.ndim != 3 or x.shape[0] != 1:
                raise RuntimeError('Frozen batch-one calibration path required')
            observations[key].append(x.reshape(-1, x.shape[-1]).float().square().sum(0).cpu())
            calls[key] += 1
        handles.append(mapping[name].register_forward_hook(hook))

    def stop(mod, inp, out):
        raise PrefixDone()
    handles.append(model.model.transformer.blocks[block_index].register_forward_hook(stop))
    device = next(model.parameters()).device
    try:
        for i, state in enumerate(states):
            try:
                model(torch.tensor(state['noisy_ids'], device=device))
            except PrefixDone:
                pass
            else:
                raise RuntimeError('Native block stop hook was not reached')
            if progress:
                progress(i + 1)
    finally:
        for handle in handles:
            handle.remove()
    if any((v != len(states) for v in calls.values())):
        raise RuntimeError('Missing or duplicate projection observations')
    return {n: torch.stack(v).mean(0).to(mapping[n].weight.device) for n, v in observations.items()}

def wanda_mask(weight, activation, count):
    if activation.shape != (weight.shape[1],) or not torch.isfinite(activation).all() or (activation < 0).any():
        raise RuntimeError('Invalid Wanda activation')
    if not 0 <= count <= weight.shape[1]:
        raise RuntimeError('Invalid row budget')
    score = weight.float().abs() * activation.sqrt()[None, :]
    order = torch.argsort(score, dim=1, stable=True)
    return torch.zeros_like(weight, dtype=torch.bool).scatter_(1, order[:, :count], True)

@torch.inference_mode()
def sequential(model, mapping, refs, states, counts, method, folder=None):
    from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
    from experiments.projection_capacity_allocation_65.run import save_tensor
    from experiments.projection_capacity_followup_65.run_heldout import selected_mask
    previous = torch.load(old.STATS, map_location='cpu', weights_only=False)['statistics']
    entries = []
    for block in range(32):
        checkpoint = folder / f'block_{block:02d}.json' if folder else None
        if checkpoint and checkpoint.exists():
            rows = read(checkpoint)
            for ref, row, k in zip(refs[block * 7:block * 7 + 7], rows, counts[block * 7:block * 7 + 7]):
                if row['name'] != ref['name'] or row['selected_mask']['prune_per_row'] != k:
                    raise RuntimeError('Resume allocation mismatch')
                mask = selected_mask(row, mapping[row['name']].weight.device)
                if not (mask.sum(1) == k).all():
                    raise RuntimeError('Resume mask budget mismatch')
                mapping[row['name']].weight.masked_fill_(mask, 0)
            entries.extend(rows)
            continue
        event(method, 'calibration', block=block, completed=0, total=len(states))
        activation = collect_block(model, mapping, refs, states, block, lambda i: event(method, 'calibration', block=block, completed=i, total=len(states)))
        rows = []
        for i in range(block * 7, block * 7 + 7):
            ref = refs[i]
            name = ref['name']
            weight = mapping[name].weight
            a = activation[name]
            dense_a = previous[name]['overall_uniform'].float().to(a.device)
            error = float((a - dense_a).norm() / dense_a.norm())
            if block == 0 and error > 0.0001:
                raise RuntimeError(f'First-block dense calibration control failed: {name}: {error}')
            mask = wanda_mask(weight, a, counts[i])
            packed = pack_mask(mask.cpu())
            if block == 0:
                control = wanda_mask(weight, a, int(weight.shape[1] * 0.65))
                if mask_sha256(pack_mask(control.cpu())) != ref['masks'][3]['mask_sha256']:
                    raise RuntimeError('First-block historical Wanda ranking mismatch')
            row = dict(name=name, shape=ref['shape'], weights=ref['weights'], module_index=i, assigned_sparsity=counts[i] / ref['shape'][1], dense_activation_relative_change=error)
            if folder:
                path = STORE / method / 'masks' / f'{name}.pt'
                if path.exists():
                    if mask_sha256(torch.load(path, map_location='cpu', weights_only=False)) != mask_sha256(packed):
                        raise RuntimeError('Partial mask differs; preserve and investigate')
                else:
                    save_tensor(path, packed)
                row['selected_mask'] = dict(path=str(path), file_sha256=sha(path), mask_sha256=mask_sha256(packed), prune_per_row=counts[i], pruned=counts[i] * ref['shape'][0])
            weight.masked_fill_(mask, 0)
            rows.append(row)
        if checkpoint:
            write(checkpoint, rows)
        entries.extend(rows)
        event(method, 'pruned_block', completed=block + 1, total=32)
    if sum((k * r['shape'][0] for k, r in zip(counts, refs))) != 3489660928:
        raise RuntimeError('Global discrete budget mismatch')
    return entries

def canonical(graph):
    return '-'.join(graph.split('-')[:3] + ['(7)'])

def tuples(value):
    return tuple((tuples(v) for v in value)) if isinstance(value, list) else value

@torch.inference_mode()
def search(model, mapping, refs, states, cfg):
    settings = cfg['dsa']
    rng = random.Random(settings['seed'])
    Engine = graphlib.engine_class()
    history_path = ROOT / 'dsa' / 'search.json'
    history = read(history_path) if history_path.exists() else dict(candidates={}, generations=[])
    dense = {n: m.weight.detach().cpu().clone() for n, m in mapping.items()}
    stats = torch.load(old.STATS, map_location='cpu', weights_only=False)['statistics']

    def restore():
        for n, m in mapping.items():
            m.weight.copy_(dense[n])

    def sample():
        return canonical(graphlib.random_graph(rng, Engine))
    for generation in range(settings['generations']):
        if generation < len(history['generations']):
            record = history['generations'][generation]
            rng.setstate(tuples(record['rng_after_population']))
            population = record['population']
        else:
            if generation == 0:
                population = [canonical(graphlib.FIXED)]
                while len(population) < settings['population']:
                    graph = sample()
                    if graph not in population:
                        population.append(graph)
            else:
                last = history['generations'][-1]['population']
                ranked = sorted((g for g in last if history['candidates'][g]['status'] == 'valid'), key=lambda g: history['candidates'][g]['mean_ce'])
                if len(ranked) < settings['elites']:
                    raise RuntimeError('Insufficient valid parents')
                population = ranked[:settings['elites']]
                for _ in range(10000):
                    parents = [rng.choice(ranked[:settings['parents']]).split('-') for j in range(2)]
                    parts = [rng.choice(pair) for pair in zip(*parents)]
                    if rng.random() < settings['mutation_probability']:
                        j = rng.randrange(3)
                        parts[j] = sample().split('-')[j]
                    graph = canonical('-'.join(parts))
                    if graphlib.feasible(graph) and graph not in history['candidates'] and (graph not in population):
                        population.append(graph)
                    if len(population) == settings['population']:
                        break
                else:
                    raise RuntimeError('Cannot generate distinct offspring')
            history['generations'].append(dict(population=population, rng_after_population=rng.getstate()))
            write(history_path, history)
        for graph in population:
            if graph in history['candidates']:
                continue
            restore()
            values = []
            try:
                for block in range(32):
                    pooled = torch.cat([(mapping[r['name']].weight.float().abs() * stats[r['name']]['overall_uniform'].float().to(mapping[r['name']].weight.device).sqrt()[None, :]).flatten() for r in refs[block * 7:block * 7 + 7]])
                    value = Engine.from_string(graph).compute_importance(pooled)
                    if value == -1 or not np.isfinite(value):
                        raise ValueError('Invalid public graph')
                    values.append(value)
                if settings['lam'] != 0.08:
                    raise RuntimeError('Public mapping implementation is frozen at lambda=.08')
                rates = dsa_rates(values, target=0.5)
                counts, budget = exact_row_counts(refs, rates, cfg['pruned'])
            except (ValueError, RuntimeError, TypeError, IndexError) as exc:
                history['candidates'][graph] = dict(status='invalid', error=str(exc))
                write(history_path, history)
                continue
            sequential(model, mapping, refs, states, counts, 'dsa')
            scores = masked_ce(model, read(DEV)['states'], 'dsa', 'development', graph=graph)
            history['candidates'][graph] = dict(status='valid', mean_ce=scores['mean_ce'], state_ce=scores['state_ce'], counts=counts, rates=rates.tolist(), budget=budget)
            write(history_path, history)
        event('dsa', 'search_generation', completed=generation + 1, total=settings['generations'])
    winner = min((g for g, r in history['candidates'].items() if r['status'] == 'valid'), key=lambda g: history['candidates'][g]['mean_ce'])
    old.frozen(ROOT / 'dsa' / 'selection.json', dict(graph=winner, **history['candidates'][winner]))
    restore()
    return history['candidates'][winner]['counts']

@torch.inference_mode()
def masked_ce(model, states, method, stage, **kw):
    dev = next(model.parameters()).device
    values = []
    for i, s in enumerate(states):
        mask = torch.tensor(s['mask'], device=dev, dtype=torch.bool)
        if not mask.any():
            raise RuntimeError('Empty masked state')
        logits = model(torch.tensor(s['noisy_ids'], device=dev)).logits
        values.append(float(F.cross_entropy(logits[mask].float(), torch.tensor(s['clean_ids'], device=dev)[mask])))
        event(method, stage, completed=i + 1, total=len(states), **kw)
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite masked CE')
    return dict(mean_ce=float(np.mean(values)), state_ce=values)
