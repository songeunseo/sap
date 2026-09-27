"""Prepare isolated, hash-bound inputs without model forwards."""
from pathlib import Path
import copy
import numpy as np
from experiments.dlm_multiscale_ac50.artifacts import DEFAULT_ROOT as OLD, read, sha, freeze, checked
from experiments.dlm_multiscale_ac50.core import metrics, marginal_cost
from experiments.dlm_multiscale_ac50.prepare import validate as validate_base
from experiments.dlm_owl65.core import exact_row_counts
from .core import ARMS, allocation_rates, SEED, DRAWS

HERE = Path(__file__).resolve().parent
ROOT = HERE/'output'

def source_costs():
    c = validate_base(OLD)
    config_hash = sha(OLD/'config.json')
    bank = read(c['banks']['calibration']['path'])
    teacher_path = OLD/'readouts/dense/calibration.json'
    teacher = read(teacher_path)['values']
    bound = {str(OLD/'config.json'):config_hash, str(teacher_path):sha(teacher_path)}
    costs, slopes = [], []
    for b in range(32):
        p = OLD/'probes'/f'block{b:02d}.json'; row = read(p)
        if row['config_sha256'] != config_hash or row['block'] != b:
            raise ValueError('Wrong source probe identity')
        bound[str(p)] = sha(p)
        lo, hi = [row['conditions'][str(s)] for s in (.48, .52)]
        results = []
        for cond in (lo, hi):
            checked(cond['readout_path'], cond['readout_sha256'])
            bound[cond['readout_path']] = cond['readout_sha256']
            result = metrics(read(cond['readout_path'])['values'], teacher, bank)
            if result != cond['metrics']:
                raise ValueError('Raw readouts and source metrics differ')
            results.append(result)
        low, high = results
        if marginal_cost(low, high, lo['pruned'], hi['pruned']) != row['costs']:
            raise ValueError('Source marginal differs')
        if [s['sequence_index'] for s in low['sequences']] != list(range(8)) or [s['sequence_index'] for s in high['sequences']] != list(range(8)):
            raise ValueError('Expected aligned eight calibration spans')
        costs.append([(h['metrics']['Multi']-l['metrics']['Multi'])/(hi['pruned']-lo['pruned']) for l,h in zip(low['sequences'],high['sequences'],strict=True)])
        slopes.append(row['costs']['Multi'])
    return c, np.asarray(costs), slopes, bound

def derive(costs, refs, target):
    rates, draws = allocation_rates(costs)
    result = {}
    for name in ARMS:
        counts, budget = exact_row_counts(refs, rates[name], target)
        result[name] = dict(rates=rates[name].tolist(), row_counts=counts, budget=budget,
                            achieved_projection_rates=[k/r['shape'][1] for k,r in zip(counts,refs)])
    return result, draws.tolist()

def prepare():
    if (ROOT/'started.json').exists():
        raise RuntimeError('Started experiment is immutable; use validate')
    base, costs, slopes, bound = source_costs()
    refs = read(base['legacy_manifests']['uniform']['path'])['entries']
    old_allocation = read(OLD/'allocation.json')
    from experiments.dlm_multiscale_ac50.core import rank_rates
    rr = rank_rates(slopes)
    counts, budget = exact_row_counts(refs, rr, base['pruning']['pruned'])
    old = old_allocation['allocations']['Multi']
    if old['scores'] != slopes or old['rates'] != rr.tolist() or old['row_counts'] != counts or old['budget'] != budget:
        raise ValueError('Source Multi allocation cannot be reproduced')
    if not np.allclose(costs.mean(axis=1), slopes, atol=1e-22, rtol=1e-12):
        raise ValueError('Span averaging differs from source objective')
    allocations, draws = derive(costs, refs, base['pruning']['pruned'])
    config = copy.deepcopy(base)
    bound[str(OLD/'allocation.json')] = sha(OLD/'allocation.json')
    for split in ('calibration','diagnostic'):
        path = OLD/'readouts/dense'/f'{split}.json'; bound[str(path)] = sha(path)
    for path in sorted(HERE.glob('*.py')):
        if not path.name.startswith('test_'): bound[str(path)] = sha(path)
    for path in (HERE/'PLAN.md',HERE/'run.sh'):
        bound[str(path)] = sha(path)
    # Bind imported controller dependency and its scientific imports transitively.
    screen = Path(__file__).resolve().parents[1]/'dlm_ac_screen50'
    for path in sorted(screen.glob('*.py')):
        if not path.name.startswith('test_'): bound[str(path)] = sha(path)
    for path in (OLD/'gsm8k/development/Multi').rglob('*.json'):
        bound[str(path)] = sha(path)
    for path in (OLD/'candidates/Multi/model_identity.json',OLD/'candidates/Multi/mask_manifest.json'):
        bound[str(path)] = sha(path)
    # Freeze the physical reference masks, not only their declared identities.
    import torch
    from experiments.dlm_loss_aggregation.core import mask_sha256
    for entry in read(OLD/'candidates/Multi/mask_manifest.json')['entries']:
        meta = entry['selected_mask']; checked(meta['path'], meta['file_sha256'])
        if mask_sha256(torch.load(meta['path'], map_location='cpu', weights_only=False)) != meta['mask_sha256']:
            raise ValueError('Reference packed mask identity differs')
        bound[meta['path']] = meta['file_sha256']
    for split in ('calibration','diagnostic'):
        rp = OLD/'readouts/Multi'/f'{split}.json'
        dp = OLD/'diagnostics/Multi'/f'{split}.json'
        tp = OLD/'readouts/dense'/f'{split}.json'
        result = metrics(read(rp)['values'], read(tp)['values'], read(base['banks'][split]['path']))
        diagnostic = read(dp)
        if any(diagnostic[k] != v for k,v in result.items()) or diagnostic['readout_sha256'] != sha(rp) or diagnostic['teacher_sha256'] != sha(tp):
            raise ValueError('Reference diagnostic cannot be reproduced')
        for p in (rp,dp): bound[str(p)] = sha(p)
    config['sources'].update(bound)
    config['followup'] = dict(arms=list(ARMS), source_root=str(OLD), seed=SEED, draws=DRAWS,
                             radii_pp=[2,3.5], reference='Multi', plan_sha256=sha(HERE/'PLAN.md'))
    # Restrict evaluator inputs to development. No confirmation requests available.
    requests = read(OLD/'requests.json'); requests['confirmation'] = []
    if [r['example_id'] for r in requests['development']] != list(range(100)):
        raise ValueError('Unexpected development sample')
    freeze(ROOT/'requests.json', requests)
    freeze(ROOT/'cached_development.json', read(OLD/'cached_development.json'))
    config['requests_sha256'] = sha(ROOT/'requests.json')
    freeze(ROOT/'config.json', config)
    ch = sha(ROOT/'config.json')
    freeze(ROOT/'allocation.json', dict(config_sha256=ch, allocations=allocations,
          span_costs=costs.tolist(), bootstrap_draws=draws, source_probe_hashes=bound,
          reference_reproduced=True))
    for split in ('calibration','diagnostic'):
        old_teacher = read(OLD/'readouts/dense'/f'{split}.json')
        teacher = copy.deepcopy(old_teacher)
        if teacher['fingerprint']['config_sha256'] != sha(OLD/'config.json') or teacher['fingerprint']['bank_sha256'] != base['banks'][split]['sha256'] or not teacher['complete']:
            raise ValueError('Source teacher identity incomplete')
        teacher['fingerprint']['config_sha256'] = ch
        freeze(ROOT/'readouts/dense'/f'{split}.json', teacher)
    validate()
    return allocations

def validate():
    c = validate_base(ROOT)
    if c['followup']['arms'] != list(ARMS) or c['followup']['seed'] != SEED or c['followup']['draws'] != DRAWS:
        raise ValueError('Follow-up contract differs')
    row = read(ROOT/'allocation.json')
    if row['config_sha256'] != sha(ROOT/'config.json'):
        raise ValueError('Allocation belongs to another config')
    refs = read(c['legacy_manifests']['uniform']['path'])['entries']
    allocations, draws = derive(row['span_costs'], refs, c['pruning']['pruned'])
    # Bind the stored costs to the original raw-readout-verified probes again.
    _, costs, _, _ = source_costs()
    if row['span_costs'] != costs.tolist() or row['allocations'] != allocations or row['bootstrap_draws'] != draws:
        raise ValueError('Allocation derivation mismatch')
    for split in ('calibration','diagnostic'):
        source = read(OLD/'readouts/dense'/f'{split}.json')
        source['fingerprint']['config_sha256'] = sha(ROOT/'config.json')
        if read(ROOT/'readouts/dense'/f'{split}.json') != source:
            raise ValueError('Reused teacher changed')
    if read(ROOT/'requests.json')['confirmation']:
        raise ValueError('Confirmation is out of scope')
    return c
