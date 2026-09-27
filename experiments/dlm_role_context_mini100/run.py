"""One-shot Role-Wanda recalibration candidates, frozen historical mini-100."""
import argparse
import fcntl
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.dlm_role_context_mini100.core import role_sums, pooled_curves
from experiments.dlm_dual_role_allocation.io import load_frozen_inputs, atomic_write_json as write, file_sha256 as sha
from experiments.dlm_dual_role_allocation.core import allocation_mask_xor
from experiments.dlm_dual_role_mini100.run import TARGET, WEIGHTS, _validate_rows
from experiments.dlm_dual_role_mini100.core import same_selected_masks, validate_manifest
from experiments.dlm_loss_aggregation.exp002.run import (
    _evaluate_gsm8k, _evaluation_config_hash, _read_jsonl, _write_jsonl, load_config as eval_config)
from experiments.projection_capacity_allocation_65.core import GRID, allocate
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, load_dense, read_mask
from experiments.projection_capacity_followup_65.core import build_selected_manifest, paired_binary_comparison
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
from experiments.wanda_failure_characterization.run_failure_map import model_sha

ROOT = Path('experiments/dlm_role_context_mini100')
OLD = Path('experiments/dlm_dual_role_mini100')
ROLE = OLD / 'role65_mask_manifest.json'
RAW = Path('experiments/dlm_dual_role_allocation/role_reconstruction_raw.json')
EVAL = Path('experiments/dlm_loss_aggregation/exp002/config.yaml')
METHODS = ('sparse_context', 'dense_target')


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError('Frozen artifact mismatch: ' + str(path))
    else:
        write(path, value)


def event(method, stage, **kw):
    row = dict(method=method, stage=stage, time=time.time(), **kw)
    write(ROOT / method / 'progress.json', row)
    print(json.dumps(row), flush=True)


def freeze():
    inputs = load_frozen_inputs()
    cfg = eval_config(EVAL)
    protocol_hash, protocol = _evaluation_config_hash(cfg)
    reference = _read_jsonl(Path('experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'))
    baseline = _read_jsonl(OLD / 'gsm8k/role_100_predictions.jsonl')
    receipt = json.loads((OLD / 'gsm8k/role_100_predictions.receipt.json').read_text())
    _validate_rows(baseline, reference, protocol_hash)
    if receipt['fingerprint']['manifest_sha256'] != sha(ROLE) or receipt['predictions_sha256'] != sha(OLD / 'gsm8k/role_100_predictions.jsonl'):
        raise RuntimeError('Baseline receipt invalid')
    if receipt['fingerprint']['protocol_sha256'] != protocol_hash or sum(r['correct'] for r in baseline) != 24:
        raise RuntimeError('Historical Role reference mismatch')
    sources = dict(inputs.receipt['files'])
    for p in [RAW, ROLE, EVAL, OLD / 'gsm8k/role_100_predictions.jsonl',
              OLD / 'gsm8k/role_100_predictions.receipt.json', OLD / 'config.json',
              Path('experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'),
              Path('experiments/projection_capacity_allocation_65/core.py'),
              Path('experiments/dlm_loss_aggregation/exp002/run.py'),
              Path('experiments/projection_capacity_followup_65/run_heldout.py'),
              *ROOT.glob('*.py'), ROOT / 'run.sh']:
        sources[str(p)] = sha(p)
    config = dict(model=inputs.config['model'], state_digest=inputs.metadata['state_digest'], states=80,
                  source_receipt=inputs.receipt['receipt_sha256'], methods=list(METHODS), grid=list(GRID),
                  pruned=TARGET, weights=WEIGHTS, initial_model='frozen historical Role-Wanda-65',
                  definitions={'sparse_context':'||(W_s-W)x_sparse||^2 / fixed historical dense role output energy',
                               'dense_target':'||W_s x_sparse - W x_dense||^2 / same fixed historical denominator'},
                  arithmetic='separate BF16 batch1 Linear outputs then FP32 squared differences, FP64 pooled sums',
                  updates=1, aggregation='max of pooled role levels, raw adjacent marginal greedy; exact row-floor budget',
                  dense_control_gate='batch1 dense reconstruction must reproduce historical Role allocation before mini',
                  evaluation=dict(limit=100, protocol_hash=protocol_hash, protocol=protocol, baseline_correct=24),
                  no_smoothing_no_retuning=True, source_hashes=sources)
    frozen(ROOT / 'config.json', config)
    return config


def validate():
    c = json.loads((ROOT / 'config.json').read_text())
    for path, digest in c['source_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Changed frozen input: ' + path)
    return c


@torch.inference_mode()
def collect(method):
    config = validate(); dest = ROOT / method
    if (dest / 'collection.json').exists():
        receipt = json.loads((dest / 'collection.json').read_text())
        if receipt['config_sha256'] != sha(ROOT / 'config.json') or receipt['curves_sha256'] != sha(dest / 'curves.json'):
            raise RuntimeError('Invalid completed curve receipt')
        return
    inputs = load_frozen_inputs()
    raw = json.loads(RAW.read_text())['projections']
    names = inputs.metadata['module_names']
    if [r['name'] for r in raw] != names:
        raise RuntimeError('Role denominator order mismatch')
    denominators = np.array([[sum(float(s['levels'][0]['den_' + role]) for s in r['states'])
                              for role in ('masked', 'unmasked')] for r in raw])
    role_manifest = json.loads(ROLE.read_text())
    dense, dm = load_dense(); sparse, sm = load_dense()
    if list(dm) != names or list(sm) != names:
        raise RuntimeError('Module order changed')
    if apply_manifest(sparse, sm, role_manifest) != TARGET:
        raise RuntimeError('Role baseline budget mismatch')
    sparse_hash = model_sha(sparse)
    old_receipt = json.loads((OLD / 'gsm8k/role_100_predictions.receipt.json').read_text())
    if sparse_hash != old_receipt['sparse_model_sha256']:
        raise RuntimeError('Role model hash mismatch')
    device = next(dense.parameters()).device
    event(method, 'loading_masks', completed=0, total=224)
    # ~39GiB masks plus two BF16 models fits each 96GiB GPU; no six-model weight cache.
    masks = {}
    for i, entry in enumerate(inputs.candidate['entries']):
        masks[entry['name']] = [read_mask(entry, k, device) for k in range(6)]
        if (i+1) % 32 == 0:
            event(method, 'loading_masks', completed=i+1, total=224)
    state = {}
    handles = []
    for i, name in enumerate(names):
        def dense_hook(module, inp, out, index=i, name=name):
            x = inp[0]
            if x.shape[0] != 1:
                raise RuntimeError('Batch shape drift')
            ref = out.detach()
            if not torch.equal(F.linear(x, module.weight, module.bias), ref):
                raise RuntimeError('Dense module output does not match local Linear')
            state['teacher'][name] = ref.clone()
            for k, mask in enumerate(masks[name]):
                candidate = F.linear(x, module.weight.masked_fill(mask, 0), module.bias)
                state['control'][index, k] = role_sums(candidate, ref, state['mask'])
        def sparse_hook(module, inp, out, index=i, name=name):
            x = inp[0]; teacher = state['teacher'][name]
            dmodule = dm[name]
            ref = F.linear(x, dmodule.weight, dmodule.bias) if method == 'sparse_context' else teacher
            if not torch.equal(F.linear(x, module.weight, module.bias), out):
                raise RuntimeError('Sparse module output does not match local Linear')
            for k, mask in enumerate(masks[name]):
                candidate = F.linear(x, dmodule.weight.masked_fill(mask, 0), dmodule.bias)
                state['numbers'][index, k] = role_sums(candidate, ref, state['mask'])
        handles.append(dm[name].register_forward_hook(dense_hook))
        handles.append(sm[name].register_forward_hook(sparse_hook))
    records, controls = [], []
    started = time.monotonic(); completed_new = 0
    try:
        for i, s in enumerate(inputs.states['states']):
            path = dest / 'states' / f'{i:03d}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['config_sha256'] != sha(ROOT / 'config.json') or saved['state_index'] != i:
                    raise RuntimeError('State checkpoint mismatch')
            else:
                state.clear()
                state.update(teacher={}, mask=torch.tensor(s['mask'], dtype=torch.bool, device=device),
                             numbers=np.zeros((224, 6, 2)), control=np.zeros((224, 6, 2)))
                ids = torch.tensor(s['noisy_ids'], dtype=torch.long, device=device)
                dense(ids); sparse(ids)
                if len(state['teacher']) != 224:
                    raise RuntimeError('Missing projection hook')
                saved = dict(config_sha256=sha(ROOT / 'config.json'), state_index=i,
                             numbers=state['numbers'].tolist(), control=state['control'].tolist())
                write(path, saved); completed_new += 1
                state.clear()
            records.append(saved['numbers']); controls.append(saved['control'])
            elapsed = time.monotonic() - started
            event(method, 'collecting', completed=i+1, total=80,
                  eta_seconds=(80-i-1)*elapsed/completed_new if completed_new else None)
    finally:
        for handle in handles:
            handle.remove()
    if model_sha(dense) != DENSE_SHA or model_sha(sparse) != sparse_hash:
        raise RuntimeError('Collection mutated weights')
    values, role = pooled_curves(records, denominators)
    control_values, control = pooled_curves(controls, denominators)
    shapes = [e['shape'] for e in inputs.candidate['entries']]
    control_allocation = allocate(control, shapes)
    historical_levels = role_manifest['allocation_levels']
    historical_curve = np.array([[sum(float(s['levels'][k]['num_' + r]) for s in row['states']) / denominators[i, ri]
                                   for k in range(6)] for i, row in enumerate(raw) for ri, r in enumerate(('masked', 'unmasked'))]).reshape(224, 2, 6).transpose(0, 2, 1)
    control_result = dict(passed=control_allocation['levels'] == historical_levels,
                          changed_projections=sum(a != b for a,b in zip(control_allocation['levels'], historical_levels)),
                          max_abs_curve_drift=float(np.max(abs(control_values-historical_curve))),
                          allocation=control_allocation)
    write(dest / 'dense_control.json', control_result)
    write(dest / 'curves.json', dict(config_sha256=sha(ROOT / 'config.json'), names=names, grid=list(GRID),
                                  masked=values[:,:,0].tolist(), unmasked=values[:,:,1].tolist(), role=role.tolist(),
                                  fixed_denominators=denominators.tolist()))
    if not control_result['passed']:
        raise RuntimeError('Dense batch1 control changed historical allocation; stop before downstream')
    write(dest / 'collection.json', dict(status='complete', config_sha256=sha(ROOT / 'config.json'),
                                        curves_sha256=sha(dest / 'curves.json'), sparse_hash=sparse_hash,
                                        dense_control_sha256=sha(dest / 'dense_control.json')))
    del masks, dm, sm, dense, sparse
    gc.collect(); torch.cuda.empty_cache()


def prepare(method):
    config = validate(); inputs = load_frozen_inputs(); dest = ROOT / method
    curves = json.loads((dest / 'curves.json').read_text())
    receipt = json.loads((dest / 'collection.json').read_text())
    if receipt['curves_sha256'] != sha(dest / 'curves.json'):
        raise RuntimeError('Curve receipt mismatch')
    allocation = allocate(curves['role'], [e['shape'] for e in inputs.candidate['entries']])
    if allocation['pruned'] != TARGET or allocation['budget_error']:
        raise RuntimeError('Exact target infeasible')
    manifest = build_selected_manifest(method, inputs.candidate['entries'], allocation['sparsities'], GRID)
    manifest.update(config_sha256=sha(ROOT / 'config.json'), allocation_levels=allocation['levels'])
    validate_manifest(manifest, inputs.metadata['module_names'], TARGET, WEIGHTS)
    frozen(dest / 'mask_manifest.json', manifest)
    historical = json.loads(ROLE.read_text())
    delta = allocation_mask_xor(allocation['levels'], historical['allocation_levels'], inputs.candidate['entries'])
    frozen(dest / 'allocation.json', dict(allocation=allocation, difference_from_role=delta,
                                         manifest_sha256=sha(dest / 'mask_manifest.json')))
    event(method, 'allocation_complete', **delta)


@torch.inference_mode()
def evaluate(method):
    c = validate(); dest = ROOT / method
    if (dest / 'results.json').exists():
        r = json.loads((dest / 'results.json').read_text())
        if r['config_sha256'] != sha(ROOT / 'config.json') or r['predictions_sha256'] != sha(dest / 'predictions.jsonl'):
            raise RuntimeError('Completed result receipt mismatch')
        event(method, 'complete', correct=r['correct'], total=100); return
    manifest = json.loads((dest / 'mask_manifest.json').read_text())
    allocation = json.loads((dest / 'allocation.json').read_text())
    if allocation['manifest_sha256'] != sha(dest / 'mask_manifest.json'):
        raise RuntimeError('Manifest changed')
    cfg = eval_config(EVAL); ph, _ = _evaluation_config_hash(cfg)
    reference = _read_jsonl(Path('experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'))
    baseline = _read_jsonl(OLD / 'gsm8k/role_100_predictions.jsonl')
    _validate_rows(baseline, reference, ph)
    reused = same_selected_masks(manifest, json.loads(ROLE.read_text()))
    if reused:
        rows = baseline; metrics = dict(reused='identical historical Role masks'); sparse_hash = None
    else:
        model, mapping = load_dense()
        if apply_manifest(model, mapping, manifest) != TARGET:
            raise RuntimeError('Applied budget mismatch')
        sparse_hash = model_sha(model)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model']['id'], revision=cfg['model']['revision'], trust_remote_code=True)
        event(method, 'gsm8k', completed=0, total=100)
        metrics, rows = _evaluate_gsm8k(model, tokenizer, cfg, method, 100, ph)
        if model_sha(model) != sparse_hash:
            raise RuntimeError('Evaluation mutated model')
    _validate_rows(rows, reference, ph)
    _write_jsonl(dest / 'predictions.jsonl', rows)
    result = dict(status='complete', method=method, config_sha256=sha(ROOT / 'config.json'),
                  manifest_sha256=sha(dest / 'mask_manifest.json'), protocol_hash=ph,
                  predictions_sha256=sha(dest / 'predictions.jsonl'), sparse_model_sha256=sparse_hash,
                  correct=sum(r['correct'] for r in rows), total=100, baseline_correct=24,
                  paired_vs_role=paired_binary_comparison([r['correct'] for r in baseline], [r['correct'] for r in rows]),
                  metrics=metrics, reused_identical_masks=reused,
                  interpretation='fixed mini100 development screen; not full or independent confirmation')
    write(dest / 'results.json', result)
    event(method, 'complete', correct=result['correct'], total=100)


def main():
    p = argparse.ArgumentParser(); p.add_argument('method', choices=(*METHODS, 'freeze'))
    args = p.parse_args()
    if args.method == 'freeze':
        freeze(); return
    method = args.method; dest = ROOT / method; dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            event(method, 'preflight'); collect(method); prepare(method); evaluate(method)
        except Exception as exc:
            event(method, 'failed', error=str(exc)); raise


if __name__ == '__main__':
    main()
