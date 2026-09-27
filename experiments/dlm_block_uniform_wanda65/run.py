"""Block-uniform Wanda: equal budget per Transformer block, pooled across projections."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_allocation_sequential65 import run as seq
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write, file_sha256 as sha

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
STORE = Path('/DATA/tmluser1/dlm-allocation-sequential65/block_uniform')
CONFIG = ROOT / 'config.json'
CAL = REPO / seq.CAL
REF_MANIFEST = seq.old.SOURCE / 'candidate_mask_manifest.json'
METHOD = 'block_uniform_wanda65'


def read(path):
    return json.loads(Path(path).read_text())


def event(stage, **values):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **values)
    write(ROOT / 'progress.json', value)
    print(json.dumps(value), flush=True)


def freeze():
    if CONFIG.exists():
        return validate()
    parent = seq.validate()
    refs = read(REF_MANIFEST)['entries']
    if len(refs) != 224:
        raise RuntimeError('expected 224 projection references')
    # Existing Uniform's exact row-floor count, split equally across 32 blocks.
    uniform_counts = [int(r['shape'][1] * .65) for r in refs]
    block_counts = [sum(k * r['shape'][0] for k, r in zip(uniform_counts[b * 7:(b + 1) * 7], refs[b * 7:(b + 1) * 7])) for b in range(32)]
    if len(set(block_counts)) != 1:
        raise RuntimeError('existing Uniform does not have equal block budgets')
    sources = dict(parent['sources'])
    sources.update({str(p): sha(p) for p in (Path(__file__), ROOT / 'test_block_uniform.py', REF_MANIFEST)})
    cfg = dict(
        method=METHOD,
        parent_sequential_config_sha256=sha(seq.ROOT / 'config.json'),
        model=parent['model'],
        target=.65,
        weights=parent['weights'],
        pruned=parent['pruned'],
        calibration_states=80,
        projections=224,
        blocks=32,
        projections_per_block=7,
        allocation='equal exact row-floor pruning count per block; pooled Wanda score threshold within block',
        local_score='abs(weight) * sqrt(input activation energy), same Standard Wanda score',
        prefix='native batch-one sparse-prefix replay; later activations observe earlier block masks',
        block_pruned_count=block_counts[0],
        block_pruned_counts=block_counts,
        protocol_hash=parent['protocol_hash'],
        sources=sources,
    )
    write(CONFIG, cfg)
    return cfg


def validate():
    cfg = read(CONFIG)
    for path, digest in cfg['sources'].items():
        if sha(path) != digest:
            raise RuntimeError('changed source: ' + path)
    if sha(seq.ROOT / 'config.json') != cfg['parent_sequential_config_sha256']:
        raise RuntimeError('parent sequential config changed')
    return cfg


def pooled_masks(model, mapping, refs, states, block, block_count):
    """Collect sparse-prefix activations and select the block's lowest pooled scores."""
    activation = seq.collect_block(model, mapping, refs, states, block,
                                   lambda i: event('calibration', block=block, completed=i, total=len(states)))
    score_parts = []
    for i in range(block * 7, block * 7 + 7):
        ref = refs[i]
        weight = mapping[ref['name']].weight
        a = activation[ref['name']].to(weight.device)
        score = (weight.float().abs() * a.sqrt()[None, :]).cpu()
        score_parts.append(score.reshape(-1))
    flat = torch.cat(score_parts)
    if not torch.isfinite(flat).all() or (flat < 0).any():
        raise RuntimeError('invalid pooled Wanda scores')
    # kthvalue gives an exact threshold; deterministic tie fill preserves exact budget.
    threshold = torch.kthvalue(flat, int(block_count)).values
    chosen = flat < threshold
    need = int(block_count - int(chosen.sum().item()))
    if need < 0:
        raise RuntimeError('threshold selected too many weights')
    if need:
        equal = torch.nonzero(flat == threshold, as_tuple=False).flatten()
        if equal.numel() < need:
            raise RuntimeError('insufficient threshold ties')
        chosen[equal[:need]] = True
    if int(chosen.sum().item()) != block_count:
        raise RuntimeError('block pooled budget mismatch')
    masks = []
    offset = 0
    for part, i in zip(score_parts, range(block * 7, block * 7 + 7)):
        shape = tuple(refs[i]['shape'])
        n = int(part.numel())
        masks.append(chosen[offset:offset + n].reshape(shape))
        offset += n
    del activation, score_parts, flat, chosen
    return masks


@torch.inference_mode()
def build(model, mapping, cfg):
    from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
    from experiments.projection_capacity_allocation_65.run import save_tensor, DENSE_SHA
    from experiments.wanda_failure_characterization.run_failure_map import model_sha

    manifest_path = ROOT / 'mask_manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest.get('config_sha256') != sha(CONFIG):
            raise RuntimeError('manifest config hash mismatch')
        return manifest
    refs = read(REF_MANIFEST)['entries']
    states = read(CAL)['states']
    if list(mapping) != [r['name'] for r in refs]:
        raise RuntimeError('projection order mismatch')
    STORE.mkdir(parents=True, exist_ok=True)
    entries = []
    block_receipts = []
    for block in range(32):
        event('block_start', block=block, total=32)
        masks = pooled_masks(model, mapping, refs, states, block, cfg['block_pruned_count'])
        block_total = 0
        for local, i in enumerate(range(block * 7, block * 7 + 7)):
            ref = refs[i]; name = ref['name']; mask = masks[local]
            packed = pack_mask(mask)
            path = STORE / 'masks' / f'{name}.pt'
            if path.exists():
                old = torch.load(path, map_location='cpu', weights_only=False)
                if mask_sha256(old) != mask_sha256(packed):
                    raise RuntimeError('partial mask differs')
            else:
                save_tensor(path, packed)
            pruned = int(mask.sum().item()); block_total += pruned
            rows = mask.sum(1).tolist()
            entries.append(dict(
                module_index=i, name=name, shape=ref['shape'], weights=ref['weights'],
                assigned_sparsity=pruned / ref['weights'],
                ideal_block_sparsity=.65,
                selected_mask=dict(path=str(path), file_sha256=sha(path), mask_sha256=mask_sha256(packed),
                                   pruned=pruned, row_pruned_counts=rows),
            ))
            event('mask', block=block, module=i, total=224, pruned=pruned)
        if block_total != cfg['block_pruned_count']:
            raise RuntimeError('block total mismatch')
        block_receipts.append(dict(block=block, pruned=block_total))
        # Apply this block before collecting the next block's sparse-prefix activations.
        for local, i in enumerate(range(block * 7, block * 7 + 7)):
            mapping[refs[i]['name']].weight.masked_fill_(masks[local].to(mapping[refs[i]['name']].weight.device), 0)
        event('block_complete', block=block, total=32, pruned=block_total)
    if model_sha(model) != DENSE_SHA:
        # DENSE_SHA is the unmasked model digest; masks are expected to change it.
        pass
    manifest = dict(method=METHOD, config_sha256=sha(CONFIG), entries=entries,
                    pruned=sum(e['selected_mask']['pruned'] for e in entries), weights=sum(e['weights'] for e in entries),
                    block_receipts=block_receipts)
    if manifest['pruned'] != cfg['pruned']:
        raise RuntimeError('global budget mismatch')
    write(manifest_path, manifest)
    write(ROOT / 'allocation.json', dict(block_pruned_counts=[x['pruned'] for x in block_receipts],
                                          projection_sparsities=[e['assigned_sparsity'] for e in entries]))
    return manifest


@torch.no_grad()
def apply_manifest(model, mapping, manifest):
    from experiments.dlm_loss_aggregation.core import unpack_mask, mask_sha256
    counted = 0
    for row in manifest['entries']:
        module = mapping[row['name']]; meta = row['selected_mask']; path = Path(meta['path'])
        if sha(path) != meta['file_sha256']:
            raise RuntimeError('mask file changed')
        packed = torch.load(path, map_location='cpu', weights_only=False)
        if mask_sha256(packed) != meta['mask_sha256']:
            raise RuntimeError('mask payload changed')
        mask = unpack_mask(packed).to(module.weight.device)
        if tuple(mask.shape) != tuple(module.weight.shape) or int(mask.sum()) != meta['pruned']:
            raise RuntimeError('mask shape/count mismatch')
        module.weight.masked_fill_(mask, 0); counted += int(mask.sum())
    if counted != manifest['pruned']:
        raise RuntimeError('applied budget mismatch')
    return counted


def run():
    cfg = validate()
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.dlm_loss_aggregation.exp002.run import load_config, _evaluation_config_hash, _evaluate_gsm8k, _write_jsonl
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    from transformers import AutoTokenizer

    result_path = ROOT / 'results.json'
    if result_path.exists():
        event('complete', correct=read(result_path)['correct']); return
    event('loading_dense'); model, mapping = load_dense()
    refs = read(REF_MANIFEST)['entries']; manifest = build(model, mapping, cfg)
    # build leaves the model sparse; reload dense for a clean manifest application.
    del model, mapping
    model, mapping = load_dense()
    apply_manifest(model, mapping, manifest)
    evalcfg = load_config(seq.old.EVAL); ph, _ = _evaluation_config_hash(evalcfg)
    if ph != cfg['protocol_hash']:
        raise RuntimeError('protocol mismatch')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['id'], revision=cfg['model']['revision'], trust_remote_code=True)
    event('gsm8k', completed=0, total=100)
    metrics, predictions = _evaluate_gsm8k(model, tokenizer, evalcfg, METHOD, 100, ph)
    baseline_path = seq.ROOT / 'uniform' / 'predictions.jsonl'
    baseline = seq.old.read_jsonl(baseline_path)
    _validate_rows(predictions, baseline, ph)
    _write_jsonl(ROOT / 'predictions.jsonl', predictions)
    result = dict(status='complete', correct=sum(r['correct'] for r in predictions), total=100,
                  metrics=metrics, protocol_hash=ph, config_sha256=sha(CONFIG),
                  manifest_sha256=sha(ROOT / 'mask_manifest.json'), predictions_sha256=sha(ROOT / 'predictions.jsonl'),
                  comparisons={'sequential_uniform': paired_binary_comparison([r['correct'] for r in baseline], [r['correct'] for r in predictions])})
    write(result_path, result)
    event('complete', correct=result['correct'], total=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'run'])
    args = parser.parse_args(); ROOT.mkdir(exist_ok=True)
    with (ROOT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'freeze': freeze()
        else: run()
