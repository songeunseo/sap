#!/usr/bin/env python3
"""Prepare and evaluate the frozen global role-risk allocation on mini GSM8K."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.dlm_capacity_predictor.audit_existing import strict_exact_match
from experiments.dlm_dual_role_allocation.io import atomic_write_json, file_sha256, load_frozen_inputs
from experiments.dlm_dual_role_mini100.core import validate_manifest
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples, _evaluate_gsm8k, _evaluation_config_hash,
    _read_jsonl, _write_jsonl, load_config as eval_config,
)
from experiments.dlm_role_validation.run import load_dense
from experiments.projection_capacity_allocation_65.run import DENSE_SHA
from experiments.projection_capacity_followup_65.core import build_selected_manifest, paired_binary_comparison
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
from experiments.wanda_failure_characterization.run_failure_map import model_sha

ROOT = Path('experiments/dlm_role_global_minimax_mini100')
DP = Path('experiments/dlm_role_tradeoff/dp_results.json')
CANDIDATES = Path('experiments/projection_capacity_allocation_65/candidate_mask_manifest.json')
OLD_MINI = Path('experiments/dlm_dual_role_mini100/mini100_results.json')
OLD_ROLE = Path('experiments/dlm_dual_role_mini100/role65_mask_manifest.json')
EVAL_CONFIG = Path('experiments/dlm_loss_aggregation/exp002/config.yaml')
GRID = (.50, .55, .60, .65, .70, .75)
TARGET = 4_536_008_704
WEIGHTS = 6_979_321_856
METHOD = 'global_role_supported_minimax'


def sha(path): return file_sha256(path)


def event(kind, **values):
    print(json.dumps({'event': kind, 'time': time.time(), **values}, sort_keys=True), flush=True)


def validate_rows(rows, reference, protocol_hash):
    if len(rows) != 100:
        raise RuntimeError('mini evaluation requires exactly 100 examples')
    _assert_same_examples(reference[:100], rows)
    for index, row in enumerate(rows):
        if row.get('example_id') != index or row.get('evaluation_config_hash') != protocol_hash:
            raise RuntimeError('prediction identity/protocol mismatch')
        if strict_exact_match(row.get('extracted_answer'), row.get('reference_answer')) != row.get('correct'):
            raise RuntimeError('stored strict EM mismatch')


def prepare():
    inputs = load_frozen_inputs()
    dp = json.loads(DP.read_text())
    if dp.get('status') != 'complete' or dp.get('frontier_search_capped'):
        raise RuntimeError('tradeoff analysis is incomplete or capped')
    candidate = dp['supported_minimax']
    levels = list(map(int, candidate['levels']))
    if len(levels) != 224 or any(not 0 <= x < 6 for x in levels):
        raise RuntimeError('invalid allocation levels')
    protocol_hash, protocol = _evaluation_config_hash(eval_config(EVAL_CONFIG))
    config = {
        'status': 'frozen_before_mini', 'method': METHOD,
        'model': inputs.config['model'], 'dense_model_sha256': DENSE_SHA,
        'selector': 'persisted exact Standard Wanda candidate masks',
        'allocation': {
            'source': str(DP), 'alpha_masked': candidate['alpha_masked'],
            'objective': 'supported frontier point minimizing maximum relative total role risk',
            'relative_risks': candidate['relative_risks'], 'levels': levels,
        },
        'target_pruned': TARGET, 'weights': WEIGHTS, 'grid': list(GRID),
        'evaluation': {'dataset': 'historical fixed GSM8K mini-100',
                       'protocol_hash': protocol_hash, 'protocol': protocol,
                       'primary': 'strict exact match'},
        'references': {'role_max': 24, 'aggregate': 19, 'uniform': 12},
        'decision_rule': 'directional screen against frozen Role-Max; no alpha or grid retuning',
        'source_hashes': {str(DP): sha(DP), str(CANDIDATES): sha(CANDIDATES),
                          str(OLD_MINI): sha(OLD_MINI), str(OLD_ROLE): sha(OLD_ROLE),
                          str(EVAL_CONFIG): sha(EVAL_CONFIG)},
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('frozen config differs')
    if not config_path.exists(): atomic_write_json(config_path, config)
    for path, digest in config['source_hashes'].items():
        if sha(path) != digest: raise RuntimeError(f'frozen source changed: {path}')

    source = json.loads(CANDIDATES.read_text())
    if [x['name'] for x in source['entries']] != inputs.metadata['module_names']:
        raise RuntimeError('candidate module order mismatch')
    sparsities = [GRID[x] for x in levels]
    manifest = build_selected_manifest(METHOD, source['entries'], sparsities, GRID)
    manifest.update(config_sha256=sha(config_path), allocation_levels=levels,
                    alpha_masked=candidate['alpha_masked'], source_dp_sha256=sha(DP))
    validate_manifest(manifest, inputs.metadata['module_names'], TARGET, WEIGHTS)
    manifest_path = ROOT / 'mask_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('frozen manifest differs')
    if not manifest_path.exists(): atomic_write_json(manifest_path, manifest)

    old_role = json.loads(OLD_ROLE.read_text())
    changed = sum(a['selected_mask']['mask_sha256'] != b['selected_mask']['mask_sha256']
                  for a, b in zip(manifest['entries'], old_role['entries']))
    xor = sum(abs(int(a['selected_mask']['pruned']) - int(b['selected_mask']['pruned']))
              for a, b in zip(manifest['entries'], old_role['entries']))
    prep = {'status': 'frozen', 'config_sha256': sha(config_path),
            'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
            'changed_projections_vs_role': changed, 'mask_xor_vs_role': xor,
            'mask_xor_fraction_vs_role': xor / WEIGHTS,
            'level_counts': {str(x): sparsities.count(x) for x in GRID}}
    prep_path = ROOT / 'preparation.json'
    if prep_path.exists() and json.loads(prep_path.read_text()) != prep:
        raise RuntimeError('frozen preparation differs')
    if not prep_path.exists(): atomic_write_json(prep_path, prep)
    event('preparation_complete', changed=changed, xor_fraction=xor / WEIGHTS)
    return config, prep, manifest


@torch.inference_mode()
def evaluate():
    config_doc, prep, manifest = prepare()
    config = eval_config(EVAL_CONFIG)
    protocol_hash, protocol = _evaluation_config_hash(config)
    if protocol_hash != config_doc['evaluation']['protocol_hash']:
        raise RuntimeError('evaluation protocol changed')
    dense_rows = _read_jsonl(Path('experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'))
    historical = json.loads(OLD_MINI.read_text())
    refs = {name: _read_jsonl(Path(historical['methods'][name]['predictions']))
            for name in ('role', 'aggregate')}
    for rows in refs.values(): validate_rows(rows, dense_rows, protocol_hash)

    output = ROOT / 'gsm8k/minimax_100_predictions.jsonl'
    receipt_path = output.with_suffix('.receipt.json')
    fingerprint = {'config_sha256': sha(ROOT/'config.json'),
                   'manifest_sha256': prep['manifest_sha256'],
                   'protocol_sha256': protocol_hash, 'method': METHOD, 'limit': 100}
    rows = None
    if output.exists() or receipt_path.exists():
        if not output.exists() or not receipt_path.exists(): raise RuntimeError('partial output artifacts')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('fingerprint') != fingerprint or receipt.get('predictions_sha256') != sha(output):
            raise RuntimeError('completed output receipt mismatch')
        rows = _read_jsonl(output); validate_rows(rows, dense_rows, protocol_hash)
        metrics = receipt['metrics']
    if rows is None:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['id'], revision=config['model']['revision'], trust_remote_code=True)
        model, mapping = load_dense()
        if apply_manifest(model, mapping, manifest) != TARGET: raise RuntimeError('applied budget mismatch')
        sparse_sha = model_sha(model)
        metrics, rows = _evaluate_gsm8k(model, tokenizer, config, METHOD, 100, protocol_hash)
        if model_sha(model) != sparse_sha: raise RuntimeError('sparse model changed')
        validate_rows(rows, dense_rows, protocol_hash)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output, rows)
        atomic_write_json(receipt_path, {'status': 'complete', 'fingerprint': fingerprint,
                                         'predictions_sha256': sha(output),
                                         'sparse_model_sha256': sparse_sha, 'metrics': metrics})
        del model; gc.collect(); torch.cuda.empty_cache()
    correct = int(sum(bool(x['correct']) for x in rows))
    result = {'status': 'complete', 'method': METHOD, 'correct': correct, 'total': 100,
              'metrics': metrics, 'predictions': str(output), 'predictions_sha256': sha(output),
              'protocol_hash': protocol_hash, 'protocol': protocol,
              'references': {'role': 24, 'aggregate': 19, 'uniform': 12},
              'paired_vs_role': paired_binary_comparison([x['correct'] for x in refs['role']], [x['correct'] for x in rows]),
              'paired_vs_aggregate': paired_binary_comparison([x['correct'] for x in refs['aggregate']], [x['correct'] for x in rows]),
              'decision': 'DIRECTIONALLY ABOVE ROLE' if correct > 24 else ('TIED ROLE' if correct == 24 else 'BELOW ROLE'),
              'interpretation_limit': 'exploratory fixed mini-100 screen'}
    atomic_write_json(ROOT/'mini100_results.json', result)
    event('mini_complete', correct=correct, decision=result['decision'])
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=('prepare','evaluate','all')); args=parser.parse_args()
    torch.set_num_threads(8); torch.manual_seed(0); np.random.seed(0)
    if args.phase in ('prepare','all'): prepare()
    if args.phase in ('evaluate','all'): evaluate()


if __name__ == '__main__': main()
