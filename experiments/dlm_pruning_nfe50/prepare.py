"""Prepare isolated CPU artifacts; never rewrite an earlier experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
from pathlib import Path

from experiments.dlm_multiscale_ac50.artifacts import read, sha, write, freeze, digest, mask_identity

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = HERE / 'output'
CROSS = REPO / 'experiments/dlm_crosschain_control50/output'
DENSE_SHA = '2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc'
EXPECTED = {
    'config.json': '59912694489062d4d7c08dece821c36ad292f95a4b42a7603851b7d14347d06b',
    'requests.json': '3ed3e16ecd93ab9081c40a9bd523f09e6bffb8b6da7f78803b7a729609d4f5a8',
    'candidates/A/mask_manifest.json': '5afb660b8a005bfb7f9ec3f630de54fe13e0d56dee8b7da66c683b857903b7bc',
}


def protocol_hash(protocol):
    import json
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def requests_for(cfg):
    rows = read(cfg['source']['selected_requests_path'])['development']
    if [r['example_id'] for r in rows] != cfg['exposed_ids']:
        raise RuntimeError('Selected request IDs changed')
    return rows


def cell_info(cfg, arm, steps):
    return cfg['cells'][f'{arm}_{steps}']


def validate(config_path=None):
    cfg = read(config_path or ROOT / 'config.json')
    for path, expected in cfg['sources'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    if len(cfg['exposed_ids']) != 200 or len(set(cfg['exposed_ids'])) != 200:
        raise RuntimeError('Expected 200 unique exposed development IDs')
    requests_for(cfg)
    for key, cell in cfg['cells'].items():
        if protocol_hash(cell['protocol']) != cell['protocol_hash']:
            raise RuntimeError('Protocol hash mismatch: ' + key)
        ev = cell['evaluation']
        if ev['generation_length'] != 256 or ev['block_length'] != 256 or ev['temperature'] != 0:
            raise RuntimeError('Unapproved decoding change')
        if cell['fingerprint']['protocol_hash'] != cell['protocol_hash']:
            raise RuntimeError('Fingerprint protocol mismatch')
    receipt = ROOT / 'code_receipt.json'
    if receipt.exists():
        cr = read(receipt)
        if cr['config_sha256'] != sha(config_path or ROOT / 'config.json'):
            raise RuntimeError('Frozen code receipt has wrong config')
        for path, expected in cr['sources'].items():
            if sha(path) != expected:
                raise RuntimeError('Frozen run code changed: ' + path)
    return cfg


def prepare():
    if (ROOT / 'started.json').exists():
        raise RuntimeError('Experiment already started; preparation is disabled')
    for path, expected in EXPECTED.items():
        if sha(CROSS / path) != expected:
            raise RuntimeError('Approved source changed: ' + path)
    old = read(CROSS / 'config.json')
    original = read(CROSS / 'requests.json')
    manifest = read(CROSS / 'candidates/A/mask_manifest.json')
    ident = read(CROSS / 'candidates/A/model_identity.json')
    if mask_identity(manifest) != ident['mask_identity'] or manifest['pruned'] != 3489660928:
        raise RuntimeError('A mask identity/budget changed')
    ids = old['crosschain_control']['exposed_ids']
    selected = [r for r in original['development'] if r['example_id'] in set(ids)]
    if [r['example_id'] for r in selected] != ids or len(ids) != 200:
        raise RuntimeError('Development sample mismatch')
    from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol, grade, validate_prediction
    from transformers import AutoTokenizer
    task, old_protocol, old_hash = task_and_protocol(old)
    if old_protocol != original['protocol'] or old_hash != original['protocol_hash']:
        raise RuntimeError('Frozen official evaluation protocol mismatch')
    tokenizer = AutoTokenizer.from_pretrained(old['model']['id'], revision=old['model']['revision'],
                                             trust_remote_code=True, local_files_only=True)
    for req in selected:
        if digest(tokenizer(req['prompt'])['input_ids']) != req['input_ids_sha256']:
            raise RuntimeError('Tokenized prompt changed')
    selected_doc = dict(development=selected, confirmation=[],
                        dataset_fingerprint=original['dataset_fingerprint'],
                        protocol=old_protocol, protocol_hash=old_hash,
                        exposure='Previously exposed development200; exploratory')
    freeze(ROOT / 'requests.json', selected_doc)
    relevant = [REPO / p for p in (
        'generate.py', 'eval_llada.py',
        'experiments/dlm_multiscale_ac50/evaluation.py',
        'experiments/dlm_multiscale_ac50/artifacts.py',
        'experiments/projection_capacity_allocation_65/run.py',
        'experiments/projection_capacity_followup_65/run_heldout.py',
        'experiments/wanda_failure_characterization/run_failure_map.py',
        'experiments/dlm_loss_aggregation/run.py',
        'experiments/dlm_loss_aggregation/core.py',
        'experiments/dlm_loss_aggregation/config.yaml',
        'model/modeling_llada.py', 'model/configuration_llada.py')]
    for p in relevant:
        if str(p) in old['sources'] and sha(p) != old['sources'][str(p)]:
            raise RuntimeError('Legacy model/evaluation code changed: ' + str(p))
    sources = {str(p):sha(p) for p in relevant}
    sources.update({str(CROSS / p):sha(CROSS / p) for p in EXPECTED})
    sources[str(CROSS / 'candidates/A/model_identity.json')] = sha(CROSS / 'candidates/A/model_identity.json')
    sources[str(ROOT / 'requests.json')] = sha(ROOT / 'requests.json')
    for e in manifest['entries']:
        s = e['selected_mask']
        if sha(s['path']) != s['file_sha256']:
            raise RuntimeError('Packed A mask changed')
        sources[s['path']] = s['file_sha256']
    proposal = REPO / 'research/next_experiment_2026-09-27/proposal.md'
    sources[str(proposal)] = sha(proposal)
    source = dict(config_path=str(CROSS / 'config.json'), config_sha256=sha(CROSS / 'config.json'),
                  requests_path=str(CROSS / 'requests.json'), requests_sha256=sha(CROSS / 'requests.json'),
                  manifest_path=str(CROSS / 'candidates/A/mask_manifest.json'),
                  manifest_sha256=sha(CROSS / 'candidates/A/mask_manifest.json'),
                  selected_requests_path=str(ROOT / 'requests.json'),
                  selected_requests_sha256=sha(ROOT / 'requests.json'),
                  model_load_config_path=str(REPO / 'experiments/dlm_loss_aggregation/config.yaml'))
    identity = dict(protocol_hash=old_hash, model_revision=old['model']['revision'],
                    model_sha256_dense=DENSE_SHA, model_sha256_A=ident['sparse_model_sha256'],
                    mask_hash=ident['mask_identity'], pruned_count=3489660928, mask_id=126336)
    science = dict(source=source, identity=identity, model=old['model'], evaluation=old['evaluation'],
                   exposed_ids=ids, schedules=[256,64,32], arms=['Dense','A'],
                   diagnostic=dict(steps=32,capture_steps=[0,8,16,24],top_k=8,readout_dtype='float32'),
                   statistics=dict(primary='I32=(Dense32-A32)-(Dense256-A256)',
                                   secondary='I64=(Dense64-A64)-(Dense256-A256)',
                                   unit='question',bootstrap_seed=20260927,bootstrap_draws=10000))
    cells = {}
    science_hash = digest(science)
    for arm in science['arms']:
        for steps in science['schedules']:
            key = f'{arm}_{steps}'
            evaluation = copy.deepcopy(old['evaluation']); evaluation['denoising_steps'] = steps
            protocol = copy.deepcopy(old_protocol); protocol['evaluation'] = evaluation
            ph = protocol_hash(protocol)
            fp = dict(config_identity=science_hash,cell=key,model_sha256=identity['model_sha256_dense' if arm=='Dense' else 'model_sha256_A'],
                      protocol_hash=ph,requests_sha256=source['selected_requests_sha256'])
            cells[key] = dict(evaluation=evaluation,protocol=protocol,protocol_hash=ph,fingerprint=fp)
    cache_rows, cache_sources = [], []
    for req in selected:
        idx = req['example_id']; folder = CROSS / 'gsm8k/A' / f'shard{idx//128:02d}'
        p = folder / 'examples' / f'{idx:04d}.json'
        saved = read(p); row = saved['row']
        fp = saved['fingerprint']
        expected_fp = dict(config_sha256=source['config_sha256'],mask_identity=identity['mask_hash'],
                           protocol_hash=old_hash,requests_sha256=source['requests_sha256'],shard=idx//128,
                           sparse_model_sha256=identity['model_sha256_A'],split='full')
        if fp != expected_fp or saved['row_sha256'] != digest(row):
            raise RuntimeError('A256 source identity changed')
        if read(folder / 'identity.json')['fingerprint'] != fp:
            raise RuntimeError('A256 shard identity mismatch')
        validate_prediction(row,req,old_hash)
        if grade(task,req,row['generated_text']) != {k:row[k] for k in ('correct','extracted_answer')}:
            raise RuntimeError('Official regrading differs')
        sources[str(p)] = sha(p)
        cache_rows.append(row)
        cache_sources.append(dict(example_id=idx,path=str(p),sha256=sha(p)))
    dense_path = REPO / 'experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'
    dense_eval_path = REPO / 'experiments/dlm_loss_aggregation/exp002/logs/evaluation_config.json'
    import json
    if sha(dense_path) != '1a1523c089558ee3df2ad1b49a937a3b01290060d67a51b8715e3374dc866b8e':
        raise RuntimeError('Dense256 source changed')
    dense_all = [json.loads(line) for line in dense_path.read_text().splitlines() if line.strip()]
    if len(dense_all) != 1319 or len({r['example_id'] for r in dense_all}) != 1319:
        raise RuntimeError('Dense256 source coverage changed')
    dense_map = {r['example_id']:r for r in dense_all}
    dense_rows = []
    for req in selected:
        row = dense_map[req['example_id']]
        validate_prediction(row,req,old_hash)
        if grade(task,req,row['generated_text']) != {k:row[k] for k in ('correct','extracted_answer')}:
            raise RuntimeError('Dense256 official regrading differs')
        dense_rows.append(row)
    sources[str(dense_path)] = sha(dense_path)
    if dense_eval_path.exists():
        sources[str(dense_eval_path)] = sha(dense_eval_path)
    cfg = dict(**science, schema=1, status='frozen_before_run',cells=cells,sources=sources,
               config_identity=science_hash,cache=dict(A256_folder=str(ROOT / 'gsm8k/A_256'),
                 A256_sources=cache_sources,Dense256_folder=str(ROOT / 'gsm8k/Dense_256'),
                 Dense256_source=str(dense_path),Dense256_source_sha256=sha(dense_path),
                 Dense256_requires_fresh_first_example_parity=True),
               package_versions={p:importlib.metadata.version(p) for p in old['package_versions']},
               planned_cost=dict(new_generation_forwards=38400,diagnostic_forwards_max=3200,
                                 validation_forwards=288,reused_generation_forwards=102400),
               limitations=['exposed development200','one fixed50% mask','no dependency causal identification'])
    freeze(ROOT / 'config.json',cfg)
    cell=cells['A_256']; folder=ROOT / 'gsm8k/A_256'
    freeze(folder / 'identity.json',dict(fingerprint=cell['fingerprint'],document_ids=ids))
    for row in cache_rows:
        freeze(folder / 'examples' / f"{row['example_id']:04d}.json",dict(fingerprint=cell['fingerprint'],row=row,row_sha256=digest(row)))
    freeze(folder / 'predictions.json',cache_rows)
    freeze(folder / 'results.json',dict(status='complete',fingerprint=cell['fingerprint'],total=200,
           correct=sum(r['correct'] for r in cache_rows),predictions_sha256=sha(folder / 'predictions.json'),
           generation_seconds=sum(r['eval_seconds'] for r in cache_rows),
           provenance='Imported old A256; historical seconds, zero new generation forwards'))
    cell=cells['Dense_256']; folder=ROOT / 'gsm8k/Dense_256'
    freeze(folder / 'identity.json',dict(fingerprint=cell['fingerprint'],document_ids=ids))
    for row in dense_rows:
        freeze(folder / 'examples' / f"{row['example_id']:04d}.json",dict(fingerprint=cell['fingerprint'],row=row,row_sha256=digest(row)))
    freeze(folder / 'predictions.json',dense_rows)
    freeze(folder / 'results.json',dict(status='complete',fingerprint=cell['fingerprint'],total=200,
           correct=sum(r['correct'] for r in dense_rows),predictions_sha256=sha(folder / 'predictions.json'),
           generation_seconds=None,provenance='Imported legacy Dense256; original row times unavailable; zero new generation forwards',
           acceptance_requires='Current physical Dense hash and first-example exact text parity at runtime'))
    freeze(ROOT / 'preparation.json',dict(status='passed' ,config_sha256=sha(ROOT / 'config.json'),
           verified_cached_rows=400,correct=dict(A256=sum(r['correct'] for r in cache_rows),Dense256=sum(r['correct'] for r in dense_rows)),
           official_regrade=True,tokenized_prompts_verified=True,packed_masks_verified=224))
    return validate()


def seal_code():
    cfg=validate()
    files=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))
    receipt=dict(config_sha256=sha(ROOT / 'config.json'),sources={str(p):sha(p) for p in files})
    freeze(ROOT / 'code_receipt.json',receipt)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','validate','seal'])
    args=parser.parse_args()
    result={'prepare':prepare,'validate':validate,'seal':seal_code}[args.command]()
    print(args.command,'passed',sha(ROOT / 'config.json'))
