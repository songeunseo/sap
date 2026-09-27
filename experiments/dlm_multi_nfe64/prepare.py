"""Import verified controls into an isolated follow-up; preserve earlier runs."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
from experiments.dlm_multiscale_ac50.artifacts import read, sha, freeze, digest, mask_identity
from experiments.dlm_pruning_nfe50 import prepare as previous

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'output'
REPO = HERE.parents[1]
CROSS = previous.CROSS
MULTI_SHA = 'dfd54070f946efced8300cf99be192ba22c055760ab5b20856805e5dd953e792'
MULTI_MASK = '5b5f674ccc88409dc4067282b3b334b90a1292f12d4e150fce4cb785caa724ec'
PRUNED = 3489660928

def requests_for(cfg):
    return previous.requests_for(cfg)

def validate(config_path=None):
    path = Path(config_path or ROOT / 'config.json')
    cfg = read(path)
    for source, expected in cfg['sources'].items():
        if sha(source) != expected:
            raise RuntimeError('Frozen source changed: ' + source)
    if len(requests_for(cfg)) != 200 or len(set(cfg['exposed_ids'])) != 200:
        raise RuntimeError('Expected the same 200 unique exposed questions')
    if cfg['identity']['model_sha256_Multi'] != MULTI_SHA or cfg['identity']['mask_hash'] != MULTI_MASK:
        raise RuntimeError('Multi identity mismatch')
    if set(cfg['cells']) != {'Multi_64','Multi_256','A_64','A_256'}:
        raise RuntimeError('Unexpected experiment cells')
    for key, cell in cfg['cells'].items():
        steps = int(key.split('_')[1])
        if previous.protocol_hash(cell['protocol']) != cell['protocol_hash']:
            raise RuntimeError('Protocol hash mismatch')
        expected = copy.deepcopy(cfg['evaluation']); expected['denoising_steps'] = steps
        if cell['evaluation'] != expected or cell['protocol']['evaluation'] != expected:
            raise RuntimeError('Unapproved evaluation change')
        if cell['fingerprint']['config_identity'] != cfg['config_identity']:
            raise RuntimeError('Fingerprint mismatch')
    receipt = path.parent / 'code_receipt.json'
    if receipt.exists():
        cr = read(receipt)
        if cr['config_sha256'] != sha(path):
            raise RuntimeError('Code receipt config mismatch')
        for source, expected in cr['sources'].items():
            if sha(source) != expected:
                raise RuntimeError('Frozen code changed: ' + source)
    return cfg

def prepare():
    if (ROOT / 'started.json').exists():
        raise RuntimeError('Already started')
    old = previous.validate()
    if read(previous.ROOT / 'report.json')['status'] != 'complete':
        raise RuntimeError('Previous experiment incomplete')
    source_cfg = read(old['source']['config_path'])
    from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol, grade, validate_prediction, read_predictions
    from transformers import AutoTokenizer
    task, protocol, ph = task_and_protocol(source_cfg)
    requests = requests_for(old)
    tokenizer = AutoTokenizer.from_pretrained(old['model']['id'], revision=old['model']['revision'],
                                              trust_remote_code=True, local_files_only=True)
    for req in requests:
        if digest(tokenizer(req['prompt'])['input_ids']) != req['input_ids_sha256']:
            raise RuntimeError('Prompt tokenization changed')
    manifest_path = CROSS / 'candidates/Multi/mask_manifest.json'
    identity_path = CROSS / 'candidates/Multi/model_identity.json'
    manifest, ident = read(manifest_path), read(identity_path)
    if mask_identity(manifest) != MULTI_MASK or ident['sparse_model_sha256'] != MULTI_SHA:
        raise RuntimeError('Multi mask/model identity changed')
    if ident['mask_identity'] != MULTI_MASK or manifest['pruned'] != PRUNED or len(manifest['entries']) != 224:
        raise RuntimeError('Multi physical budget/coverage mismatch')
    sources = copy.deepcopy(old['sources'])
    for p in [previous.ROOT/'config.json', previous.ROOT/'code_receipt.json', previous.ROOT/'report.json',
              manifest_path, identity_path]:
        sources[str(p)] = sha(p)
    for entry in manifest['entries']:
        packed = entry['selected_mask']
        if sha(packed['path']) != packed['file_sha256']:
            raise RuntimeError('Packed Multi mask changed')
        sources[packed['path']] = packed['file_sha256']
    source = copy.deepcopy(old['source'])
    source.update(manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path))
    identity = copy.deepcopy(old['identity'])
    identity.update(mask_hash=MULTI_MASK, model_sha256_Multi=MULTI_SHA)
    science = dict(source=source, identity=identity, model=old['model'], evaluation=old['evaluation'],
        exposed_ids=old['exposed_ids'], schedules=[64], arms=['Multi'],
        statistics=dict(primary='Multi64-A64', secondary='(Multi64-A64)-(Multi256-A256)',
                        bootstrap_draws=10000, bootstrap_seed=20260927, unit='question'),
        selection='64 steps selected after Dense/A results; exploratory exposed200 follow-up')
    ci = digest(science)
    cells = {}
    for arm, steps in [('Multi',64),('Multi',256),('A',64),('A',256)]:
        base = copy.deepcopy(old['cells'][f'A_{steps}'])
        base['fingerprint'] = dict(config_identity=ci, cell=f'{arm}_{steps}',
            model_sha256=MULTI_SHA if arm=='Multi' else identity['model_sha256_A'],
            protocol_hash=base['protocol_hash'], requests_sha256=source['selected_requests_sha256'])
        cells[f'{arm}_{steps}'] = base
    imported = {}
    for key in ['A_64','A_256']:
        folder = previous.ROOT / 'gsm8k' / key
        oldcell = old['cells'][key]
        if read(folder/'identity.json') != dict(fingerprint=oldcell['fingerprint'],document_ids=old['exposed_ids']):
            raise RuntimeError('Source cell identity changed')
        rows = read_predictions(folder,requests,oldcell['fingerprint'],oldcell['protocol_hash'])
        if sorted(p.name for p in (folder/'examples').glob('*.json')) != sorted(f"{r['example_id']:04d}.json" for r in requests):
            raise RuntimeError('Duplicate/noncanonical source checkpoints')
        for p in [folder/'identity.json',folder/'predictions.json',folder/'results.json',*sorted((folder/'examples').glob('*.json'))]:
            sources[str(p)] = sha(p)
        if read(folder/'predictions.json') != rows or read(folder/'results.json')['correct'] != sum(r['correct'] for r in rows):
            raise RuntimeError('Source aggregate mismatch')
        imported[key] = rows
    multi_rows = []
    for req in requests:
        idx = req['example_id']; folder = CROSS/'gsm8k/Multi'/f'shard{idx//128:02d}'
        p = folder/'examples'/f'{idx:04d}.json'
        saved = read(p); row = saved['row']
        fp = dict(config_sha256=old['source']['config_sha256'], mask_identity=MULTI_MASK,
            protocol_hash=ph, requests_sha256=old['source']['requests_sha256'], shard=idx//128,
            sparse_model_sha256=MULTI_SHA, split='full')
        if saved['fingerprint'] != fp or saved['row_sha256'] != digest(row) or read(folder/'identity.json')['fingerprint'] != fp:
            raise RuntimeError('Multi256 cache identity mismatch')
        sources[str(p)] = sha(p); sources[str(folder/'identity.json')] = sha(folder/'identity.json')
        multi_rows.append(row)
    imported['Multi_256'] = multi_rows
    for key, rows in imported.items():
        for req, row in zip(requests,rows,strict=True):
            validate_prediction(row,req,cells[key]['protocol_hash'])
            if grade(task,req,row['generated_text']) != {k:row[k] for k in ('correct','extracted_answer')}:
                raise RuntimeError('Official cache regrade mismatch')
    cfg = dict(**science,config_identity=ci,cells=cells,sources=sources,
        package_versions=old['package_versions'],planned_cost=dict(new_generation_forwards=12800,
        validation_forwards=0,reused_generation_forwards=115200),
        limitations=['Exposed development200; NFE selected after preceding results',
                     'One fixed mask per method; no causal mechanism claim'])
    freeze(ROOT/'config.json',cfg)
    for key, rows in imported.items():
        folder=ROOT/'gsm8k'/key; fp=cells[key]['fingerprint']
        freeze(folder/'identity.json',dict(fingerprint=fp,document_ids=cfg['exposed_ids']))
        for row in rows:
            freeze(folder/'examples'/f"{row['example_id']:04d}.json",dict(fingerprint=fp,row=row,row_sha256=digest(row)))
        freeze(folder/'predictions.json',rows)
        freeze(folder/'results.json',dict(status='complete',fingerprint=fp,total=200,
            correct=sum(r['correct'] for r in rows),predictions_sha256=sha(folder/'predictions.json'),
            provenance='Verified imported control; zero new generation forwards'))
    freeze(ROOT/'preparation.json',dict(status='passed',config_sha256=sha(ROOT/'config.json'),
        verified_cached_rows=600,tokenized_prompts_verified=True,packed_masks_verified=224,
        correct={key:sum(r['correct'] for r in rows) for key,rows in imported.items()}))
    return validate()

def seal():
    validate()
    files=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))
    files += sorted(previous.HERE.glob('*.py'))
    return freeze(ROOT/'code_receipt.json',dict(config_sha256=sha(ROOT/'config.json'),
        sources={str(p):sha(p) for p in files}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','validate','seal'])
    args=parser.parse_args(); {'prepare':prepare,'validate':validate,'seal':seal}[args.command]()
    print(args.command,'passed',sha(ROOT/'config.json'))
