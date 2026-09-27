"""Frozen WikiText NELBO evaluation of the four cross-chain control masks."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.dlm_wikitext_ppl import run as protocol
from experiments.dlm_wikitext_ppl import evaluate as evaluator
from experiments.dlm_wikitext_ppl.core import digest, summarize
from experiments.dlm_multiscale_ac50.artifacts import mask_identity

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent / 'output'
CROSS = REPO / 'experiments/dlm_crosschain_control50/output'
ARMS = ('A', 'Multi', 'Cross', 'CrossMatched')
SESSION = 'dlm_crosschain_wikitext50'
MODULE = 'experiments.dlm_crosschain_wikitext50.run'
TARGET = 3489660928


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    protocol.write(path, value)


def sha(path):
    return protocol.sha(path)


def manifest_path(arm):
    return CROSS / 'candidates' / arm / 'mask_manifest.json'


def identity_path(arm):
    return CROSS / 'candidates' / arm / 'model_identity.json'


def prepare():
    pcfg = protocol.validate()
    ccfg = read(CROSS / 'config.json')
    if ccfg['model']['id'] != pcfg['model']['id'] or ccfg['model']['revision'] != pcfg['model']['revision']:
        raise RuntimeError('Model/revision differs from frozen WikiText protocol')
    corpus = read(protocol.ROOT / 'corpus_manifest.json')['splits']['validation']
    if len(corpus) != 551 or sum(len(x['clean_ids']) for x in corpus) != 268163:
        raise RuntimeError('Frozen validation corpus size changed')
    sources = [Path(__file__), Path(__file__).with_name('run.sh'),
               CROSS / 'config.json', CROSS / 'allocation.json',
               protocol.ROOT / 'config.json', protocol.ROOT / 'corpus_manifest.json',
               Path(evaluator.__file__), protocol.ROOT / 'core.py']
    for arm in ARMS:
        m = read(manifest_path(arm)); ident = read(identity_path(arm))
        if m['pruned'] != TARGET or len(m['entries']) != 224 or mask_identity(m) != ident['mask_identity']:
            raise RuntimeError('Candidate mask identity/budget changed: ' + arm)
        sources.extend((manifest_path(arm), identity_path(arm)))
    cfg = dict(status='frozen', purpose='exploratory evaluation after Full GSM8K result',
               arms=list(ARMS), split='WikiText-2 filtered validation; final test untouched',
               primary_metric='token NELBO, exp(NELBO) estimate; MC128 exact-k shared draws',
               model=pcfg['model'], pruning=dict(pruned=TARGET, total=TARGET*2),
               corpus_blocks=len(corpus), corpus_tokens=sum(len(x['clean_ids']) for x in corpus),
               mc_samples=pcfg['mc_samples'], seed=pcfg['seed'],
               protocol_config_sha256=sha(protocol.ROOT / 'config.json'),
               corpus_sha256=sha(protocol.ROOT / 'corpus_manifest.json'),
               candidates={arm:dict(manifest_sha256=sha(manifest_path(arm)),
                                    mask_identity=read(identity_path(arm))['mask_identity'],
                                    sparse_model_sha256=read(identity_path(arm))['sparse_model_sha256']) for arm in ARMS},
               sources={str(p):sha(p) for p in sources})
    protocol.frozen(ROOT / 'config.json', cfg)
    return validate()


def validate():
    cfg = read(ROOT / 'config.json')
    protocol.validate()
    if cfg['protocol_config_sha256'] != sha(protocol.ROOT / 'config.json') or cfg['corpus_sha256'] != sha(protocol.ROOT / 'corpus_manifest.json'):
        raise RuntimeError('WikiText protocol/corpus changed')
    for path, expected in cfg['sources'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen source changed: ' + path)
    if cfg['arms'] != list(ARMS) or cfg['corpus_blocks'] != 551 or cfg['mc_samples'] != 128:
        raise RuntimeError('Frozen evaluation settings changed')
    for arm in ARMS:
        m = read(manifest_path(arm)); ident = read(identity_path(arm))
        if m['pruned'] != TARGET or mask_identity(m) != ident['mask_identity'] or ident['mask_identity'] != cfg['candidates'][arm]['mask_identity']:
            raise RuntimeError('Candidate identity/budget changed: ' + arm)
        if ident['sparse_model_sha256'] != cfg['candidates'][arm]['sparse_model_sha256']:
            raise RuntimeError('Sparse-model identity changed: ' + arm)
    return cfg


def block_path(arm, source):
    return ROOT / arm / 'blocks' / (digest(source['block_id'])[:20] + '.json')


def progress(arm, stage, **kw):
    write(ROOT / arm / 'progress.json', dict(arm=arm, stage=stage, updated=time.time(), **kw))


def worker(arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    if not os.environ.get('TMUX'):
        raise RuntimeError('GPU worker requires tmux')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.isdigit() or ',' in visible:
        raise RuntimeError('Exactly one GPU must be visible')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA GPU required')
    folder = ROOT / arm
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = validate()
        pcfg = protocol.validate()
        corpus = read(protocol.ROOT / 'corpus_manifest.json')['splits']['validation']
        config_sha = sha(ROOT / 'config.json')
        run_cfg = dict(arm=arm, split='validation', config_sha256=config_sha,
                       mask_manifest_sha256=cfg['candidates'][arm]['manifest_sha256'],
                       mask_identity=cfg['candidates'][arm]['mask_identity'],
                       sparse_model_sha256=cfg['candidates'][arm]['sparse_model_sha256'],
                       protocol_config_sha256=cfg['protocol_config_sha256'],
                       corpus_sha256=cfg['corpus_sha256'], mc_samples=128)
        protocol.frozen(folder / 'config.json', run_cfg)
        ch = sha(folder / 'config.json')
        rows = []
        for source in corpus:
            path = block_path(arm, source)
            if path.exists():
                row = read(path); evaluator.verify_row(row, source, pcfg, ch)
                rows.append(row)
        if len(rows) == len(corpus):
            progress(arm, 'aggregating', completed=551, total=551, gpu=visible)
        else:
            progress(arm, 'loading_model', completed=len(rows), total=551, gpu=visible)
            from experiments.projection_capacity_allocation_65.run import load_dense
            from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
            from experiments.wanda_failure_characterization.run_failure_map import model_sha
            model, mapping = load_dense()
            manifest = read(manifest_path(arm))
            if list(mapping) != [e['name'] for e in manifest['entries']]:
                raise RuntimeError('Projection order mismatch')
            for entry in manifest['entries']:
                selected = entry['selected_mask']
                if sha(selected['path']) != selected['file_sha256']:
                    raise RuntimeError('Packed mask changed: ' + entry['name'])
            if apply_manifest(model, mapping, manifest) != TARGET:
                raise RuntimeError('Physical prune count mismatch')
            actual = model_sha(model)
            if actual != run_cfg['sparse_model_sha256']:
                raise RuntimeError('Physical sparse-model SHA mismatch')
            progress(arm, 'evaluating_validation', completed=len(rows), total=551, gpu=visible)
            for i, source in enumerate(corpus):
                path = block_path(arm, source)
                if path.exists():
                    continue
                def tick(current):
                    if current % 64 == 0:
                        progress(arm, 'evaluating_validation', completed=i, total=551,
                                 draw=current, gpu=visible)
                row = evaluator.seal(dict(evaluator.evaluate_block(model, source, pcfg, tick),
                                          config_sha256=ch))
                evaluator.verify_row(row, source, pcfg, ch)
                protocol.frozen(path, row)
                progress(arm, 'evaluating_validation', completed=i+1, total=551, gpu=visible)
        ordered = []
        for source in corpus:
            row = read(block_path(arm, source))
            evaluator.verify_row(row, source, pcfg, ch)
            ordered.append(row)
        if len({r['block_id'] for r in ordered}) != 551:
            raise RuntimeError('Duplicate block coverage')
        result = dict(status='complete', arm=arm, split='validation', config_sha256=ch,
                      sparse_model_sha256=run_cfg['sparse_model_sha256'],
                      summary=summarize(ordered),
                      evaluation_seconds=sum(r['evaluation_seconds'] for r in ordered),
                      block_checkpoint_digests={r['block_id']:r['row_sha256'] for r in ordered})
        protocol.frozen(folder / 'results.json', result)
        progress(arm, 'complete', completed=551, total=551, gpu=visible, summary=result['summary'])


def report():
    validate()
    cfg = read(ROOT / 'config.json')
    pcfg = protocol.validate()
    corpus = read(protocol.ROOT / 'corpus_manifest.json')['splits']['validation']
    out = {}
    for arm in ARMS:
        path = ROOT / arm / 'results.json'
        if not path.exists():
            raise RuntimeError('Arm incomplete: ' + arm)
        result = read(path)
        if result['status'] != 'complete' or result['sparse_model_sha256'] != cfg['candidates'][arm]['sparse_model_sha256']:
            raise RuntimeError('Result identity changed: ' + arm)
        ch = sha(ROOT / arm / 'config.json')
        rows = [read(block_path(arm, s)) for s in corpus]
        for row, source in zip(rows, corpus, strict=True):
            evaluator.verify_row(row, source, pcfg, ch)
        if result['summary'] != summarize(rows) or result['block_checkpoint_digests'] != {r['block_id']:r['row_sha256'] for r in rows}:
            raise RuntimeError('Summary/checkpoint mismatch: ' + arm)
        out[arm] = result['summary']
    write(ROOT / 'report.json', dict(status='complete', config_sha256=sha(ROOT / 'config.json'),
                                     split='validation', arms=out,
                                     note='Exploratory after GSM8K; same MC draws, filtered WikiText validation; not autoregressive PPL'))
    return out


def status():
    state = read(ROOT / 'controller.json') if (ROOT / 'controller.json').exists() else {}
    print('WikiText exact-k MC128 |', state.get('status', 'prepared'))
    for arm in ARMS:
        p = ROOT / arm / 'progress.json'
        x = read(p) if p.exists() else {}
        summary = x.get('summary', {})
        score = f"NELBO {summary['token_nelbo']:.6f}, exp {summary['ppl_upper_bound_estimate']:.4f}" if summary else ''
        print(f"{arm:12s} {x.get('stage','pending'):23s} {x.get('completed',0):3d}/551 GPU{x.get('gpu','-')} {score}")
    if state.get('error'):
        print('Error:', state['error'])


def controller(gpus):
    if not os.environ.get('TMUX'):
        raise RuntimeError('Controller must run inside tmux')
    if not gpus or len(gpus) != len(set(gpus)) or not all(x.isdigit() for x in gpus):
        raise ValueError('Distinct GPU indices required')
    validate()
    for arm in ARMS:
        if (ROOT / arm / 'results.json').exists():
            result = read(ROOT / arm / 'results.json')
            if result.get('status') != 'complete':
                raise RuntimeError('Bad completed result: ' + arm)
    with (ROOT / 'controller.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = [a for a in ARMS if not (ROOT / a / 'results.json').exists()]
        active = {}
        stopping = [False]
        def stop(signum, frame):
            stopping[0] = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        write(ROOT / 'controller.json', dict(status='running', pid=os.getpid(), gpus=gpus,
                                             pending=queue, active=[], started=time.time()))
        try:
            while queue or active:
                if stopping[0]:
                    raise RuntimeError('Controller stopped by signal')
                for gpu in gpus:
                    if gpu in active or not queue:
                        continue
                    arm = queue.pop(0)
                    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
                    log = ROOT / arm / 'worker.log'; log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('a') as output:
                        proc = subprocess.Popen([sys.executable, '-B', '-u', '-m', MODULE, 'worker', '--arm', arm],
                                                env=env, stdout=output, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                    active[gpu] = (arm, proc)
                write(ROOT / 'controller.json', dict(status='running', pid=os.getpid(), gpus=gpus,
                                                     pending=queue, active={g:a for g,(a,_) in active.items()},
                                                     updated=time.time()))
                time.sleep(3)
                for gpu, (arm, proc) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    del active[gpu]
                    if code != 0 or not (ROOT / arm / 'results.json').exists():
                        raise RuntimeError(f'Worker {arm} on GPU{gpu} failed (exit {code}); see worker.log')
            report()
            write(ROOT / 'controller.json', dict(status='complete', ended=time.time(), gpus=gpus))
        except BaseException as exc:
            for arm, proc in active.values():
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for arm, proc in active.values():
                proc.wait(timeout=30)
            write(ROOT / 'controller.json', dict(status='failed', error=repr(exc), ended=time.time(), gpus=gpus))
            raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'validate', 'status', 'report'):
        sub.add_parser(name)
    for name in ('worker', 'launch'):
        p = sub.add_parser(name)
        p.add_argument('--arm' if name == 'worker' else '--gpus', required=True)
    args = parser.parse_args()
    if args.command != 'worker':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.command == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif args.command == 'validate':
        print('valid', sha(ROOT / 'config.json'))
        validate()
    elif args.command == 'status':
        status()
    elif args.command == 'report':
        print(json.dumps(report(), indent=2))
    elif args.command == 'worker':
        worker(args.arm)
    else:
        controller(args.gpus.split(','))


if __name__ == '__main__':
    main()
