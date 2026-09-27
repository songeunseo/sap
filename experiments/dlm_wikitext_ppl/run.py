"""Frozen WikiText validation/test protocol and one timing-only Uniform65 pilot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.dlm_wikitext_ppl.core import (
    articles, block_seed, digest, mask_digest, masks, normalized_nelbo, summarize,
)
from experiments.dlm_dual_role_allocation.io import (
    atomic_write_json as write, file_sha256 as sha, MODEL_ID, MODEL_REVISION,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MANIFEST = REPO / 'experiments/dlm_allocation_sequential65/uniform/mask_manifest.json'
PRIOR = [REPO / x for x in (
    'experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json',
    'experiments/wanda_failure_characterization/heldout_state_manifest.json',
    'experiments/projection_capacity_followup_65/new_heldout_state_manifest.json',
)]


def read(path):
    return json.loads(Path(path).read_text())


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError('frozen artifact mismatch: ' + str(path))
    else:
        write(path, value)


def event(stage, **kw):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **kw)
    write(ROOT / 'progress.json', value)
    print(json.dumps(value), flush=True)


def prepare():
    if (ROOT / 'config.json').exists():
        return validate()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    event('preparing')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                             trust_remote_code=True, local_files_only=True)
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
    if mask_id != 126336:
        raise RuntimeError('unexpected mask token')
    protected = set()
    for path in PRIOR:
        for state in read(path)['states']:
            ids = state['clean_ids'][0]
            protected.update(tuple(ids[i:i+32]) for i in range(len(ids)-31))
    rows, rejected, corpus_digests = {}, [], {}
    seen_articles, seen_chunks, seen_ngrams = set(), set(), set()
    for split in ('validation', 'test'):
        data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
        lines = list(data['text']); corpus_digests[split] = digest(lines)
        output, split_ngrams = [], set()
        for article_index, article in enumerate(articles(lines)):
            article_sha = digest(article['text'])
            if article_sha in seen_articles:
                rejected.append(dict(split=split, article=article_index, reason='duplicate article across splits'))
                continue
            ids = tokenizer.encode(article['text'], add_special_tokens=False)
            if mask_id in ids:
                raise RuntimeError('clean corpus contains MASK token')
            for start in range(0, len(ids), 512):
                chunk = ids[start:start+512]
                if not chunk:
                    continue
                ngrams = {tuple(chunk[i:i+32]) for i in range(len(chunk)-31)}
                if ngrams & protected or ngrams & seen_ngrams or digest(chunk) in seen_chunks:
                    rejected.append(dict(split=split, article=article_index, start=start,
                                         reason='prior-state/cross-split token overlap'))
                    continue
                block_id = f'{split}:article{article_index}:offset{start}'
                seed = block_seed(2025, block_id)
                output.append(dict(block_id=block_id, article_index=article_index,
                                   article_sha256=article_sha, title=article['title'], start=start,
                                   clean_ids=chunk, mask_seed=seed,
                                   mask_sha256=mask_digest(len(chunk), 128, seed)))
                split_ngrams.update(ngrams)
            seen_articles.add(article_sha)
        if not output:
            raise RuntimeError('empty data split')
        seen_chunks.update(digest(r['clean_ids']) for r in output)
        seen_ngrams.update(split_ngrams)
        rows[split] = output
        event('prepared_split', split=split, blocks=len(output), tokens=sum(len(r['clean_ids']) for r in output))
    selected = []
    for row in rows['validation']:
        if len(row['clean_ids']) == 512 and row['article_index'] not in {r['article_index'] for r in selected}:
            selected.append(row)
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise RuntimeError('eight distinct full-length timing articles unavailable')
    payload = dict(splits=rows, pilot_block_ids=[r['block_id'] for r in selected])
    frozen(ROOT / 'corpus_manifest.json', payload)
    frozen(ROOT / 'state_verification.json', dict(
        status='verified', source_split={'calibration_and_prior_dlm_states':'train',
                                        'development':'validation', 'final':'test'},
        rule='exclude duplicate article texts/chunks across splits and any chunk sharing a contiguous 32-token substring with prior persisted DLM states or the other split',
        corpus_sha256=corpus_digests, rejected=rejected,
        prior_manifests={str(p):sha(p) for p in PRIOR},
        mask_draws='CPU exact-k RNG, per-block seed, regenerated boolean masks verified by SHA256'))
    sources = [Path(__file__), ROOT/'core.py', ROOT/'test_core.py',
               REPO/'get_log_likelihood.py', MANIFEST, *PRIOR,
               ROOT/'corpus_manifest.json', ROOT/'state_verification.json',
               REPO/'experiments/projection_capacity_allocation_65/run.py',
               REPO/'experiments/projection_capacity_followup_65/run_heldout.py']
    config = dict(model=dict(id=MODEL_ID, revision=MODEL_REVISION),
                  dataset='Salesforce/wikitext', dataset_config='wikitext-2-raw-v1',
                  development_split='validation', final_split='test',
                  sequence_length=512, conditioning='unconditional within each article chunk',
                  special_tokens=False, remainder='include nonempty short article tails; parameter-free token-weighted aggregation',
                  mc_samples=128, batch_size=1, seed=2025, mask_id=mask_id, cfg_scale=0,
                  estimator='LLaDA Eq.6 exact-k uniform without replacement; token NELBO then exp',
                  dtype='BF16 model, FP32 masked CE', torch_version=torch.__version__,
                  tokenizer_sha256=digest(tokenizer.backend_tokenizer.to_str()),
                  pilot='eight distinct validation articles, one 512-token chunk each; timing only, not selection or final evaluation',
                  pruning='reuse exact completed sequential Uniform-Wanda65 mask manifest; no new local selector',
                  sources={str(p):sha(p) for p in sources})
    frozen(ROOT/'config.json', config)
    event('prepared', validation_blocks=len(rows['validation']), test_blocks=len(rows['test']))
    return config


def validate():
    cfg = read(ROOT/'config.json')
    for path, value in cfg['sources'].items():
        if sha(path) != value:
            raise RuntimeError('frozen input changed: '+path)
    if cfg['torch_version'] != torch.__version__:
        raise RuntimeError('mask RNG torch version changed')
    return cfg


@torch.inference_mode()
def pilot():
    cfg = validate()
    if (ROOT/'pilot_results.json').exists():
        result = read(ROOT/'pilot_results.json')
        if result['config_sha256'] != sha(ROOT/'config.json'):
            raise RuntimeError('pilot result config mismatch')
        event('complete', result=result['summary'])
        return
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
    started = time.monotonic(); event('loading_dense')
    model, mapping = load_dense()
    manifest = read(MANIFEST)
    if manifest['pruned'] != 4536008704:
        raise RuntimeError('wrong Uniform65 budget')
    apply_manifest(model, mapping, manifest)
    torch.cuda.synchronize()
    preparation_seconds = time.monotonic()-started
    corpus = read(ROOT/'corpus_manifest.json')
    by_id = {r['block_id']:r for r in corpus['splits']['validation']}
    device = next(model.parameters()).device
    rows = []
    for block_id in corpus['pilot_block_ids']:
        source = by_id[block_id]
        path = ROOT/'pilot_blocks'/f"{digest(block_id)[:20]}.json"
        if path.exists():
            row = read(path)
            if row['config_sha256'] != sha(ROOT/'config.json') or row['block_id'] != block_id:
                raise RuntimeError('checkpoint mismatch')
            rows.append(row); continue
        length = len(source['clean_ids'])
        if mask_digest(length, cfg['mc_samples'], source['mask_seed']) != source['mask_sha256']:
            raise RuntimeError('mask draws changed')
        clean = torch.tensor(source['clean_ids'], device=device)
        values = []
        torch.cuda.synchronize(); block_started = time.monotonic()
        for sample_index, cpu_mask in enumerate(masks(length, cfg['mc_samples'], source['mask_seed'])):
            mask = cpu_mask.to(device)
            noisy = clean.masked_fill(mask, cfg['mask_id']).unsqueeze(0)
            logits = model(noisy).logits[0]
            values.append(float(normalized_nelbo(logits, clean, mask)))
            del logits
            if (sample_index+1)%16 == 0:
                event('evaluating_pilot', completed=len(rows)*cfg['mc_samples']+sample_index+1,
                      total=8*cfg['mc_samples'], block_id=block_id)
        torch.cuda.synchronize()
        row = dict(block_id=block_id, tokens=length, mc_samples=len(values),
                   token_nelbo=statistics.mean(values), mc_variance=statistics.variance(values),
                   sample_token_nelbo=values, evaluation_seconds=time.monotonic()-block_started,
                   config_sha256=sha(ROOT/'config.json'), mask_sha256=source['mask_sha256'])
        frozen(path,row); rows.append(row)
    evaluation_seconds = sum(r['evaluation_seconds'] for r in rows)
    seconds_per_forward = evaluation_seconds/(8*cfg['mc_samples'])
    eta = {split:dict(blocks=len(blocks), forward_passes=len(blocks)*cfg['mc_samples'],
                     evaluation_seconds_upper_length_estimate=len(blocks)*cfg['mc_samples']*seconds_per_forward)
           for split,blocks in corpus['splits'].items()}
    result = dict(status='complete', method='sequential Uniform-Wanda65',
                  scope='timing pilot only; does not replace full development/final scores',
                  summary=summarize(rows), preparation_seconds=preparation_seconds,
                  evaluation_seconds=evaluation_seconds, seconds_per_forward=seconds_per_forward,
                  projected_full_split_times=eta,
                  runtime_caveat='512-token pilot extrapolation; short tails may run faster; excludes new mask construction/proxy recomputation and GPU contention',
                  config_sha256=sha(ROOT/'config.json'), manifest_sha256=sha(MANIFEST),
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), rows=rows)
    validate(); frozen(ROOT/'pilot_results.json',result)
    event('complete', result=result['summary'], projected_full_split_times=eta)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','pilot'])
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            prepare() if args.command=='prepare' else pilot()
        except Exception as exc:
            event('failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    main()
