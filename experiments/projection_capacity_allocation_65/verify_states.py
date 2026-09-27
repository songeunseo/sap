"""Verify frozen corruption states and corpus interval disjointness."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAL = Path('experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json')
HELD = Path('experiments/wanda_failure_characterization/heldout_state_manifest.json')
EXPECTED = ['1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df',
            'bb2cdaa6985a6ae6f75b7623a58653dcc80c73b7240cddbea551d62e7bab2b5e']
KEYS = ('timestep_index', 'timestep', 'sequence_index', 'mask_seed', 'p_mask',
        'clean_ids', 'noisy_ids', 'mask')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch
    from lib.dlm_gradient_sensitivity import make_masked_state

    docs = [json.loads(p.read_text()) for p in (CAL, HELD)]
    result = {'status': 'checking', 'splits': [], 'disjoint': False}
    try:
        model = docs[0]['model']
        assert model == docs[1]['model']
        assert model['revision'] == '0f2787f2d87eac5eed8a087d5ecd24277e6255b2'
        tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], trust_remote_code=True)
        corpus = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
        tokens = tokenizer(' '.join(corpus['text']), return_tensors='pt').input_ids[0]
        intervals = []
        for split, (path, doc, expected) in enumerate(zip((CAL, HELD), docs, EXPECTED)):
            raw = {'version': 1, 'states': [{k: s[k] for k in KEYS} for s in doc['states']]}
            digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            assert digest == doc['historical_state_sha256'] == expected
            assert len(doc['states']) == (80 if split == 0 else 40)
            times = [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95] if split == 0 else [.1, .3, .5, .7, .9]
            assert sorted({s['timestep'] for s in doc['states']}) == times
            clean = {s['sequence_index']: s['clean_ids'][0] for s in doc['states']}
            assert len(clean) == 8
            rng = random.Random(doc['global_random_seed'])
            spans = []
            for index in sorted(clean):
                start = rng.randint(0, len(tokens) - 256 - 1)
                assert tokens[start:start + 256].tolist() == clean[index], 'corpus span reconstruction mismatch'
                spans.append({'sequence_index': index, 'start': start, 'end_exclusive': start + 256})
            for state in doc['states']:
                assert state['clean_ids'][0] == clean[state['sequence_index']]
                noisy, mask, p = make_masked_state(torch.tensor(state['clean_ids']), state['timestep'],
                                                  doc['mask_id'], state['mask_seed'], eps=.001)
                assert noisy.tolist() == state['noisy_ids'] and mask.tolist() == state['mask']
                assert abs(p - state['p_mask']) < 1e-12
            intervals.append(spans)
            result['splits'].append({'path': str(path), 'file_sha256': sha(path), 'state_sha256': digest,
                                     'states': len(doc['states']), 'timesteps': times, 'spans': spans,
                                     'all_corruption_masks_regenerated_exactly': True})
        overlaps = [{'calibration': a, 'heldout': b} for a in intervals[0] for b in intervals[1]
                    if max(a['start'], b['start']) < min(a['end_exclusive'], b['end_exclusive'])]
        result.update(corpus_token_count=len(tokens), overlaps=overlaps, disjoint=not overlaps)
        assert not overlaps, 'allocation and held-out corpus intervals overlap'
        result['status'] = 'verified'
    except Exception as exc:
        result.update(status='STOP', error=str(exc), error_type=type(exc).__name__)
        write_json(ROOT / 'state_verification.json', result)
        raise
    write_json(ROOT / 'state_verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    verify()
