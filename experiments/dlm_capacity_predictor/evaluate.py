#!/usr/bin/env python3
"""Full jointly sparse confirmatory gate, then optional frozen GSM8K evaluation."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_capacity_predictor.core import (
    ROOT, SOURCE, FOLLOWUP, TARGET, sha, write_json, passes_gate, state_indices,
    dependency_complete)
from experiments.dlm_capacity_predictor.audit_existing import run_audit, strict_exact_match
from experiments.dlm_loss_aggregation.run import historical_state_digest
from experiments.projection_capacity_allocation_65.run import (
    load_dense, DENSE_SHA, diagnostics, summarize_states)
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest, selected_mask
from experiments.projection_capacity_followup_65.core import paired_comparison, paired_binary_comparison
from experiments.wanda_failure_characterization.run_failure_map import model_sha, state_tensors


def _load_json(path):
    return json.loads(Path(path).read_text())


def _full_downstream(document):
    # The frozen allocation-era artifact predates the explicit status field.
    if 'status' in document and document.get('status') != 'complete':
        return False
    limits = set()
    for row in document.get('evaluations', []):
        if 'limit' in row:
            limits.add(row['limit'])
        else:
            limits.update(item.get('limit') for key, item in row.items()
                          if key in ('uniform', 'capacity') and isinstance(item, dict))
    return {100, 1319}.issubset(limits)


def require_historical_complete():
    allocation = _load_json(SOURCE/'downstream.json')
    followup = _load_json(FOLLOWUP/'downstream.json')
    if not _full_downstream(allocation) or not dependency_complete(followup):
        raise RuntimeError('historical full downstream results are incomplete')
    audit = run_audit(Path(__file__).resolve().parents[2])
    if audit.get('status') != 'complete':
        raise RuntimeError('historical full downstream audit is not complete')
    return audit


def validate_heldout_result(result, method, fingerprint, expected_model_sha256, expected_states):
    if result.get('status') != 'complete' or result.get('method') != method:
        raise RuntimeError('heldout resume method/status mismatch')
    if result.get('fingerprint') != fingerprint:
        raise RuntimeError('heldout resume fingerprint mismatch')
    if result.get('model_sha256') != expected_model_sha256:
        raise RuntimeError('heldout resume model mismatch')
    if result.get('pruned') != TARGET or result.get('weights') != 6_979_321_856:
        raise RuntimeError('heldout resume budget mismatch')
    rows = result.get('per_state', [])
    expected = [(i, int(state['sequence_index']), round(float(state['timestep']), 8))
                for i, state in enumerate(expected_states)]
    actual = [(r.get('state_index'), r.get('sequence_index'), round(float(r.get('timestep', -1)), 8))
              for r in rows]
    if actual != expected:
        raise RuntimeError('heldout resume state identity/order mismatch')
    values = np.asarray([r.get('mean_kl') for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError('heldout resume requires finite mean_kl')
    if not np.isclose(float(result.get('summary', {}).get('mean_kl', np.nan)), values.mean(),
                      rtol=0, atol=1e-12):
        raise RuntimeError('heldout resume summary mismatch')


def load_verified_heldout(path, receipt, method, fingerprint, expected_states):
    if (receipt.get('fingerprint') != fingerprint
            or receipt.get('result_sha256') != sha(path)):
        raise RuntimeError('heldout resume receipt mismatch')
    result = _load_json(path)
    validate_heldout_result(result, method, fingerprint, receipt.get('model_sha256'), expected_states)
    return result


def load_verified_aggregate(path, receipt, fingerprint):
    if (receipt.get('fingerprint') != fingerprint
            or receipt.get('aggregate_sha256') != sha(path)):
        raise RuntimeError('heldout aggregate receipt mismatch')
    aggregate = _load_json(path)
    if aggregate.get('status') != 'complete' or aggregate.get('fingerprint') != fingerprint:
        raise RuntimeError('heldout aggregate provenance mismatch')
    return aggregate


def validate_prediction_rows(rows, limit, protocol_sha):
    if len(rows) != limit:
        raise RuntimeError('incorrect prediction count')
    for i, row in enumerate(rows):
        required = ('doc_hash', 'prompt_hash', 'target_hash', 'reference_answer', 'extracted_answer')
        if any(row.get(key) is None for key in required):
            raise RuntimeError('GSM8K row identity/evaluator fields missing')
        if row.get('example_id') != i or row.get('evaluation_config_hash') != protocol_sha:
            raise RuntimeError('GSM8K rows/protocol mismatch')
        if strict_exact_match(row.get('extracted_answer'), row.get('reference_answer')) != row.get('correct'):
            raise RuntimeError('saved strict exact-match disagrees with recomputation')


def preflight_mask_payloads(paths, manifests, candidate_names):
    event('mask_preflight_started', methods=list(manifests))
    fingerprint = {name: {'manifest': sha(paths[name]), 'payloads': [
        sha(entry['selected_mask']['path']) for entry in manifest['entries']]}
        for name, manifest in manifests.items()}
    receipt_path = ROOT/'mask_preflight_receipt.json'
    if receipt_path.exists():
        cached = _load_json(receipt_path)
        if cached.get('status') == 'verified' and cached.get('fingerprint') == fingerprint:
            event('mask_preflight_reused')
            return
    uniform = manifests['uniform']['entries']
    for method, manifest in manifests.items():
        for index, (entry, reference) in enumerate(zip(manifest['entries'], uniform, strict=True)):
            mask = selected_mask(entry, 'cpu')
            if tuple(mask.shape) != tuple(entry['shape']):
                raise RuntimeError(f'{method} actual mask shape mismatch')
            if int(mask.sum()) != int(entry['selected_mask']['pruned']):
                raise RuntimeError(f'{method} actual mask count mismatch')
            if method in candidate_names:
                base = selected_mask(reference, 'cpu')
                if int(torch.logical_xor(mask, base).sum()) != abs(int(mask.sum()) - int(base.sum())):
                    raise RuntimeError(f'{method} masks are not nested')
            if (index + 1) % 32 == 0 or index + 1 == len(uniform):
                event('mask_preflight_progress', method=method, checked=index+1, total=len(uniform))
    write_json(receipt_path, {'status': 'verified', 'fingerprint': fingerprint})
    event('mask_preflight_complete')


def validate_inputs():
    candidates = json.loads((ROOT/'candidates.json').read_text())
    verification = json.loads((ROOT/'state_verification.json').read_text())
    states = json.loads((ROOT/'heldout_state_manifest.json').read_text())
    if candidates['status'] != 'frozen' or candidates['config_sha256'] != sha(ROOT/'config.json'):
        raise RuntimeError('candidate decisions not frozen')
    if len(candidates['candidates']) > 2:
        raise RuntimeError('at most one candidate per path')
    if not verification['all_four_splits_disjoint'] or verification['status'] != 'verified':
        raise RuntimeError('unverified heldout states')
    if (sha(ROOT/'heldout_state_manifest.json') != verification['manifest_sha256']
            or historical_state_digest(states) != verification['state_sha256']
            or sha(ROOT/'candidates.json') != verification['candidates_sha256']):
        raise RuntimeError('state/candidate freeze changed')
    for path, digest in verification['prior_manifest_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('prior states changed')
    state_indices([(s['sequence_index'], s['timestep']) for s in states['states']],
                  range(24, 32), [.1, .3, .5, .7, .9])
    paths = {'uniform': SOURCE/'uniform65_mask_manifest.json',
             'eis_type': FOLLOWUP/'eis_type65_mask_manifest.json'}
    for candidate in candidates['candidates']:
        if sha(candidate['mask_manifest']) != candidate['mask_manifest_sha256']:
            raise RuntimeError('candidate mask changed')
        paths[candidate['method']] = Path(candidate['mask_manifest'])
    manifests = {name: json.loads(path.read_text()) for name, path in paths.items()}
    layout = [(e['name'], e['shape']) for e in manifests['uniform']['entries']]
    for method, manifest in manifests.items():
        if (len(manifest['entries']) != 224
                or [(e['name'], e['shape']) for e in manifest['entries']] != layout
                or manifest['pruned'] != TARGET
                or sum(e['selected_mask']['pruned'] for e in manifest['entries']) != TARGET):
            raise RuntimeError(f'{method} layout/budget mismatch')
    preflight_mask_payloads(paths, manifests, [c['method'] for c in candidates['candidates']])
    return states, paths, manifests, candidates


def event(kind, **kwargs):
    print(json.dumps(dict(event=kind, **kwargs)), flush=True)


@torch.inference_mode()
def evaluate_heldout():
    require_historical_complete()
    states, paths, manifests, candidates = validate_inputs()
    started = time.monotonic()
    candidate_names = [c['method'] for c in candidates['candidates']]
    if not candidate_names:
        write_json(ROOT/'heldout_dlm_results.json', dict(status='no_eligible_candidates', decisions={}))
        return
    aggregate_path = ROOT/'heldout_dlm_results.json'
    aggregate_receipt_path = aggregate_path.with_suffix('.receipt.json')
    aggregate_fingerprint = dict(config_sha256=sha(ROOT/'config.json'),
        candidates_sha256=sha(ROOT/'candidates.json'), state_sha256=states['historical_state_sha256'])
    if aggregate_path.exists():
        aggregate = _load_json(aggregate_path)
        if aggregate.get('status') == 'complete':
            if not aggregate_receipt_path.exists():
                raise RuntimeError('completed heldout aggregate has no receipt')
            aggregate = load_verified_aggregate(
                aggregate_path, _load_json(aggregate_receipt_path), aggregate_fingerprint)
            for method in manifests:
                destination = ROOT/f'heldout_{method}.json'
                receipt_path = destination.with_suffix('.receipt.json')
                if not destination.exists() or not receipt_path.exists():
                    raise RuntimeError(f'completed aggregate missing heldout receipt: {method}')
                fingerprint = dict(mask_manifest_sha256=sha(paths[method]),
                    state_sha256=states['historical_state_sha256'], config_sha256=sha(ROOT/'config.json'))
                verified = load_verified_heldout(destination, _load_json(receipt_path), method,
                                                 fingerprint, states['states'])
                if aggregate.get('methods', {}).get(method) != verified:
                    raise RuntimeError(f'aggregate embedded result mismatch: {method}')
            return
    model, mapping = load_dense()
    torch.cuda.reset_peak_memory_stats()
    device = next(model.parameters()).device
    references, sham_max = [], 0.
    for i, state in enumerate(states['states']):
        noisy, _, mask = state_tensors(state, device)
        dense = model(noisy).logits
        sham = model(noisy).logits
        diff = float((dense-sham).abs().max().item())
        if diff != 0:
            raise RuntimeError('same-path dense sham mismatch')
        sham_max = max(sham_max, diff)
        references.append(dense[0, mask[0]].cpu())
        del dense, sham
        event('dense_reference', state=i+1, total=40)
    backup = {name: module.weight.detach().cpu().clone() for name, module in mapping.items()}
    results, times = {}, {}
    for method, manifest in manifests.items():
        fingerprint = dict(mask_manifest_sha256=sha(paths[method]),
                           state_sha256=states['historical_state_sha256'],
                           config_sha256=sha(ROOT/'config.json'))
        destination = ROOT/f'heldout_{method}.json'
        receipt_path = destination.with_suffix('.receipt.json')
        if destination.exists():
            if not receipt_path.exists():
                raise RuntimeError(f'heldout result has no provenance receipt: {method}')
            results[method] = load_verified_heldout(destination, _load_json(receipt_path), method,
                                                    fingerprint, states['states'])
            continue
        for name, module in mapping.items():
            module.weight.copy_(backup[name])
        if model_sha(model) != DENSE_SHA:
            raise RuntimeError('dense restoration failed')
        mask_start = time.monotonic()
        # Verify actual XOR and nested-prefix invariant, not just nominal sparsity differences.
        xor_count = 0
        if method in candidate_names:
            for entry, ref in zip(manifest['entries'], manifests['uniform']['entries']):
                a, b = selected_mask(entry, 'cpu'), selected_mask(ref, 'cpu')
                xor = int(torch.logical_xor(a, b).sum().item())
                if xor != abs(entry['selected_mask']['pruned']-ref['selected_mask']['pruned']):
                    raise RuntimeError('historical candidate masks are not nested')
                xor_count += xor
                del a, b
        applied = apply_manifest(model, mapping, manifest)
        sparse_sha = model_sha(model)
        mask_seconds = time.monotonic()-mask_start
        forward_start = time.monotonic()
        rows, token_kl = [], []
        for i, state in enumerate(states['states']):
            noisy, clean, mask = state_tensors(state, device)
            sparse = model(noisy).logits[0, mask[0]]
            metrics, kl = diagnostics(sparse, references[i].to(device),
                                      clean[0, mask[0]], state['p_mask'])
            metrics.update(state_index=i, sequence_index=state['sequence_index'], timestep=state['timestep'])
            rows.append(metrics)
            token_kl.append(kl)
            event('heldout_state', method=method, state=i+1, mean_kl=metrics['mean_kl'])
        torch.cuda.synchronize()
        times[method] = dict(mask_validation_application_hash_seconds=mask_seconds,
                             heldout_forward_metric_seconds=time.monotonic()-forward_start)
        if model_sha(model) != sparse_sha:
            raise RuntimeError('evaluation altered model')
        result = dict(status='complete', method=method, fingerprint=fingerprint,
                      model_sha256=sparse_sha, per_state=rows, summary=summarize_states(rows, token_kl),
                      pruned=applied, weights=manifest['weights'], global_sparsity=applied/manifest['weights'],
                      mask_xor_vs_uniform=xor_count if method in candidate_names else None,
                      timings=times[method])
        validate_heldout_result(result, method, fingerprint, sparse_sha, states['states'])
        write_json(destination, result, frozen=True)
        write_json(receipt_path, dict(fingerprint=fingerprint, result_sha256=sha(destination),
                                      model_sha256=sparse_sha), frozen=True)
        results[method] = result
    comparisons, decisions = {}, {}
    for method in candidate_names:
        comparisons[method] = {reference: paired_comparison(results[reference]['per_state'],
                                                          results[method]['per_state'])
                               for reference in ('uniform', 'eis_type')}
        decisions[method] = 'PASS' if passes_gate(comparisons[method]['uniform']) else 'FAIL'
    write_json(aggregate_path, dict(status='complete', methods=results,
        comparisons=comparisons, decisions=decisions, dense_sham_max_abs=sham_max,
        fingerprint=aggregate_fingerprint,
        config_sha256=sha(ROOT/'config.json'), candidates_sha256=sha(ROOT/'candidates.json'),
        state_sha256=states['historical_state_sha256'], wall_seconds=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        statistical_scope='each candidate has its preregistered gate; no selection by best heldout result'))
    write_json(aggregate_receipt_path, dict(fingerprint=aggregate_fingerprint,
        aggregate_sha256=sha(aggregate_path)), frozen=True)
    event('heldout_complete', decisions=decisions)
    del model, backup, references
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_downstream():
    from transformers import AutoTokenizer
    from experiments.dlm_loss_aggregation.exp002.run import (
        _assert_same_examples, _evaluate_gsm8k, _evaluation_config_hash,
        _read_jsonl, _write_jsonl, load_config)
    require_historical_complete()
    _, paths, manifests, _ = validate_inputs()
    gate = json.loads((ROOT/'heldout_dlm_results.json').read_text())
    if gate['status'] == 'no_eligible_candidates':
        return
    if gate['status'] != 'complete' or gate['candidates_sha256'] != sha(ROOT/'candidates.json'):
        raise RuntimeError('full-model gate incomplete/changed')
    aggregate_fingerprint = dict(config_sha256=sha(ROOT/'config.json'),
        candidates_sha256=sha(ROOT/'candidates.json'), state_sha256=gate.get('state_sha256'))
    gate = load_verified_aggregate(ROOT/'heldout_dlm_results.json',
        _load_json((ROOT/'heldout_dlm_results.json').with_suffix('.receipt.json')),
        aggregate_fingerprint)
    passed = [name for name, decision in gate['decisions'].items() if decision == 'PASS']
    config = load_config('experiments/dlm_loss_aggregation/exp002/config.yaml')
    protocol_sha, protocol = _evaluation_config_hash(config)
    historical = json.loads((SOURCE/'downstream.json').read_text())
    if protocol_sha != historical['protocol_hash']:
        raise RuntimeError('GSM8K protocol mismatch')
    dense_rows = _read_jsonl(Path('experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'))
    tokenizer = AutoTokenizer.from_pretrained(config['model']['id'], revision=config['model']['revision'],
                                             trust_remote_code=True) if passed else None
    outcomes = []
    for limit in (100, 1319):
        for method in passed:
            destination = ROOT/'gsm8k'/f'{method}_{limit}_predictions.jsonl'
            receipt = destination.with_suffix('.receipt.json')
            expected = dict(protocol_sha256=protocol_sha, mask_manifest_sha256=sha(paths[method]),
                            heldout_gate_sha256=sha(ROOT/'heldout_dlm_results.json'), limit=limit)
            if destination.exists():
                if not receipt.exists():
                    raise RuntimeError('prediction file exists without verified provenance receipt')
                saved = json.loads(receipt.read_text())
                if saved['fingerprint'] != expected or saved['predictions_sha256'] != sha(destination):
                    raise RuntimeError('GSM8K resume provenance mismatch')
                rows = _read_jsonl(destination)
                elapsed = saved['wall_seconds']
            else:
                model, mapping = load_dense()
                apply_manifest(model, mapping, manifests[method])
                sparse_sha = model_sha(model)
                started = time.monotonic()
                _, rows = _evaluate_gsm8k(model, tokenizer, config, f'{method}_dlm_capacity', limit, protocol_sha)
                elapsed = time.monotonic()-started
                if model_sha(model) != sparse_sha:
                    raise RuntimeError('model changed during GSM8K')
                _assert_same_examples(dense_rows[:limit], rows)
                if len(rows) != limit:
                    raise RuntimeError('incorrect prediction count')
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_jsonl(destination, rows)
                write_json(receipt, dict(fingerprint=expected, predictions_sha256=sha(destination),
                                         wall_seconds=elapsed, model_sha256=sparse_sha), frozen=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            _assert_same_examples(dense_rows[:limit], rows)
            validate_prediction_rows(rows, limit, protocol_sha)
            comparisons = {}
            for reference, directory in (('uniform', SOURCE), ('capacity', SOURCE),
                                          ('reconstruction', FOLLOWUP), ('eis_type', FOLLOWUP)):
                ref = _read_jsonl(directory/'gsm8k'/f'{reference}_{limit}_predictions.jsonl')
                _assert_same_examples(rows, ref)
                comparisons[reference] = paired_binary_comparison([r['correct'] for r in ref],
                                                                   [r['correct'] for r in rows])
            outcomes.append(dict(method=method, limit=limit, correct=sum(r['correct'] for r in rows),
                                 predictions_sha256=sha(destination), wall_seconds=elapsed,
                                 comparisons=comparisons))
            write_json(ROOT/'downstream.json', dict(status='running', evaluations=outcomes,
                                                   protocol_hash=protocol_sha, protocol=protocol))
            event('gsm8k_complete', method=method, limit=limit, correct=outcomes[-1]['correct'])
    write_json(ROOT/'downstream.json', dict(status='complete', evaluations=outcomes,
        passed_candidates=passed, protocol_hash=protocol_sha, protocol=protocol,
        mini_policy='sanity only; no candidate selected or retuned using mini results'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downstream', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        validate_inputs()
        return
    if args.downstream:
        evaluate_downstream()
    else:
        evaluate_heldout()


if __name__ == '__main__':
    main()
