"""Final, fixed geometry feasibility gate. Dense forward and directional JVP only."""
import hashlib
import inspect
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_loss_aggregation.run import _load_model, load_config, validate_config, historical_state_digest
from experiments.wanda_failure_characterization.run_failure_map import model_sha, state_tensors
from experiments.wanda_failure_characterization.run_state_swap import l31_sham_hidden, final_suffix
from experiments.wanda_failure_characterization.propagation_core import factorize_token_feature, matched_perturbation
from experiments.wanda_failure_characterization.geometry_core import rms_forward, rms_jvp, score_directions

ROOT = Path(__file__).parent
OUT = ROOT / 'geometry_kill_gate'
SHIFT = [1, 2, 4, 8, 16, 32, 64, 128]
FD_GRID = [.3, .1, .03, .01, .003, .001]
EPS = 1e-12


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tsha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def save_json(name, obj):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def audit_sources():
    verified = {}
    def check(file, expected):
        actual = sha(ROOT / file)
        if actual != expected:
            raise RuntimeError(f'Hash mismatch: {file}')
        verified[file] = actual
    load = lambda name: json.loads((ROOT / name).read_text())
    t = load('perturbation_transplant_manifest.json')
    check('transplant_deltas.pt', t['delta_artifact_sha256'])
    check('perturbation_transplant.pt', t['result_artifact_sha256'])
    swap = load('state_swap_manifest.json')
    check('state_swap_results.pt', swap['result_sha256'])
    factor = load('token_feature_factorization_manifest.json')
    check('token_feature_factorization.pt', factor['result_sha256'])
    row = load('token_row_pairing_manifest.json')
    check('token_row_pairing_permutations.pt', row['permutation_artifact_sha256'])
    for name, expected in load('token_row_pairing_run_manifest.json')['shard_hashes'].items():
        check('token_row_pairing_shards/' + name, expected)
    for name, field, expected in [('state_swap_mapping.json', 'mapping_sha256', swap['mapping_sha']),
                                  ('token_feature_donor_manifest.json', 'mapping_sha256', factor['mapping_sha256'])]:
        doc = load(name)
        stored = doc.pop(field)
        computed = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if stored != expected or computed != expected:
            raise RuntimeError('mapping digest mismatch: ' + name)
        verified[name] = sha(ROOT / name)
    held = load('heldout_state_manifest.json')
    if historical_state_digest(held) != t['heldout_sha']:
        raise RuntimeError('frozen state digest mismatch')
    verified['heldout_state_manifest.json'] = sha(ROOT / 'heldout_state_manifest.json')
    return verified, t['weight_sha_after']


def validate_jvp(model, hiddens, deltas, tau, held):
    norm = model.model.transformer.ln_f
    head = model.model.transformer.ff_out
    device = head.weight.device
    gamma = norm.weight.detach().to(device).float()
    scale = 1 / math.sqrt(4096) if model.model.config.scale_logits else 1.
    grid_results, forward_check = [], []
    # All vocabulary outputs for first 3 masked tokens of preselected receivers.
    for si in [0, 16, 32]:
        h = hiddens[si].to(device).float()
        d, _ = matched_perturbation(deltas[si].to(device), h, tau[si].to(device), EPS)
        pos = torch.tensor(held['states'][si]['mask'][0], device=device).nonzero().flatten()[:3]
        h, d = h[pos], d[pos]
        pieces = {eps: [] for eps in FD_GRID}
        analytic, auto = [], []
        for start in range(0, head.weight.shape[0], 8192):
            w = head.weight[start:start+8192].detach().float()
            def f(x):
                return F.linear(rms_forward(x, gamma, norm.eps), w) * scale
            v = F.linear(rms_jvp(h, d, gamma, norm.eps), w) * scale
            _, jvp = torch.func.jvp(f, (h,), (d,))
            analytic.append(v)
            auto.append(jvp)
            for eps in FD_GRID:
                pieces[eps].append((f(h + eps*d) - f(h - eps*d)) / (2*eps))
        v, auto = torch.cat(analytic, -1), torch.cat(auto, -1)
        forward_check.append({'state': si, 'relative_l2': ((v-auto).norm()/auto.norm()).item(),
                              'max_abs': (v-auto).abs().max().item()})
        for eps in FD_GRID:
            fd = torch.cat(pieces[eps], -1)
            grid_results.append({'state': si, 'epsilon': eps, 'relative_l2': ((fd-v).norm()/v.norm()).item(),
                                 'cosine': F.cosine_similarity(fd.flatten(), v.flatten(), dim=0).item(),
                                 'max_abs': (fd-v).abs().max().item()})
    # Selection uses only derivative agreement, never exact KL.
    selected = min(FD_GRID, key=lambda e: sum(row['relative_l2'] for row in grid_results if row['epsilon'] == e))
    good = [row for row in grid_results if row['epsilon'] == selected]
    passed = all(row['relative_l2'] < 1e-3 and row['cosine'] > .99999 for row in good)
    passed &= all(row['relative_l2'] < 1e-5 for row in forward_check)
    output = {'status': 'pass' if passed else 'failed', 'selected_epsilon': selected,
              'selection_rule': 'minimum mean FD-vs-JVP relative error on frozen subset; independent of KL',
              'finite_differences': grid_results, 'analytic_vs_torch_func_jvp': forward_check}
    save_json('jvp_validation.json', output)
    if not passed:
        raise RuntimeError('JVP numerical gate failed; stopping without geometric alternatives')


def main():
    started = time.time()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(exist_ok=True)
    verified, expected_weight = audit_sources()
    prereg = {'status': 'frozen_before_predictor_results', 'datasets': ['A_native_same_timestep', 'B_all_280_pairs', 'C_two_families_8_shifts'],
              'shifts': SHIFT, 'fd_grid': FD_GRID, 'fd_states': [0,16,32], 'fd_tokens': 'first three masked tokens',
              'gate_relative_l2': 1e-3, 'gate_cosine': .99999, 'predictors': ['B_hidden', 'B_logit', 'Q_geom'],
              'reduction': 'mean over receiver masked tokens; sum across features/vocabulary',
              'predictor_precision': 'FP32 continuous RMSNorm+head with exact stored BF16 parameter values; FP64 reductions',
              'bootstrap_seed': 20260905, 'bootstrap_replicates': 10000, 'ratio_epsilon': EPS,
              'verified_source_hashes': verified, 'post_experiment': 'no additional mechanism study; one authorized-later prototype A/B/C or stop D'}
    save_json('preregistered_manifest.json', prereg)
    held = json.loads((ROOT/'heldout_state_manifest.json').read_text())
    delta_art = torch.load(ROOT/'transplant_deltas.pt', map_location='cpu', weights_only=True)
    deltas = delta_art['delta31']
    if tsha(deltas) != delta_art['hashes'][1]:
        raise RuntimeError('delta31 raw tensor hash mismatch')
    prior = torch.load(ROOT/'perturbation_transplant.pt', map_location='cpu', weights_only=True)
    tau = prior['tau']
    config = load_config('experiments/dlm_loss_aggregation/config.yaml')
    validate_config(config)
    model, _ = _load_model(config)
    before = model_sha(model)
    if before != expected_weight:
        raise RuntimeError('model SHA differs from frozen experiment')
    norm, head = model.model.transformer.ln_f, model.model.transformer.ff_out
    if type(norm).__name__ != 'RMSLayerNorm' or model.model.config.weight_tying or norm.bias is not None or head.bias is not None:
        raise RuntimeError('unexpected readout architecture; do not assume another norm')
    save_json('readout_path.json', {'norm_type': type(norm).__name__, 'norm_epsilon': norm.eps,
                                  'norm_weight': list(norm.weight.shape), 'head_weight': list(head.weight.shape),
                                  'head_bias': False, 'weight_tying': False, 'scale_logits': model.model.config.scale_logits,
                                  'norm_source': inspect.getsource(type(norm)), 'model_source_sha256': sha('model/modeling_llada.py'),
                                  'weight_sha_before': before})
    dev = head.weight.device
    hidden_path = OUT/'dense_l31_residuals.pt'
    # Old artifacts have no persisted dense H31. Restore the original dense path only.
    hiddens, dense_checks = [], []
    with torch.no_grad():
        for si, state in enumerate(held['states']):
            noisy, _, mask = state_tensors(state, dev)
            captured = {}
            def capture(_, inp): captured['x'] = inp[0].detach()
            handle = model.model.transformer.blocks[31].register_forward_pre_hook(capture)
            model(noisy)
            handle.remove()
            h, orig_logits = l31_sham_hidden(model, captured['x'])
            repeat_logits, _ = final_suffix(model, h.expand(3,-1,-1).contiguous())
            error = (orig_logits[0].float()-repeat_logits[0].float()).abs().max().item()
            if error != 0:
                raise RuntimeError('restored dense final suffix does not match original path')
            hiddens.append(h.cpu())
            dense_checks.append(error)
        hiddens = torch.stack(hiddens)
        torch.save({'h31': hiddens, 'heldout_sha': held['historical_state_sha256'], 'weight_sha': before}, hidden_path)
    print('Dense H31 restored for all 40 states; same-path logits exact. Validating JVP.', flush=True)
    # Forward-mode is intentionally outside inference_mode (which disables dual tensors).
    with torch.no_grad(): validate_jvp(model, hiddens, deltas, tau, held)
    print('JVP gate passed. Computing exactly three predictors on frozen directions.', flush=True)
    swap = torch.load(ROOT/'state_swap_results.pt', map_location='cpu', weights_only=True)
    swap_map = json.loads((ROOT/'state_swap_mapping.json').read_text())
    factor = torch.load(ROOT/'token_feature_factorization.pt', map_location='cpu', weights_only=True)
    pair_map = json.loads((ROOT/'token_feature_donor_manifest.json').read_text())['pairs']
    perms = torch.load(ROOT/'token_row_pairing_permutations.pt', map_location='cpu', weights_only=True)
    factors = [factorize_token_feature(d, EPS) for d in deltas]
    records, predictor_rows, rounding_checks = [], [], []
    with torch.no_grad():
        for si, state in enumerate(held['states']):
            h = hiddens[si].to(dev)
            mask = torch.tensor(state['mask'][0], device=dev, dtype=torch.bool)
            ar, ur = (x.to(dev) for x in factors[si])
            pending, descriptors = [], []
            def add(dataset, condition, pattern, target, donor=None, shift=None, source_row=None):
                d, achieved = matched_perturbation(pattern, h, tau[si].to(dev), EPS)
                pending.append(d[mask])
                descriptors.append({'receiver': si, 'sequence': state['sequence_index'], 'timestep': state['timestep'],
                                    'dataset': dataset, 'condition': condition, 'donor': donor, 'shift': shift,
                                    'source_row': source_row, 'exact_KL': float(target),
                                    'achieved_relative_l2': achieved.item()})
            donor = swap_map['same_timestep_donors'][si]
            add('A', 'native', deltas[si].to(dev), swap['results']['kl'][si,0])
            add('A', 'foreign', deltas[donor].to(dev), swap['results']['kl'][si,2], donor=donor)
            for pair in [x for x in pair_map if x['receiver_index']==si]:
                donor, pi = pair['donor_index'], pair['pair_index']
                ad, ud = (x.to(dev) for x in factors[donor])
                for ci, (name, a, u) in enumerate([('NN',ar,ur),('NF',ar,ud),('FN',ad,ur),('FF',ad,ud)]):
                    add('B', name, a[:,None]*u, factor['results']['kl'][pi,ci], donor=donor, source_row=pi)
            row_art = torch.load(ROOT/'token_row_pairing_shards'/f'state_{si:02d}.pt',map_location='cpu',weights_only=True)
            for family in ['unrestricted', 'class_preserving']:
                for k in SHIFT:
                    if family == 'unrestricted':
                        perm = torch.roll(torch.arange(256,device=dev),-k); idx=k-1
                    else:
                        indices = [i for i, ks in enumerate(perms['class_source_shifts'][si]) if k in ks]
                        if len(indices)!=1: raise RuntimeError('class shift mapping ambiguous')
                        idx=indices[0]; perm=perms['class_permutations'][si][idx].to(dev)
                    for ci,(name,a,u) in enumerate([('NN',ar,ur),('NS',ar,ur[perm]),('SN',ar[perm],ur),('SS',ar[perm],ur[perm])]):
                        add('C_'+family,name,a[:,None]*u,row_art[family]['kl'][idx,ci],shift=k,source_row=idx)
            # Audit smooth dense readout vs BF16 output, only at unperturbed h.
            smooth = []
            y = rms_forward(h[mask].float(),norm.weight.float(),norm.eps)
            for start in range(0,head.weight.shape[0],8192):
                smooth.append(F.linear(y,head.weight[start:start+8192].float()))
            smooth = torch.cat(smooth,-1)
            scale = 1/math.sqrt(4096) if model.model.config.scale_logits else 1.
            smooth *= scale
            dense_bf16, _ = final_suffix(model,h.expand(3,-1,-1).contiguous())
            rounded = dense_bf16[0,mask].float()
            rounding_checks.append({'state':si,'dense_logit_relative_l2':((smooth-rounded).norm()/rounded.norm()).item(),
                                    'dense_logit_max_abs':(smooth-rounded).abs().max().item()})
            del smooth, rounded, dense_bf16
            for start in range(0,len(pending),8):
                values = score_directions(h[mask],torch.stack(pending[start:start+8]),norm.weight,norm.eps,
                                          head.weight,scale=scale)
                predictor_rows.extend(values.cpu().tolist())
            records.extend(descriptors)
            save_json('progress.json', {'states_complete':si+1,'total_states':40,'records':len(records),'elapsed_seconds':time.time()-started})
            print(f'geometry state {si+1}/40; records {len(records)}',flush=True)
    after = model_sha(model)
    if before != after: raise RuntimeError('model weights changed')
    for record, values in zip(records,predictor_rows):
        record.update(dict(zip(['B_hidden','B_logit','Q_geom'],values)))
    save_json('per_condition.json', records)
    save_json('dense_precision_audit.json', rounding_checks)
    save_json('run_manifest.json', {'status':'complete','records':len(records),'weight_sha_before':before,'weight_sha_after':after,
                                   'source_hashes':verified,'h31_artifact_sha256':sha(hidden_path),'h31_tensor_sha256':tsha(hiddens),
                                   'predictions_sha256':sha(OUT/'per_condition.json'),
                                   'dense_same_path_max_logit_error':max(dense_checks),
                                   'elapsed_seconds':time.time()-started,'environment':{'python':platform.python_version(),'torch':torch.__version__},
                                   'code_hashes':{name:sha(ROOT/name) for name in ['geometry_core.py','run_geometry_kill_gate.py']}})


if __name__=='__main__': main()
