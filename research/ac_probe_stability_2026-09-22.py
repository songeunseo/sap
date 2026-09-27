"""Descriptive reanalysis of frozen scalar A+C JSONs; no model or new mask."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/dlm_context_response50'
OUT = Path(__file__).with_suffix('.json')
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def corr(x, y):
    xc, yc = [v - mean(x) for v in x], [v - mean(y) for v in y]
    denominator = math.sqrt(sum(v*v for v in xc) * sum(v*v for v in yc))
    return sum(a*b for a, b in zip(xc, yc)) / denominator if denominator else None


def rates(costs):
    r = ranks(costs)
    return [.5 - .1 * (v - mean(r)) / 31 for v in r]


def main():
    pairs = read(EXP / 'pairs.json')['pairs']
    receipt = read(EXP / 'collection_receipt.json')
    probes = [read(EXP / 'probes' / f'block{b:02d}.json') for b in range(32)]
    allocation = read(EXP / 'allocation.json')['allocations']
    assert len(pairs) == 80
    for path, expected in receipt['files'].items():
        if path in inputs:
            assert inputs[path] == expected, path
    costs_by_state = {k: [] for k in ('A', 'C', 'AC')}
    for p in probes:
        low, high = p['conditions']['0.48'], p['conditions']['0.52']
        assert len(low['rows']) == len(high['rows']) == len(pairs)
        denominator = high['pruned'] - low['pruned']
        for key in costs_by_state:
            c = [(h[key]-l[key])/denominator for h, l in zip(high['rows'], low['rows'])]
            assert math.isclose(mean(c), p['costs'][key], rel_tol=1e-9, abs_tol=1e-16)
            costs_by_state[key].append(c)
    costs = {k: [mean(x) for x in v] for k, v in costs_by_state.items()}
    for k in ('A', 'AC'):
        assert max(abs(a-b) for a,b in zip(rates(costs[k]), allocation[k]['rates'])) < 1e-14
    dr = [a-b for a,b in zip(allocation['AC']['rates'], allocation['A']['rates'])]
    spans = sorted({p['sequence_index'] for p in pairs})
    loo = []
    for span in spans:
        keep = [i for i,p in enumerate(pairs) if p['sequence_index'] != span]
        lc = {k: [mean(v[i] for i in keep) for v in costs_by_state[k]] for k in costs}
        lr = {k: rates(lc[k]) for k in ('A','AC')}
        ld = [a-b for a,b in zip(lr['AC'], lr['A'])]
        active = [i for i,d in enumerate(dr) if abs(d) > 1e-14]
        loo.append(dict(excluded_span=span,
            rank_rho={k:corr(ranks(costs[k]), ranks(lc[k])) for k in costs},
            delta_rate_sign_agreement=sum(dr[i]*ld[i]>0 for i in active)/len(active),
            AC_rate_mean_abs_shift_pp=mean(abs(a-b)*100 for a,b in zip(lr['AC'], allocation['AC']['rates'])),
            delta_rates_pp=[d*100 for d in ld]))
    joint = {'uniform':read(EXP / 'uniform_distortion.json')}
    for k in ('A','AC'):
        joint[k] = read(EXP / k / 'joint_distortion.json')
    by_span = []
    for span in spans:
        indices = [i for i,p in enumerate(pairs) if p['sequence_index'] == span]
        by_span.append(dict(span=span, AC_minus_A={k:mean(joint['AC']['rows'][i][k]-joint['A']['rows'][i][k] for i in indices) for k in costs}))
    improvement = {k: joint['uniform']['mean'][k]-joint['AC']['mean'][k] for k in costs}
    result = dict(status='complete', scope='Frozen in-calibration descriptive audit; no new masks, model forwards or benchmarks. Leave-one-span rank recomputation does not evaluate its resulting masks and is not held-out validation.',
        verified_receipt_files=sum(p in inputs for p in receipt['files']), input_sha256=inputs,
        A_AC_cost_spearman=corr(ranks(costs['A']), ranks(costs['AC'])),
        C_vs_A_cost_spearman=corr(ranks(costs['C']), ranks(costs['A'])),
        AC_vs_A_changed_layer_ideal_rates=sum(abs(d)>1e-14 for d in dr),
        AC_vs_A_ideal_rate_mean_abs_pp=mean(abs(d)*100 for d in dr),
        AC_vs_A_ideal_rate_max_abs_pp=max(abs(d)*100 for d in dr),
        C_fraction_of_uniform_to_AC_total_distortion_reduction=improvement['C']/improvement['AC'],
        joint_means={k:v['mean'] for k,v in joint.items()},
        joint_AC_vs_A_by_span=by_span,
        AC_beats_A_on_C_spans=sum(v['AC_minus_A']['C']<0 for v in by_span),
        AC_beats_A_on_total_spans=sum(v['AC_minus_A']['AC']<0 for v in by_span),
        leave_one_span_out=loo)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','leave_one_span_out','joint_AC_vs_A_by_span')}, indent=2))
    print('LOO rank rho AC:', [round(v['rank_rho']['AC'],4) for v in loo])
    print('LOO A-to-AC allocation direction agreement:', [round(v['delta_rate_sign_agreement'],4) for v in loo])


if __name__ == '__main__':
    main()
