"""Read-only paired audit of the frozen A and Multi GSM8K answers."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 20260927
DRAWS = 10000


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path, expected_ids):
    data = read(path)
    ids = [r['example_id'] for r in data]
    assert len(ids) == len(set(ids)) == len(expected_ids)
    assert set(ids) == set(expected_ids)
    assert all(type(r['correct']) is bool for r in data)
    return {r['example_id']: r for r in data}


def exact_mcnemar(gain, loss):
    n = gain + loss
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / 2**n)


def paired_ci(values):
    rng = random.Random(SEED)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(DRAWS))
    return [draws[int(.025 * DRAWS)], draws[int(.975 * DRAWS)]]


def compare(ids, a, multi):
    av = [a[i]['correct'] for i in ids]
    mv = [multi[i]['correct'] for i in ids]
    gain = sum(m and not x for m, x in zip(mv, av))
    loss = sum(x and not m for m, x in zip(mv, av))
    vals = [int(m) - int(x) for m, x in zip(mv, av)]
    return dict(n=len(ids), A=sum(av), Multi=sum(mv), gain=gain, loss=loss,
                difference=sum(vals), difference_pp=100 * sum(vals) / len(ids),
                exact_mcnemar_p=exact_mcnemar(gain, loss),
                paired_bootstrap_95_pp=[100 * x for x in paired_ci(vals)])


def main():
    full_root = ROOT / 'experiments/dlm_crosschain_control50/output'
    nfe_root = ROOT / 'experiments/dlm_multi_nfe64/output'
    validation_root = ROOT / 'experiments/dlm_multi_validation10050/output'
    historical_root = ROOT / 'experiments/dlm_multiscale_ac50/output'
    exposed = read(nfe_root / 'config.json')['exposed_ids']
    separate = read(validation_root / 'report.json')['document_ids']
    assert len(exposed) == len(set(exposed)) == 200
    assert exposed[:100] == list(range(100))
    assert len(separate) == 100 and set(exposed[100:]) == set(separate)
    full_ids = list(range(1319))
    heldout = sorted(set(full_ids) - set(exposed))
    assert len(heldout) == 1119
    frozen_full = read(full_root / 'config.json')['crosschain_control']
    assert set(frozen_full['exposed_ids']) == set(exposed)
    assert set(frozen_full['primary_ids']) == set(heldout)
    paths = {}
    data = {}
    for steps, directory, ids, names in [
        ('256', full_root / 'gsm8k', full_ids, ('A', 'Multi')),
        ('64', nfe_root / 'gsm8k', exposed, ('A_64', 'Multi_64')),
    ]:
        paths[steps] = {arm: directory / name / 'predictions.json' for arm, name in zip(('A', 'Multi'), names)}
        data[steps] = {arm: rows(p, ids) for arm, p in paths[steps].items()}
        for i in ids:
            a, m = data[steps]['A'][i], data[steps]['Multi'][i]
            for key in ('example_id', 'doc_hash', 'prompt_hash', 'reference_answer', 'target_hash'):
                assert a[key] == m[key], (steps, i, key)
    for i in exposed:
        for arm in ('A', 'Multi'):
            x, y = data['64'][arm][i], data['256'][arm][i]
            for key in ('example_id', 'doc_hash', 'prompt_hash', 'reference_answer', 'target_hash'):
                assert x[key] == y[key], (arm, i, key)
    reproduction = {}
    for arm in ('A', 'Multi'):
        reproduction[arm] = {}
        for name, p in [
            ('development100', historical_root / f'gsm8k/development/{arm}/predictions.json'),
            ('separate100', validation_root / f'gsm8k/validation100/{arm}/predictions.json'),
        ]:
            old = rows(p, exposed[:100] if name == 'development100' else separate)
            matched = sum(old[i]['generated_text'] == data['256'][arm][i]['generated_text'] for i in old)
            assert matched == 100
            reproduction[arm][name] = dict(historical_correct=sum(r['correct'] for r in old.values()),
                                           fullrun_exact_text_matches=matched, n=100, sha256=sha(p))
    groups = dict(development100=exposed[:100], separate100=exposed[100:],
                  exposed200=exposed, remaining1119=heldout, full1319=full_ids)
    statistics = {steps: {name: compare(ids, data[steps]['A'], data[steps]['Multi'])
                          for name, ids in groups.items() if steps == '256' or name not in ('remaining1119', 'full1319')}
                  for steps in ('256', '64')}
    result = dict(title='A versus Multi sample sensitivity audit', status='complete',
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for group in paths.values() for p in group.values()},
                  split=dict(development100='IDs 0..99', separate100='remaining frozen NFE50 exposed IDs',
                             exposed200=len(exposed), remaining1119=len(heldout), full1319=len(full_ids)),
                  reproduction=reproduction, statistics=statistics,
                  bootstrap=dict(seed=SEED, draws=DRAWS, unit='GSM8K question', method='percentile paired resampling'),
                  interpretation_limits=['All NFE256 subsets are post hoc descriptive partitions of the same full run.',
                                         'The exposed200 includes two development samples and is not independent confirmation.',
                                         'Exact McNemar p-values and unadjusted bootstrap intervals do not correct for prior model selection.',
                                         'A confidence interval crossing zero neither establishes equality nor identifies why the model ranking changed.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    lines = ['# Multi/A sample sensitivity audit', '',
             'Frozen paired GSM8K answer rows were compared by question ID. Positive difference favors Multi.', '',
             '| NFE | Sample | n | A | Multi | Difference | Gains/losses | Exact McNemar p | Paired bootstrap 95% CI (pp) |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for steps in ('256', '64'):
        for name, x in statistics[steps].items():
            lo, hi = x['paired_bootstrap_95_pp']
            lines.append(f"| {steps} | {name} | {x['n']} | {x['A']} | {x['Multi']} | {x['difference_pp']:+.2f} pp | {x['gain']}/{x['loss']} | {x['exact_mcnemar_p']:.4g} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '## Provenance and reproduction', '',
              '- Exposed200 matches the frozen full-run exposed IDs. Development100 is IDs 0–99; separate100 is exactly the other 100 exposed IDs; remaining1119 is their complement in 0–1318.',
              '- The full-run NFE256 generated strings reproduce all historical development100 and separate100 strings for both A and Multi (100/100 each of four arms/samples). Historical correctness is A 60/58 and Multi 63/59.',
              '- Each paired row has matching question, prompt, target and reference hashes; NFE64 rows also match the corresponding NFE256 questions. Source file SHA-256 digests and all numeric results are in summary.json.',
              '- NFE64 has no rows on the remaining1119; its full-benchmark effect is unmeasured.', '',
              '## Interpretation', '',
              'At NFE256, the development100 advantage is +3 questions, separate100 is +1, and remaining1119 is −10. The full1319 difference is −6 (−0.45 percentage points). At NFE64, the same exposed200 splits +5 and −6, yielding −1 overall. Thus the sign change can arise from ordinary question-to-question variation around a small observed effect. The paired intervals are wide and all include zero.', '',
              'Limiting early screening to 100 questions was reasonable for reducing generation cost, but the result was too uncertain to establish that Multi was better. Reusing and selecting methods on those questions can favor a noisy positive result; these data cannot determine how much selection contributed. The separate100 was already frozen and improves the picture, but it was also previously exposed development data. The remaining1119 at NFE256 is the strongest sample for judging that setting, and it does not support a Multi advantage. NFE64 and NFE256 differ in forward steps, so their rankings answer different questions.', '',
              'The NFE32 follow-up is exploratory because the step count was selected after the prior Dense/A and Multi64 results. Its result should be read on that basis.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
