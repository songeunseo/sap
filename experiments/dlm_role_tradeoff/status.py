"""Read-only progress for the offline tradeoff analysis."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent


def read(name, default):
    path = root / name
    return json.loads(path.read_text()) if path.exists() else default


solutions = read('solutions.json', {})
frontier = read('frontier.json', [])
folds = read('crossfit.json', [])
status = read('status.json', {}).get('status', 'running or interrupted; inspect log/process')
print('Status:', status)
print('Full-data objectives:', sum(k in solutions for k in ('masked','unmasked','local_max','minimax')), '/ 4')
print('Frontier samples:', len(frontier), '/ 11')
print('Sequence folds:', len(folds), '/ 8')
for kind in ('masked','unmasked','local_max','minimax'):
    if kind in solutions:
        s = solutions[kind]
        print(kind, 'relative risks=', s['relative_to_reference'], 'solver gap=', s['gap'])
print('Log:', root / 'logs/warm_start.log')
