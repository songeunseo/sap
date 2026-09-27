import json
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
print('DLM scale/shape50 |',datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
p=ROOT/'progress.json'
if p.exists():
 r=json.loads(p.read_text());print('Stage:',r['stage'])
 for k in ('condition','candidate','completed','total','blocks_completed','blocks_total','error'):
  if k in r:print(f'{k}: {r[k]}')
else:print('Not started')
print('Projection distributions:',len(list((ROOT/'statistics').glob('*.json'))),'/224')
print('Evaluated models:',len(list((ROOT/'evaluation').glob('*/results.json'))),'/25')
if (ROOT/'results.json').exists():
 r=json.loads((ROOT/'results.json').read_text());print('Raw minus shape NELBO:',r['primary']['raw_minus_shape']);print('95% article CI:',r['primary']['paired_article_ci'])
