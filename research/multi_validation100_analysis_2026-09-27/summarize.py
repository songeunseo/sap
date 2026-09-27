"""Read-only inputs; derived summary for the separate validation100."""
from pathlib import Path
import json,hashlib,csv
P=Path('/home/tmluser1/sap/experiments/dlm_multi_validation10050/output')
O=Path(__file__).resolve().parent
sources={}
def read(p):
 data=p.read_bytes();sources[str(p)]=hashlib.sha256(data).hexdigest();return json.loads(data)
r=read(P/'report.json');state=read(P/'execution.json');assert r['status']==state['status']=='complete'
rows={a:read(P/'gsm8k/validation100'/a/'predictions.json') for a in ('Multi','A','Uniform')}
ids=r['document_ids'];assert len(ids)==len(set(ids))==100
for a,rr in rows.items():
 assert [x['example_id'] for x in rr]==ids and sum(x['correct'] for x in rr)==r['scores'][a]['correct']
 for x,y in zip(rr,rows['Multi']):assert all(x[k]==y[k] for k in ('doc_hash','prompt_hash','target_hash','evaluation_config_hash'))
comparisons={}
for ref,a in (('A','Multi'),('Uniform','Multi'),('Uniform','A')):
 key=f'{a}_minus_{ref}';g=[x['example_id'] for x,y in zip(rows[a],rows[ref]) if x['correct'] and not y['correct']];l=[x['example_id'] for x,y in zip(rows[a],rows[ref]) if not x['correct'] and y['correct']]
 assert len(g)==r['paired'][key]['gained'] and len(l)==r['paired'][key]['lost']
 comparisons[key]=dict(**r['paired'][key],gained_ids=g,lost_ids=l)
summary=dict(scores=r['scores'],paired=comparisons,invalid_counts={a:sum(x['extracted_answer'] in ('[invalid]','',None) for x in rr) for a,rr in rows.items()},wall_minutes=(state['ended']-state['started'])/60,forward_calls=r['cost_accounting']['cumulative_generation_forwards'],complete_accounting=r['cost_accounting']['complete_accounting'],scope='Previously frozen separate100; historical project exposure not ruled out; separate from development100')
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(O/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
with (O/'questions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['example_id','Multi_correct','A_correct','Uniform_correct'])
 for i,id in enumerate(ids):w.writerow([id,*[rows[a][i]['correct'] for a in rows]])
print(json.dumps(summary,indent=2))
