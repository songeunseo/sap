"""Post-hoc CPU analysis of the frozen Multi follow-up; raw artifacts stay read-only."""
from pathlib import Path
import json,csv,math,hashlib
import numpy as np
BASE=Path('/home/tmluser1/sap')
OUT=Path(__file__).resolve().parent
NEW=BASE/'experiments/dlm_multi_followup50/output'
OLD=BASE/'experiments/dlm_multiscale_ac50/output'
SOURCES={}
def read(p):
 p=Path(p);data=p.read_bytes();SOURCES[str(p)]=hashlib.sha256(data).hexdigest();return json.loads(data)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
report=read(NEW/'report.json');assert report['status']=='complete'
arms=['Multi','Multi-Bag','Multi-R2','Multi-R3.5'];roots={a:OLD if a=='Multi' else NEW for a in arms}
rows={a:read(roots[a]/'gsm8k/development'/a/'predictions.json') for a in arms}
for a,rr in rows.items():
 assert len(rr)==100 and [r['example_id'] for r in rr]==list(range(100))
 assert sum(r['correct'] for r in rr)==report['scores'][a]['correct']
 for x,y in zip(rows['Multi'],rr):
  assert all(x[k]==y[k] for k in ('example_id','doc_hash','prompt_hash','target_hash','evaluation_config_hash'))
def invalid(r):return r['extracted_answer'] in ('[invalid]','',None)
def state(r):return 'correct' if r['correct'] else 'invalid' if invalid(r) else 'valid_wrong'
paired={}
for a in arms[1:]:
 rr=rows[a];g=[i for i in range(100) if rr[i]['correct'] and not rows['Multi'][i]['correct']];l=[i for i in range(100) if rows['Multi'][i]['correct'] and not rr[i]['correct']]
 n=len(g)+len(l);p=min(1,2*sum(math.comb(n,i) for i in range(min(len(g),len(l))+1))/2**n)
 assert abs(p-report['paired_vs_Multi'][a]['exact_mcnemar_p'])<1e-14
 trans={}
 for x,y in zip(rows['Multi'],rr):
  key=state(x)+' -> '+state(y);trans[key]=trans.get(key,0)+1
 paired[a]=dict(gained_ids=g,lost_ids=l,transitions=trans,**report['paired_vs_Multi'][a])
invalid_ids={a:[r['example_id'] for r in rr if invalid(r)] for a,rr in rows.items()}
common_lost=sorted(set.intersection(*(set(x['lost_ids']) for x in paired.values())))
common_gained=sorted(set.intersection(*(set(x['gained_ids']) for x in paired.values())))
diags={a:{s:read(roots[a]/'diagnostics'/a/f'{s}.json') for s in ('calibration','diagnostic')} for a in arms}
uniform={s:read(OLD/'diagnostics/uniform'/f'{s}.json') for s in ('calibration','diagnostic')}
metrics=['A','C1','C2','C4','Multi','query_CE','response_sign_flip_rate']
comparisons={}
for a in arms[1:]:
 comparisons[a]={}
 for s in ('calibration','diagnostic'):
  old=diags['Multi'][s];new=diags[a][s];oldseq=old['sequences'];newseq=new['sequences']
  assert [x['sequence_index'] for x in oldseq]==[x['sequence_index'] for x in newseq]
  d={k:new['mean'][k]-old['mean'][k] for k in metrics}
  per={k:[y['metrics'][k]-x['metrics'][k] for x,y in zip(oldseq,newseq)] for k in metrics}
  phase=(np.array([x['phase_mse'] for x in newseq])-np.array([x['phase_mse'] for x in oldseq])).mean(0)
  comps={'A':d['A'],'C1/3':d['C1']/3,'C2/3':d['C2']/3,'C4/3':d['C4']/3}
  assert abs(sum(comps.values())-d['Multi'])<1e-12
  gap=uniform[s]['mean']['Multi']-old['mean']['Multi']
  comparisons[a][s]=dict(delta=d,relative_percent={k:100*d[k]/old['mean'][k] for k in metrics},span_delta=per,improved_spans={k:sum(v<0 for v in per[k]) for k in metrics},objective_delta_components=comps,objective_delta_component_percent={k:100*v/d['Multi'] for k,v in comps.items()},phase_mse_delta=phase.tolist(),fraction_uniform_to_multi_gain_lost=d['Multi']/gap)
oldalloc=read(OLD/'allocation.json')['allocations']['Multi'];newalloc=read(NEW/'allocation.json')['allocations'];c=read(NEW/'config.json');refs=read(c['legacy_manifests']['uniform']['path'])['entries'];base_rates=np.array(oldalloc['rates']);base_counts=np.array(oldalloc['row_counts']);weights=sum(x['shape'][0]*x['shape'][1] for x in refs)
alloc={}
for a,x in newalloc.items():
 rates=np.array(x['rates']);counts=np.array(x['row_counts']);actual=counts/np.array([r['shape'][1] for r in refs]);delta=rates-base_rates
 xor=sum(int(abs(k-j))*r['shape'][0] for k,j,r in zip(counts,base_counts,refs))
 alpha=float(np.dot(rates-.5,base_rates-.5)/np.dot(base_rates-.5,base_rates-.5));resid=rates-(.5+alpha*(base_rates-.5))
 order=np.argsort(-abs(delta))[:6]
 alloc[a]=dict(ideal_min_percent=100*rates.min(),ideal_max_percent=100*rates.max(),actual_min_percent=100*actual.min(),actual_max_percent=100*actual.max(),mean_abs_rate_change_pp=100*abs(delta).mean(),mean_abs_deviation_from_uniform_pp=100*abs(rates-.5).mean(),base_mean_abs_deviation_from_uniform_pp=100*abs(base_rates-.5).mean(),nested_support_implied_xor_percent=100*xor/weights,changed_projections=int(sum(counts!=base_counts)),best_scalar_shrink_factor=alpha,shrink_residual_rms_pp=100*float(np.sqrt((resid**2).mean())),largest_changes=[dict(block=int(i),old_percent=100*base_rates[i],new_percent=100*rates[i],delta_pp=100*delta[i]) for i in order],budget=x['budget'])
cost=report['cost_accounting'];assert cost['complete_accounting'] and len(cost['attempts'])==3
assert cost['cumulative_state_forwards']==768 and cost['cumulative_generation_forwards']==76800
execution=read(NEW/'execution.json');assert execution['status']=='complete'
res=dict(scope='Post-hoc descriptive analysis; mini100 and diagnostic bank are reused development evidence; no new GPU runs',scores=report['scores'],paired=paired,invalid_ids=invalid_ids,common_lost_ids=common_lost,common_gained_ids=common_gained,diagnostic_means={a:{s:x['mean'] for s,x in d.items()} for a,d in diags.items()},uniform_means={s:x['mean'] for s,x in uniform.items()},comparisons=comparisons,allocation=alloc,cost=cost,wall_minutes=(execution['ended']-execution['started'])/60)
dump(OUT/'analysis.json',res)
with (OUT/'question_comparison.csv').open('w') as f:
 fields=['example_id']+[a+'_'+k for a in arms for k in ('correct','extracted_answer','state')];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for i in range(100):
  row={'example_id':i}
  for a in arms:
   row.update({a+'_correct':rows[a][i]['correct'],a+'_extracted_answer':rows[a][i]['extracted_answer'],a+'_state':state(rows[a][i])})
  w.writerow(row)
req=read(NEW/'requests.json')['development'];changed=sorted(set.union(*(set(x['gained_ids']+x['lost_ids']) for x in paired.values())))
text=['# All changed questions','All IDs are zero-based. Includes every question whose correctness changed vs Multi.']
for i in changed:
 text += [f'\n## ID {i}',req[i]['doc']['question'],'\nReference:\n'+rows['Multi'][i]['reference_answer']]
 for a in arms:text += [f'\n### {a}: {state(rows[a][i])}',rows[a][i]['generated_text']]
(OUT/'changed_questions.md').write_text('\n\n'.join(text))
dump(OUT/'sources.json',SOURCES)
print(json.dumps({k:res[k] for k in ('paired','invalid_ids','common_lost_ids','common_gained_ids','allocation','wall_minutes')},indent=2))
for a in arms[1:]:
 print(a)
 for s in ('calibration','diagnostic'):
  z=comparisons[a][s];print(s,'REL',z['relative_percent'],'BETTER SPANS',z['improved_spans'],'COMPONENTS',z['objective_delta_component_percent'],'PHASE',z['phase_mse_delta'],'UNIFORM GAP LOST',z['fraction_uniform_to_multi_gain_lost'])
print('UNIFORM',res['uniform_means'])
