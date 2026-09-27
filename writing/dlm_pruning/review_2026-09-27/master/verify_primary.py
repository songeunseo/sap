"""Master's independent count/statistic check; no imports from experiment analysis."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,math
import numpy as np

REPO=Path('/home/tmluser1/sap')
ROOT=REPO/'writing/dlm_pruning'
RUN=REPO/'experiments/dlm_crosschain_control50/output'
OUT=ROOT/'review_2026-09-27/master'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

config=read(RUN/'config.json'); closeout=read(ROOT/'closeout.json')
report=read(RUN/'report.json'); execution=read(RUN/'execution.json')
assert report['status']==execution['status']=='complete'
assert not execution['workers']
assert sha(ROOT/'closeout.json')==read(ROOT/'state.json')['closeout_sha256']
sources={str(p):sha(p) for p in [RUN/'config.json',RUN/'report.json',RUN/'execution.json',ROOT/'closeout.json']}
arms=('A','Multi','Cross','CrossMatched'); rows={}
for arm in arms:
    p=RUN/'gsm8k'/arm/'predictions.json'; data=read(p)
    assert len(data)==1319 and [x['example_id'] for x in data]==list(range(1319))
    assert all(type(x['correct']) is bool for x in data)
    rows[arm]={x['example_id']:x for x in data};sources[str(p)]=sha(p)
primary=config['crosschain_control']['primary_ids']; exposed=config['crosschain_control']['exposed_ids']
assert len(set(primary))==len(primary)==1119 and len(set(exposed))==len(exposed)==200
assert set(primary).isdisjoint(exposed) and set(primary+exposed)==set(range(1319))
groups={'full_1319':list(range(1319)), 'previously_seen_200':exposed,'primary_remaining_1119':primary}
output={}
for name,ids in groups.items():
    scores={a:{'correct':sum(rows[a][i]['correct'] for i in ids),'total':len(ids)} for a in arms}
    assert scores==closeout['full']['scores'][name]==report['scores'][name]
    contrasts={}
    for control in ('A','Cross','CrossMatched'):
        table=[[0,0],[0,0]]
        delta=[]
        for i in ids:
            a=int(rows[control][i]['correct']);b=int(rows['Multi'][i]['correct'])
            table[a][b]+=1;delta.append(b-a)
        gain,loss=table[0][1],table[1][0];d=gain+loss
        p=float(min(Fraction(1),2*sum((Fraction(math.comb(d,k),2**d) for k in range(min(gain,loss)+1)),Fraction(0)))) if d else 1.
        effect=100*sum(delta)/len(delta)
        rng=np.random.default_rng(20260927);v=np.array(delta,dtype=np.int8)
        boots=np.concatenate([v[rng.integers(0,len(v),size=(1000,len(v)))].sum(axis=1)/len(v) for _ in range(10)])
        ci=(100*np.percentile(boots,[2.5,97.5])).tolist()
        values={'gain':gain,'loss':loss,'net':gain-loss,'difference_pp':effect,'exact_mcnemar_p':p,'paired_bootstrap_95pp_unadjusted':ci}
        key='Multi-'+control
        for k,value in values.items():
            assert np.allclose(value,closeout['full']['comparisons'][name][key][k],rtol=0,atol=1e-12),(name,key,k)
        contrasts[key]=values
    if name=='primary_remaining_1119':
        running=0.
        for index,(key,r) in enumerate(sorted(contrasts.items(),key=lambda kv:kv[1]['exact_mcnemar_p'])):
            running=max(running,min(1.,(3-index)*r['exact_mcnemar_p']))
            r['holm_p']=running
            assert r['holm_p']==closeout['full']['comparisons'][name][key]['holm_p']
    output[name]={'scores':scores,'comparisons':contrasts}
protected=read(ROOT/'review_2026-09-27/source-snapshot.json')
assert all(sha(Path(p))==value for p,value in protected.items())
payload={'status':'verified','role':'master','method':'independent contingency counts, rational exact binomial sums, Holm, NumPy bootstrap with fixed seed; no imported experiment statistic functions','groups':output,'source_hashes':sources,'frozen_v0_artifacts_unchanged':True,'uncertainty_note':'Bootstrap intervals are unadjusted; Holm correction applies to the three primary tests.'}
(OUT/'numeric-verification.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'status':payload['status'],'primary':output['primary_remaining_1119'],'sources':len(sources)}))
