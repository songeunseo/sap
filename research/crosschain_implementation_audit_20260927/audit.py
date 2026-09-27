"""Independent CPU review; never modifies frozen experiment artifacts or calls CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json, hashlib, tempfile, shutil, math, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, binomtest
import torch
from experiments.dlm_multiscale_ac50.artifacts import read, sha, digest, write, mask_identity
from experiments.dlm_multiscale_ac50.evaluation import grade, task_and_protocol, read_predictions
from experiments.dlm_owl65.core import exact_row_counts
from experiments.dlm_crosschain_control50 import run
from experiments.dlm_crosschain_control50.prepare import ROOT, OLD, PRIOR_VALIDATION

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
run.launch_gates()  # read-only: do NOT call checks()/validate CLI, which rewrites its receipt.
c=read(ROOT/'config.json'); allocation=read(ROOT/'allocation.json'); requests=read(ROOT/'requests.json')['development']
ch=sha(ROOT/'config.json'); ah=sha(ROOT/'allocation.json')
out={'time':time.time(),'config_sha256':ch,'allocation_sha256':ah,'gpu_used':False,'source_files':run.code_sources()}

def independent(values,teacher):
    e=(np.asarray(values,dtype=np.float64)-np.asarray(teacher,dtype=np.float64)).reshape(8,2,8,8)
    result={'A':float(np.mean(e*e))}
    for kind in ('natural','cross'):
        terms=[]
        for d in (1,2,4):
            terms_d=[]
            for s in range(8):
                for chain in range(2):
                    other=chain if kind=='natural' else 1-chain
                    for i in range(8):
                        if i&d:continue
                        terms_d.extend(((e[s,other,i+d]-e[s,chain,i])**2).tolist())
            terms.append(float(np.mean(terms_d)))
        result[kind]=float(np.mean(terms))
    return result
teacher=read(OLD/'readouts/dense/calibration.json')['values']
u=independent(read(OLD/'readouts/uniform/calibration.json')['values'],teacher)
beta=u['natural']/u['cross'];assert abs(beta-allocation['beta'])<1e-14
scores={a:[] for a in run.ARMS}
for b in range(32):
    probe=read(OLD/'probes'/f'block{b:02d}.json')
    low,high=[probe['conditions'][str(r)] for r in (.48,.52)]
    mm=[independent(read(p['readout_path'])['values'],teacher) for p in (low,high)]
    denominator=high['pruned']-low['pruned']
    a=(mm[1]['A']-mm[0]['A'])/denominator
    cn=(mm[1]['natural']-mm[0]['natural'])/denominator
    cx=(mm[1]['cross']-mm[0]['cross'])/denominator
    for arm,v in zip(run.ARMS,(a,a+cn,a+cx,a+beta*cx)):scores[arm].append(v)
refs=read(c['legacy_manifests']['uniform']['path'])['entries']
max_errors={}
for arm in run.ARMS:
    expected=allocation['allocations'][arm]
    max_errors[arm]=float(np.max(np.abs(np.subtract(scores[arm],expected['scores']))))
    assert np.allclose(scores[arm],expected['scores'],atol=1e-20,rtol=1e-12)
    ranks=(rankdata(scores[arm],method='average')-1)/31
    rates=.5-.1*(ranks-ranks.mean())
    counts,budget=exact_row_counts(refs,rates,3489660928)
    assert counts==expected['row_counts']
    assert budget['corrected_pruned']==3489660928
out['independent_objective_replay']={'beta':beta,'max_marginal_error':max_errors,'all_four_exact_DP_counts_match':True}

lookup=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
mask_results={}
for arm in run.ARMS:
    mf=read(ROOT/'candidates'/arm/'mask_manifest.json');ident=read(ROOT/'candidates'/arm/'model_identity.json')
    assert ident['config_sha256']==ch and ident['mask_identity']==mask_identity(mf)
    actual=0
    for entry in mf['entries']:
        meta=entry['selected_mask'];p=Path(meta['path']);assert sha(p)==meta['file_sha256']
        packed=torch.load(p,map_location='cpu',weights_only=False)
        h,w=entry['shape'];assert list(packed['shape'])==[h,w] and w%8==0
        b=np.frombuffer(packed['bits'],dtype=np.uint8).reshape(h,w//8)
        row_counts=lookup[b].sum(axis=1)
        assert np.all(row_counts==meta['prune_per_row'])
        header=json.dumps({'shape':[h,w],'bitorder':'big'},sort_keys=True,separators=(',',':')).encode()
        assert hashlib.sha256(header+packed['bits']).hexdigest()==meta['mask_sha256']
        actual+=int(row_counts.sum())
    assert actual==3489660928
    if arm in ('A','Multi'):
        oldid=read(OLD/'candidates'/arm/'model_identity.json')
        assert oldid['mask_identity']==ident['mask_identity'] and oldid['sparse_model_sha256']==ident['sparse_model_sha256']
    mask_results[arm]={'projections':len(mf['entries']),'physical_pruned_bits':actual,'identity':ident}
out['physical_masks']=mask_results

assert [r['example_id'] for r in requests]==list(range(1319))
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(c['model']['id'],revision=c['model']['revision'],trust_remote_code=True,local_files_only=True)
for r in requests:
    assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_hash']
    assert digest(tokenizer(r['prompt'])['input_ids'])==r['input_ids_sha256']
out['requests']={'total':1319,'all_prompt_and_token_hashes_rechecked':True,'primary_count':len(c['crosschain_control']['primary_ids'])}
task,_,_=task_and_protocol(read(c['legacy_config']))
rows={};history={};snapshot_paths={}
for arm in run.ARMS:
    snapshot_paths[arm]=sorted((ROOT/'gsm8k'/arm).glob('shard*/examples/*.json'))
    seen=set();rr={}
    ident=mask_results[arm]['identity']
    for p in snapshot_paths[arm]:
        item=read(p);r=item['row'];i=r['example_id'];assert i not in seen;seen.add(i)
        assert p.name==f'{i:04d}.json' and p.parent.parent.name==f'shard{i//128:02d}'
        expected={'config_sha256':ch,'requests_sha256':sha(ROOT/'requests.json'),'mask_identity':ident['mask_identity'],
            'sparse_model_sha256':ident['sparse_model_sha256'],'split':'full','shard':i//128,'protocol_hash':c['protocol_hash']}
        assert item['fingerprint']==expected and digest(r)==item['row_sha256']
        assert r['method']==arm
        for key in ('doc_hash','prompt_hash','target_hash','reference_answer'):assert r[key]==requests[i][key]
        score=grade(task,requests[i],r['generated_text'])
        assert score['correct']==r['correct'] and score['extracted_answer']==r['extracted_answer']
        rr[i]=r
    rows[arm]=rr
    if arm in ('A','Multi'):
        prior=read(OLD/'gsm8k/development'/arm/'predictions.json')+read(PRIOR_VALIDATION/'gsm8k/validation100'/arm/'predictions.json')
        available=[p for p in prior if p['example_id'] in rr]
        mismatches={k:[p['example_id'] for p in available if p[k]!=rr[p['example_id']][k]] for k in ('generated_text','extracted_answer','correct')}
        history[arm]={'compared':len(available),'mismatches':mismatches}
out['live_checkpoint_snapshot']={a:{'verified':len(r),'ids':sorted(r)} for a,r in rows.items()}
out['historical_reproduction']=history

# A real isolated counterexample: two filenames encode the same ID.
source=ROOT/'gsm8k/A/shard00/examples/0000.json';item=read(source)
with tempfile.TemporaryDirectory() as td:
    f=Path(td);(f/'examples').mkdir()
    for name in ('0000.json','0.json'):shutil.copyfile(source,f/'examples'/name)
    try:
        rr=read_predictions(f,[requests[0]],item['fingerprint'],c['protocol_hash'])
        out['duplicate_filename_probe']={'rejected':False,'input_files':2,'returned_rows':len(rr)}
    except RuntimeError as e:out['duplicate_filename_probe']={'rejected':True,'error':str(e)}

# Check primary exact p and Holm implementation against direct binomial sums.
from experiments.dlm_multiscale_ac50.core import holm
for gains,losses in ((0,0),(10,2),(3,7),(100,72)):
    a=[False]*gains+[True]*losses+[True]*5;b=[True]*gains+[False]*losses+[True]*5
    value=run.paired_stats(a,b,draws=1000)
    n=gains+losses;direct=min(1.,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n) if n else 1.
    assert abs(value['exact_mcnemar_p']-direct)<1e-12
x=holm({str(i):{'exact_mcnemar_p':p} for i,p in enumerate((.02,.01,.03))})
assert [x[str(i)]['holm_p'] for i in range(3)]==[.04,.03,.04]
out['statistics']={'exact_McNemar_checked':True,'Holm_three_checked':True,'bootstrap_unit':'paired question; diagnostic paired span'}

costs={}
for folder in (ROOT,ROOT.parent/'output_attempt1_scheduler_failure_20260927'):
    attempts=sorted((folder/'attempts').glob('*.json'));cc=sorted((folder/'costs').glob('*.json'))
    costs[folder.name]={'attempts':len(attempts),'finalized_cost_records':len(cc),'recorded_forwards':sum(read(p)['forward_calls'] for p in cc),
        'generation_forwards':sum(read(p)['forward_calls'] for p in cc if read(p)['job'] not in ('teacher','init_A','init_Multi','init_Cross','init_CrossMatched')),
        'unfinalized_attempts':[p.name for p in attempts if not (folder/'costs'/p.name).exists()]}
out['cost_snapshot']=costs
out['all_observed_scientific_integrity_checks_passed']=True
write(HERE/'evidence.json',out)
write(HERE/'effective_plan.json',{'role':'audit companion; original frozen config remains unchanged','config_sha256':ch,
    'arms':list(run.ARMS),'full_gsm8k':True,'questions_per_arm':1319,'fresh_answers':5276,'previously_seen_ids':c['crosschain_control']['exposed_ids'],
    'primary_ids':c['crosschain_control']['primary_ids'],'primary_family':c['crosschain_control']['primary_family'],
    'model':c['model'],'evaluation':c['evaluation'],'pruning':c['pruning'],'beta':allocation['beta'],
    'gpu_authorization':[0,1,2,3],'nominal_generation_forwards':1350656,'nominal_diagnostic_forwards':1152,
    'historical_config_plan_metadata_is_not_current_plan':True})
print(json.dumps({k:out[k] for k in ('independent_objective_replay','historical_reproduction','duplicate_filename_probe','cost_snapshot','all_observed_scientific_integrity_checks_passed')},indent=2))
