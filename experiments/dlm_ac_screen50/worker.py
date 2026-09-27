"""GPU workers; imported only after explicit device selection inside tmux."""
import os
import time
import resource
from pathlib import Path
import numpy as np
import torch
from experiments.dlm_multiscale_ac50.artifacts import read,write,freeze,sha,checked,digest,mask_identity,Progress
from experiments.dlm_multiscale_ac50.gpu import Runtime,require_gpu_worker,save_tensor
from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol,grade,generator,evaluate_requests,read_predictions
from experiments.dlm_context_response50.core import log_odds
from .prepare import ROOT,MULTI,LEGACY,validate
from .core import square_metrics,vector_pair,mean_rows,proposals,accept
from .integrity import ReadoutStore,read_checked,write_checked,freeze_checked
from .exchange_state import recover as recover_exchange,commit as commit_exchange
from .allocation import load_allocation

class Worker:
    def __init__(self,label):
        self.started=time.monotonic();self.started_wall=time.time()
        require_gpu_worker();torch.set_num_threads(1)
        self.m=validate();self.ch=sha(ROOT/'manifest.json');self.label=label
        self._validate_manifest_dependencies()
        self.progress=Progress(ROOT,label);self.progress('loading_dense')
        from experiments.dlm_multiscale_ac50.prepare import validate as oldvalidate
        from experiments.projection_capacity_allocation_65.run import load_dense
        from eval_llada import set_seed
        c=oldvalidate(MULTI);set_seed(c['evaluation']['torch_seed'])
        self.model,mapping=load_dense();self.model.eval()
        self.rt=Runtime(MULTI,c,self.model,mapping,self.progress)
        self.calls=0;self.checked_teachers=set()
        def tick(module,args,output):self.calls+=1;self.rt.memory_check()
        self.hook=self.model.register_forward_hook(tick)
        self._weight_keys=None
    def _validate_manifest_dependencies(self):
        teacher=self.m.get('legacy_teacher')
        if not isinstance(teacher,dict) or not teacher.get('path') or not teacher.get('sha256'):
            raise RuntimeError('Manifest legacy teacher identity missing')
        checked(teacher['path'],teacher['sha256'])
        hashes=self.m.get('checkpoint_sha256');mapping=self.m.get('checkpoint_map')
        if not isinstance(hashes,dict) or not hashes or not isinstance(mapping,dict) or not mapping:
            raise RuntimeError('Manifest checkpoint identity missing')
        for path,expected in hashes.items():checked(path,expected)
        for name,entry in mapping.items():
            if not isinstance(entry,dict) or not all(k in entry for k in ('path','key','shape','dtype')):
                raise RuntimeError(f'Checkpoint map entry incomplete: {name}')
            if entry['path'] not in hashes:raise RuntimeError(f'Checkpoint map source missing: {name}')
            if not isinstance(entry['shape'],list) or len(entry['shape'])!=2:raise RuntimeError(f'Checkpoint map shape: {name}')
        if not self.m.get('dense_model_sha256'):raise RuntimeError('Manifest dense model identity missing')
        self._legacy_teacher=teacher;self._checkpoint_map=mapping
    def receipt(self,status,error=None):
        self.hook.remove()
        write(ROOT/'costs'/f'{self.label}_{time.time_ns()}.json',dict(job=self.label,status=status,error=error,forward_calls=self.calls,wall_seconds=time.monotonic()-self.started,gpu_assigned_seconds=time.monotonic()-self.started,started=self.started_wall,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    def bank(self,family,split):return read(self.m['banks'][family+'_'+split]['path'])
    @torch.inference_mode()
    def logits(self,ids,query):
        t=torch.tensor(ids if isinstance(ids[0],list) else [ids],device=next(self.model.parameters()).device)
        z=self.model(t).logits[0,query].float().cpu().numpy()
        if not np.isfinite(z).all():raise ValueError('Nonfinite emitted logits')
        return z
    @torch.inference_mode()
    def margin(self,ids,query,gold):
        t=torch.tensor(ids if isinstance(ids[0],list) else [ids],device=next(self.model.parameters()).device)
        z=self.model(t).logits[0,query]
        return log_odds(z,gold).cpu()
    def fp(self,family,split,identity):
        return dict(config_sha256=self.ch,bank_sha256=self.m['banks'][family+'_'+split]['sha256'],mask_identity=identity,readout=family+'-v1')
    def scalar_square(self,split,label,identity):
        b=self.bank('Square',split);p=ROOT/'readouts/Square'/label/f'{split}.json'
        fp=self.fp('Square',split,identity)
        if label!='dense':fp['teacher_sha256']=sha(ROOT/'readouts/Square/dense'/f'{split}.json')
        store=ReadoutStore(p,fp,b['states'],8)
        self.progress('Square_readout',completed=len(store.values),total=b['states'],candidate=label,split=split)
        for i,(q,n) in enumerate((q,n) for q in b['quartets'] for n in q['nodes']):
            if i<len(store.values):continue
            v=self.margin(n['input_ids'],q['query'],q['gold']).tolist();store.append(v)
            self.progress('Square_readout',completed=i+1,total=b['states'],candidate=label,split=split)
        return np.asarray(store.complete()).reshape(-1,4,8)
    def teacher_vectors(self,split):
        b=self.bank('Vector',split);folder=ROOT/'teachers/Vector'/split
        folder.mkdir(parents=True,exist_ok=True)
        fp=self.fp('Vector',split,'dense')
        records=[];missing=[]
        for i,p in enumerate(b['pairs']):
            for ep in ['before','after']:
                path=folder/f'{i:03d}_{ep}.npy';meta=path.with_suffix('.json')
                if meta.exists():
                    row=read_checked(meta,{'fingerprint':fp})
                    checked(path,row['sha256']);records.append(row)
                else:missing.append((i,p,ep,path,meta))
        reserve=folder/'reservation.bin'
        need=sum(len(p['query'])*self.m['memory']['vocab']*4+512 for _,p,_,_,_ in missing)
        if need:
            import shutil
            if shutil.disk_usage(folder).free+ (reserve.stat().st_blocks*512 if reserve.exists() else 0)<need+2**30:raise RuntimeError('Vector teacher disk reservation failed')
            with reserve.open('wb') as f:os.posix_fallocate(f.fileno(),0,need)
        for j,(i,p,ep,path,meta) in enumerate(missing):
            z=self.logits(p[ep],p['query'])
            if z.shape!=(len(p['query']),self.m['memory']['vocab']):raise ValueError('Vocabulary/query width')
            need-=z.nbytes+512
            with reserve.open('r+b') as f:f.truncate(max(0,need))
            tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f:np.save(f,z,allow_pickle=False);f.flush();os.fsync(f.fileno())
            tmp.replace(path)
            row=dict(path=str(path),sha256=sha(path),fingerprint=fp,index=i,endpoint=ep,shape=list(z.shape),dtype='float32')
            freeze_checked(meta,row);records.append(row)
            self.progress('Vector_teacher',completed=160-len(missing)+j+1,total=160,split=split)
        reserve.unlink(missing_ok=True)
        freeze_checked(folder/'complete.json',dict(fingerprint=fp,files=sorted(records,key=lambda x:(x['index'],x['endpoint']))))
    def vector_teacher(self,split):
        folder=ROOT/'teachers/Vector'/split;r=read_checked(folder/'complete.json',{'fingerprint':self.fp('Vector',split,'dense')})
        if r['fingerprint']!=self.fp('Vector',split,'dense') or len(r['files'])!=160:raise ValueError('Incomplete Vector teacher')
        if split not in self.checked_teachers:
            for x in r['files']:checked(x['path'],x['sha256'])
            self.checked_teachers.add(split)
        return folder
    def pair_score(self,split,label,identity,scalar=False,repeat=False):
        b=self.bank('Vector',split);folder=self.vector_teacher(split)
        fp=self.fp('Vector',split,identity);fp.update(reduction='scalar' if scalar else 'vector-fp64-mean-pair',teacher_sha256=sha(folder/'complete.json'))
        rows=[]
        for i,p in enumerate(b['pairs']):
            path=ROOT/'pair_metrics'/('scalar' if scalar else 'Vector')/label/split/f'{i:03d}.json'
            if path.exists() and not repeat:
                old=read_checked(path,{'fingerprint':fp,'pair_sha256':digest(p)})
                rows.append(old['metrics']);continue
            dense=[np.load(folder/f'{i:03d}_{ep}.npy',mmap_mode='r') for ep in ['before','after']]
            sparse=[self.logits(p[ep],p['query']) for ep in ['before','after']]
            if scalar:
                ee=np.stack([(log_odds(torch.from_numpy(s.copy()),p['gold']).double()-log_odds(torch.from_numpy(np.array(d)),p['gold']).double()).numpy() for s,d in zip(sparse,dense)])
                a=float(np.mean(ee**2));c=float(np.mean((ee[1]-ee[0])**2));metrics=dict(A=a,C=c,AC=a+c)
            else:metrics=vector_pair(sparse,dense,self.m['memory']['chunk'])
            if any(not np.isfinite(float(metrics.get(k,float('nan')))) for k in ('A','C','AC')):raise ValueError('Nonfinite pair metric')
            row=dict(fingerprint=fp,pair_sha256=digest(p),metrics=metrics)
            freeze_checked(path,row);rows.append(metrics)
            self.progress('scalar_pairs' if scalar else 'Vector_pairs',completed=i+1,total=len(b['pairs']),candidate=label,split=split)
            del sparse,dense
        return dict(mean=mean_rows(rows),rows=rows)
    def score(self,family,split,label,identity):
        if family=='Square':
            pred=self.scalar_square(split,label,identity)
            bank=self.bank('Square',split)
            dense_row=read_checked(ROOT/'readouts/Square/dense'/f'{split}.json', {'fingerprint':self.fp('Square',split,'dense'), 'count':bank['states'], 'width':8})
            result=square_metrics(pred,np.asarray(dense_row['values']).reshape(-1,4,8))
        else:result=self.pair_score(split,label,identity)
        freeze_checked(ROOT/'metrics'/family/label/f'{split}.json',dict(fingerprint=self.fp(family,split,identity),**result))
        return result
    def original_block(self,block):
        if self._weight_keys is None:
            from safetensors import safe_open
            self._weight_keys={name:(entry['path'],entry['key']) for name,entry in self._checkpoint_map.items()}
        from safetensors import safe_open
        out={}
        for r in self.rt.refs[block*7:block*7+7]:
            path,key=self._weight_keys[r['name']]
            with safe_open(path,framework='pt',device='cpu') as f:
                tensor=f.get_tensor(key);meta=self._checkpoint_map[r['name']]
                dtype_tokens={'F16':'float16','BF16':'bfloat16','F32':'float32','F64':'float64','I64':'int64','I32':'int32','I16':'int16','I8':'int8','U8':'uint8'}
                actual_dtype=str(tensor.dtype).split('.')[-1]
                expected_dtype=dtype_tokens.get(str(meta['dtype']).upper().split('.')[-1],str(meta['dtype']).split('.')[-1])
                if list(tensor.shape)!=meta['shape'] or actual_dtype!=expected_dtype:raise ValueError(f'Checkpoint tensor identity: {r["name"]}')
                out[r['name']]=tensor
        return out
    def build(self,label,counts):
        path=ROOT/'candidates'/label/'mask_manifest.json'
        if path.exists():
            m=read(path)
            if m['config_sha256']!=self.ch or [r['selected_mask']['prune_per_row'] for r in m['entries']]!=counts:raise ValueError('Candidate manifest counts/config')
            with torch.no_grad():
                for b in range(32):
                    for name,tensor in self.original_block(b).items():self.rt.mapping[name].weight.copy_(tensor)
            self.rt.apply_manifest(m);return m
        entries=[]
        for b in range(32):
            entries+=self.rt.mask_block(b,counts,originals=self.original_block(b),save_folder=path.parent)
            self.progress('build_mask',completed=b+1,total=32,candidate=label)
        total=sum(r['selected_mask']['pruned'] for r in entries)
        if total!=self.m['target']:raise ValueError('Wrong final budget')
        m=dict(config_sha256=self.ch,entries=entries,pruned=total,counts_sha256=digest(counts))
        freeze(path,m);return m
    def generate(self,label,manifest):
        from experiments.wanda_failure_characterization.run_failure_map import model_sha
        identity=mask_identity(manifest);physical=model_sha(self.model)
        freeze(ROOT/'candidates'/label/'model_identity.json',dict(config_sha256=self.ch,mask_identity=identity,sparse_model_sha256=physical))
        req=read(MULTI/'requests.json')['development'];fp=dict(config_sha256=self.ch,requests_sha256=sha(MULTI/'requests.json'),mask_identity=identity,sparse_model_sha256=physical,split='development',protocol_hash=self.m['protocol_hash'])
        task,_,_=task_and_protocol(read(LEGACY/'config.json'));cached=None
        for arm,m in self.rt.config['legacy_manifests'].items():
            if m['identity']==identity:
                if m['sparse_model_sha256']!=physical:raise ValueError('Same mask, wrong model')
                cached=read(MULTI/'cached_development.json')[arm];break
        if cached is None:
            for parent in [ROOT/'candidates',MULTI/'candidates']:
                for ip in parent.glob('*/model_identity.json'):
                    mi=read(ip)
                    if mi['mask_identity']!=identity or mi['sparse_model_sha256']!=physical:continue
                    folder=(ROOT/'gsm8k' if parent==ROOT/'candidates' else MULTI/'gsm8k/development')/ip.parent.name
                    if (folder/'results.json').exists():
                        r=read(folder/'results.json');checked(folder/'predictions.json',r['predictions_sha256'])
                        cached=read_predictions(folder,req,r['fingerprint'],self.m['protocol_hash']);break
                if cached is not None:break
        if cached is None:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(self.m['model']['id'],revision=self.m['model']['revision'],trust_remote_code=True,local_files_only=True)
            gen=generator(self.model,tok,self.m['evaluation'],self.bank('Square','calibration')['mask_id'])
        else:
            byid={r['example_id']:r for r in cached};gen=lambda q:byid[q['example_id']]['generated_text']
            freeze(ROOT/'gsm8k'/label/'cache_reuse.json',dict(mask_identity=identity,rows=100))
        result=evaluate_requests(ROOT/'gsm8k'/label,req,fp,self.m['protocol_hash'],label,gen,lambda q,t:grade(task,q,t),self.progress)
        if model_sha(self.model)!=physical:raise ValueError('Weights changed in generation')
        self.progress('complete',completed=100,total=100,candidate=label,correct=result['correct'])

def collect(w,family,blocks):
    originals={b:w.original_block(b) for b in blocks}
    w.rt.uniform(verify_ranking=True)
    for b in blocks:
        path=ROOT/'probes'/family/f'block{b:02d}.json'
        if path.exists():
            read_checked(path,{'config_sha256':w.ch,'block':b})
            continue
        cond={}
        try:
            for rate in [.48,.52]:
                counts=[int(r['shape'][1]*rate) for r in w.rt.refs]
                entries=w.rt.mask_block(b,counts,originals=originals[b]);ident=digest(dict(uniform=mask_identity(w.rt.base),changed=mask_identity(dict(entries=entries))))
                label=f'probe_{b:02d}_{int(rate*100)}';res=w.score(family,'calibration',label,ident)
                metrics_path=ROOT/'metrics'/family/label/'calibration.json'
                cond[str(rate)]=dict(pruned=sum(r['selected_mask']['pruned'] for r in entries),mask_identity=ident,metrics=res,metrics_path=str(metrics_path),metrics_sha256=sha(metrics_path))
        finally:w.rt.restore_block(b,originals[b])
        lo,hi=cond['0.48'],cond['0.52'];costs={k:(hi['metrics']['mean'][k]-lo['metrics']['mean'][k])/(hi['pruned']-lo['pruned']) for k in ['A','AC']}
        freeze_checked(path,dict(config_sha256=w.ch,family=family,block=b,conditions=cond,costs=costs))
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    if model_sha(w.model)!=w.rt.config['legacy_manifests']['uniform']['sparse_model_sha256']:raise ValueError('Uniform restore failed')


# Kept as a compatibility alias while the controller migrates to allocation.py.
from .allocation import allocate

def exchange(w):
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from experiments.dlm_context_response50.core import distortion

    anchor=read(LEGACY/'allocation.json')['allocations']['AC']
    base=read(w.rt.config['legacy_manifests']['AC']['path'])
    w.rt.apply_manifest(base)
    original=anchor['row_counts']
    if len(original)!=len(w.rt.refs):raise ValueError('Exchange anchor length')
    pairs=read(LEGACY/'pairs.json')['pairs']
    if len(pairs)!=80:raise ValueError('Exchange requires original 80 pairs')
    teacher_spec=w._legacy_teacher
    teacher_path=Path(teacher_spec['path'])
    teacher_sha=sha(teacher_path)
    if teacher_sha!=teacher_spec['sha256']:raise ValueError('Legacy teacher changed')
    bank_path=LEGACY/'pairs.json';bank_sha=sha(bank_path)
    readout='legacy-scalar-v1';reduction='equal-pair-fp64'
    teacher=torch.load(teacher_path,map_location='cpu',weights_only=False)
    if not isinstance(teacher,(list,tuple)) or len(teacher)!=len(pairs):raise ValueError('Legacy teacher pair count')
    for i,p in enumerate(pairs):
        if len(p.get('query',[]))!=len(p.get('gold',[])) or len(p.get('query',[]))==0:raise ValueError(f'Exchange pair query/gold {i}')
        t=torch.as_tensor(teacher[i])
        if tuple(t.shape)!=(2,len(p['query'])) or not torch.isfinite(t).all():raise ValueError(f'Legacy teacher shape/value {i}')

    def finite_metrics(metrics):
        if set(metrics)!={'A','C','AC'} or any(not np.isfinite(float(metrics[k])) for k in metrics):raise ValueError('Nonfinite exchange metrics')
        return metrics
    def fingerprint(identity):
        return dict(config_sha256=w.ch,mask_identity=identity,bank_sha256=bank_sha,teacher_sha256=teacher_sha,readout=readout,reduction=reduction)
    def score(label,identity):
        folder=ROOT/'exchange/readouts'/label;fp=fingerprint(identity);rows=[]
        for i,pair in enumerate(pairs):
            path=folder/f'{i:03d}.json';pair_sha=digest(pair);expected={'fingerprint':fp,'pair_sha256':pair_sha}
            if path.exists():
                row=read_checked(path,expected);rows.append(finite_metrics(row['metrics']));continue
            pp=torch.stack([w.margin(pair[ep],pair['query'],pair['gold']) for ep in ['before','after']])
            if tuple(pp.shape)!=(2,len(pair['query'])) or not torch.isfinite(pp).all():raise ValueError(f'Exchange sparse shape/value {i}')
            value=finite_metrics(distortion(pp,torch.as_tensor(teacher[i])))
            freeze_checked(path,dict(fingerprint=fp,pair_sha256=pair_sha,metrics=value));rows.append(value)
            w.progress('exchange_pairs',completed=i+1,total=len(pairs),candidate=label)
        result=mean_rows(rows);return finite_metrics(result)
    def check_score(label,identity):
        fp=fingerprint(identity);rows=[]
        for i,pair in enumerate(pairs):
            row=read_checked(ROOT/'exchange/readouts'/label/f'{i:03d}.json',{'fingerprint':fp,'pair_sha256':digest(pair)})
            rows.append(finite_metrics(row['metrics']))
        return finite_metrics(mean_rows(rows))
    def validate_counts(counts):
        if len(counts)!=len(w.rt.refs) or any(not isinstance(k,int) for k in counts):raise ValueError('Exchange row counts')
        if any(k<0 or k>r['shape'][1] for k,r in zip(counts,w.rt.refs)):raise ValueError('Exchange row quota bounds')
        if sum(k*r['shape'][0] for k,r in zip(counts,w.rt.refs))!=w.m['target']:raise ValueError('Exchange exact quota')
    def cache_item(path,p,key,physical):
        item=read_checked(path)
        if item.get('cache_key')!=key or item.get('config_sha256')!=w.ch or item.get('bank_sha256')!=bank_sha or item.get('teacher_sha256')!=teacher_sha or item.get('readout')!=readout or item.get('reduction')!=reduction or item.get('physical_sha256')!=physical or item.get('row_counts')!=p['row_counts']:
            raise ValueError('Exchange cache identity')
        finite_metrics(item['metrics']);return item

    statepath=ROOT/'exchange/state.json';ident=mask_identity(base);initial_physical=model_sha(w.model)
    validate_counts(original)
    if not statepath.exists():
        first=score('initial_1',initial_physical);eps=max(1e-6,1e-5*abs(first['AC']))
        freeze_checked(ROOT/'exchange/initial.json',dict(loss=first,epsilon=eps,config_sha256=w.ch,mask_identity=ident,physical_sha256=initial_physical,bank_sha256=bank_sha,teacher_sha256=teacher_sha,readout=readout,reduction=reduction))
        second=score('initial_2',initial_physical)
        if abs(second['AC']-first['AC'])>eps:raise RuntimeError('Exchange repeat outside frozen tolerance')
        state=dict(config_sha256=w.ch,counts=original,mask_identity=ident,physical_sha256=initial_physical,loss=first['AC'],initial_loss=first['AC'],epsilon=eps,round=0,evaluations=0,fresh_evaluations=0,cache_hits=0,remaining_budget=24,accepted=[],done=False,bank_sha256=bank_sha,teacher_sha256=teacher_sha,readout=readout,reduction=reduction)
        write_checked(statepath,state)
    state=read_checked(statepath)
    if state.get('config_sha256')!=w.ch or state.get('bank_sha256')!=bank_sha or state.get('teacher_sha256')!=teacher_sha or state.get('readout')!=readout or state.get('reduction')!=reduction:raise ValueError('Exchange state identity')
    validate_counts(state['counts'])
    initial=read_checked(ROOT/'exchange/initial.json',{'config_sha256':w.ch,'bank_sha256':bank_sha,'teacher_sha256':teacher_sha,'readout':readout,'reduction':reduction})
    if state['initial_loss']!=initial['loss']['AC'] or state['epsilon']!=initial['epsilon']:raise ValueError('Exchange initial evidence mismatch')
    check_score('initial_1',initial['physical_sha256'])
    check_score('initial_2',initial['physical_sha256'])
    state=recover_exchange(ROOT/'exchange',state)
    validate_counts(state['counts'])
    mf=w.build('exchange_resume_'+digest(state['counts'])[:16],state['counts'])
    physical=model_sha(w.model)
    if physical!=state['physical_sha256'] or mask_identity(mf)!=state['mask_identity']:raise ValueError('Exchange incumbent identity mismatch')
    cache_dir=ROOT/'exchange/cache';cache_dir.mkdir(parents=True,exist_ok=True)
    cache_files=list(cache_dir.glob('*.json'));fresh_total=0
    for path in cache_files:
        item=read_checked(path)
        if item.get('fresh_measurement') is not True:raise ValueError('Exchange cache freshness marker')
        if item.get('cache_key')!=path.stem or item.get('config_sha256')!=w.ch or item.get('bank_sha256')!=bank_sha or item.get('teacher_sha256')!=teacher_sha or item.get('readout')!=readout or item.get('reduction')!=reduction:raise ValueError('Exchange cache provenance')
        finite_metrics(item['metrics']);fresh_total+=1
    if fresh_total>24:raise ValueError('Exchange fresh budget exceeded')
    if fresh_total<state.get('fresh_evaluations',0):raise ValueError('Exchange fresh budget regressed')
    state['fresh_evaluations']=fresh_total;state['evaluations']=fresh_total;state['remaining_budget']=24-fresh_total
    # Reconcile validated orphan cache evidence before writing the next round journal.
    # This keeps the journal's `before` state byte-for-byte equal to persisted state
    # if a prior process died after a cache commit but before its round commit.
    write_checked(statepath,state)
    def apply_counts(counts,prior):
        for b in range(32):
            if counts[b*7:b*7+7]!=prior[b*7:b*7+7]:w.rt.mask_block(b,counts,originals=w.original_block(b))
    while not state['done'] and state['round']<3:
        offered=proposals(state['counts'],original,w.rt.refs,anchor['scores'])[:8]
        freeze_checked(ROOT/'exchange'/f'proposals{state["round"]}.json',dict(config_sha256=w.ch,incumbent=state['mask_identity'],offers=offered))
        values=[];items=[];current=state['counts'];hits=0;fresh=0
        for p in offered:
            key=digest(p['row_counts']);cache=cache_dir/f'{key}.json'
            if not cache.exists() and fresh_total+fresh>=24:break
            apply_counts(p['row_counts'],current);current=p['row_counts'];physical=model_sha(w.model)
            if cache.exists():item=cache_item(cache,p,key,physical);hits+=1
            else:
                loss=score(key,physical);item=dict(**p,cache_key=key,config_sha256=w.ch,physical_sha256=physical,bank_sha256=bank_sha,teacher_sha256=teacher_sha,readout=readout,reduction=reduction,metrics=loss,fresh_measurement=True);freeze_checked(cache,item);fresh+=1
            values.append(item['metrics']['AC']);items.append(item)
        chosen=accept(state['loss'],values,state['epsilon']) if values else None
        nextstate=dict(state);nextstate['round']=state['round']+1;nextstate['cache_hits']=state.get('cache_hits',0)+hits;nextstate['fresh_evaluations']=fresh_total+fresh;nextstate['evaluations']=fresh_total+fresh;nextstate['remaining_budget']=24-nextstate['fresh_evaluations']
        if chosen is None:
            if nextstate['remaining_budget']==0:nextstate.update(done=True,termination='search_budget_exhausted')
            else:nextstate.update(done=True,termination='no_feasible_proposals' if not items else 'no_improvement_among_offered')
        else:
            item=items[chosen];nextstate.update(counts=item['row_counts'],loss=values[chosen],accepted=state['accepted']+[dict(round=state['round'],donor=item['donor'],receiver=item['receiver'],measured_gain=state['loss']-values[chosen],predicted_gain=item['predicted_gain'])])
            apply_counts(nextstate['counts'],current);nextmf=w.build('exchange_resume_'+digest(nextstate['counts'])[:16],nextstate['counts']);nextstate.update(mask_identity=mask_identity(nextmf),physical_sha256=model_sha(w.model))
        if nextstate['round']==3 and not nextstate['done']:nextstate.update(done=True,termination='round_budget_exhausted')
        state=commit_exchange(ROOT/'exchange',state,items,nextstate);fresh_total=state['fresh_evaluations']
        if chosen is None:apply_counts(nextstate['counts'],current)
    apply_counts(original,state['counts']);w.pair_score('diagnostic','Exchange-anchor',ident,scalar=True)
    apply_counts(state['counts'],original);mf=w.build('Exchange-AC',state['counts'])
    if state['counts']==original:freeze_checked(ROOT/'exchange/diagnostic_reuse.json',dict(source='Exchange-anchor',unchanged=True))
    else:w.pair_score('diagnostic','Exchange-AC',mask_identity(mf),scalar=True)
    w.generate('Exchange-AC',mf)

def record_stage(method):
    import functools
    @functools.wraps(method)
    def wrapped(self,*args,**kwargs):
        start=time.monotonic();calls=self.calls
        try:return method(self,*args,**kwargs)
        finally:
            write(ROOT/'stage_costs'/f'{self.label}_{method.__name__}_{time.time_ns()}.json',dict(job=self.label,stage=method.__name__,arguments=[str(x) for x in args if isinstance(x,str)],forward_calls=self.calls-calls,wall_seconds=time.monotonic()-start))
    return wrapped
for _name in ['teacher_vectors','scalar_square','pair_score','generate']:
    setattr(Worker,_name,record_stage(getattr(Worker,_name)))

def main(job):
    label=job['id'];w=Worker(label)
    try:
        kind=job['kind'];family=job.get('family')
        if kind=='smoke':
            q=w.bank('Square','calibration')['quartets'][0];n=q['nodes'][0]
            a=w.margin(n['input_ids'],q['query'],q['gold']);b=w.margin(n['input_ids'],q['query'],q['gold'])
            if not torch.equal(a,b):raise RuntimeError('Dense repeat mismatch')
            original=w.original_block(0);w.rt.uniform(verify_ranking=True)
            baseline=w.margin(n['input_ids'],q['query'],q['gold'])
            counts=[int(r['shape'][1]*.48) for r in w.rt.refs]
            w.rt.mask_block(0,counts,originals=original)
            w.margin(n['input_ids'],q['query'],q['gold'])
            w.rt.restore_block(0,original)
            restored=w.margin(n['input_ids'],q['query'],q['gold'])
            if not torch.equal(baseline,restored):raise RuntimeError('Physical restore mismatch')
            if w.calls>32:raise RuntimeError('Smoke cap exceeded')
            freeze(ROOT/'smoke.json',dict(config_sha256=w.ch,status='passed',forward_calls=w.calls,uniform_reproduced_224=True,physical_restore_exact=True))
        elif kind=='teacher':
            if family=='Square':w.scalar_square(job['split'],'dense','dense')
            else:w.teacher_vectors(job['split'])
        elif kind=='uniform':w.rt.uniform(verify_ranking=True);w.score(family,'calibration','uniform',mask_identity(w.rt.base))
        elif kind=='probe':collect(w,family,job['blocks'])
        elif kind=='candidate':
            counts=load_allocation(family)['allocations'][job['objective']]['row_counts'];mf=w.build(label,counts)
            for split in ['calibration','diagnostic']:w.score(family,split,label,mask_identity(mf))
            w.generate(label,mf)
        elif kind=='exchange':exchange(w)
        else:raise ValueError(kind)
        w.receipt('complete')
    except BaseException as e:w.receipt('failed',str(e));raise
