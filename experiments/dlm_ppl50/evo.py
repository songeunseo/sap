"""Commit-pinned FastOBC and EvoPress level transfers, adapted to masked DLM KL."""
import ast
import copy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.modules.conv import _ConvNd

from experiments.dlm_ppl50 import run as experiment
from experiments.dlm_ppl50 import sequential as seq
from experiments.dlm_wikitext_ppl import run as protocol

ROOT=Path(__file__).resolve().parent
UP=ROOT/'upstream/evopress'
STORE=Path('/DATA/tmluser1/dlm-ppl50/evopress')
read,write,sha=seq.read,seq.write,seq.sha


def load_helper(name):
    spec=importlib.util.spec_from_file_location('evopress_'+name,UP/'src'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fast_obc():
    # Execute the unmodified upstream class with isolated utility bindings; avoid
    # importing AR-model wrappers or colliding with DSA's module namespace.
    env=dict(torch=torch,np=np,nn=torch.nn,dist=torch.distributed,Tensor=torch.Tensor,
             List=list,_ConvNd=_ConvNd,dist_utils=load_helper('dist_utils'),
             linalg_utils=load_helper('linalg_utils'))
    tree=ast.parse((UP/'src/model_utils.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_number_of_rows_and_cols')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(UP/'src/model_utils.py'),'exec'),env)
    env['model_utils']=SimpleNamespace(get_number_of_rows_and_cols=env['get_number_of_rows_and_cols'])
    tree=ast.parse((UP/'src/fast_obc.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FastOBC')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(UP/'src/fast_obc.py'),'exec'),env)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return env['FastOBC']


def transfers(parent,rng,settings):
    child=copy.deepcopy(parent)
    flips=min(rng.randint(1,3),rng.randint(1,3))
    for _ in range(flips):
        while True:
            down=rng.randint(0,len(child)-1)
            if abs(child[down]-1)<=settings['num_levels']:break
        while True:
            up=rng.randint(0,len(child)-1)
            if abs(child[up]+1)<=settings['num_levels']:break
        child[down]-=1;child[up]+=1
    if sum(child)!=0:raise RuntimeError('equal count transfer violated')
    return child


def apply_levels(mapping,refs,levels,current=None):
    for i,(ref,level) in enumerate(zip(refs,levels)):
        if current is None or level!=current[i]:
            weight=torch.load(STORE/ref['name']/f'{level}.pth',map_location=mapping[ref['name']].weight.device,weights_only=True)
            if tuple(weight.shape)!=tuple(ref['shape']):raise RuntimeError('weight shape mismatch')
            mapping[ref['name']].weight.copy_(weight)
    return list(levels)


@torch.inference_mode()
def database(model,mapping,refs,states,cfg,event):
    settings=cfg['evopress'];FastOBC=fast_obc()
    device=next(model.parameters()).device
    for block in range(32):
        path=ROOT/'evopress/database'/f'block_{block:02d}.json'
        blockrefs=refs[block*7:block*7+7]
        if path.exists():
            rows=read(path)
            for row,ref in zip(rows,blockrefs):
                if row['name']!=ref['name']:raise RuntimeError('database ordering changed')
                for entry in row['levels']:
                    if sha(entry['path'])!=entry['file_sha256']:raise RuntimeError('database hash mismatch')
                mapping[ref['name']].weight.copy_(torch.load(STORE/ref['name']/'0.pth',map_location=device,weights_only=True))
            continue
        handles={r['name']:FastOBC(mapping[r['name']],rel_damp=.01,block_size=128) for r in blockrefs}
        hooks=[]
        for name,handle in handles.items():
            hooks.append(mapping[name].register_forward_hook(lambda mod,inp,out,h=handle:h.update(inp[0])))
        def stop(mod,inp,out):raise seq.PrefixDone()
        hooks.append(model.model.transformer.blocks[block].register_forward_hook(stop))
        try:
            for i,state in enumerate(states):
                try:model(torch.tensor(state['noisy_ids'],device=device))
                except seq.PrefixDone:pass
                event('evopress','database_calibration',block=block,completed=i+1,total=len(states))
        finally:
            for hook in hooks:hook.remove()
        rows=[];baseline=[]
        for ref in blockrefs:
            name=ref['name'];handle=handles.pop(name)
            sparsities=[.5+l*settings['weights_diff']/ref['weights'] for l in range(-8,9)]
            event('evopress','FastOBC',block=block,module=name,levels=17)
            weights=handle.prune(sparsities);entries=[]
            for level,w in zip(range(-8,9),weights):
                dest=STORE/name/f'{level}.pth';dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    saved=torch.load(dest,map_location=device,weights_only=True)
                    if not torch.equal(saved,w):raise RuntimeError('partial FastOBC weight mismatch')
                else:
                    temp=dest.with_suffix('.tmp');torch.save(w.cpu(),temp);temp.replace(dest)
                entries.append(dict(level=level,path=str(dest),file_sha256=sha(dest),
                                    nominal_sparsity=sparsities[level+8],actual_zeros=int((w==0).sum())))
            baseline.append((name,weights[8]))
            rows.append(dict(name=name,levels=entries));handle.reset();del weights,handle
        # Explicit LLaDA adaptation: later database blocks see uniform50 sparse
        # prefixes, replayed by native full forwards (no causal block wrapper).
        for name,w in baseline:mapping[name].weight.copy_(w)
        write(path,rows);event('evopress','database_block_complete',completed=block+1,total=32)


@torch.inference_mode()
def kl_fitness(model,states,target,indices):
    dev=next(model.parameters()).device;values=[]
    for index in indices:
        state=states[index];mask=torch.tensor(state['mask'],device=dev,dtype=torch.bool)
        logits=model(torch.tensor(state['noisy_ids'],device=dev)).logits[mask].float()
        qlog=torch.log_softmax(logits,-1)
        # Only masked positions carry the DLM denoising distribution.
        plog=target[index].to(dev).float()
        value=(plog.exp()*(plog-qlog)).sum(-1).mean()
        if not torch.isfinite(value):raise RuntimeError('nonfinite masked KL')
        values.append(float(value))
    return float(np.mean(values))


@torch.inference_mode()
def build(model,mapping,refs,states,cfg,event):
    devstates=read(seq.DEV)['states'];device=next(model.parameters()).device
    # Dense teacher is computed before database construction sparsifies prefixes.
    event('evopress','dense_teacher',total=len(devstates))
    target=[]
    for i,state in enumerate(devstates):
        mask=torch.tensor(state['mask'],device=device,dtype=torch.bool)
        logits=model(torch.tensor(state['noisy_ids'],device=device)).logits[mask].float()
        target.append(torch.log_softmax(logits,-1).cpu())
        event('evopress','dense_teacher',completed=i+1,total=len(devstates))
    database(model,mapping,refs,states,cfg,event)
    settings=cfg['evopress'];rng=random.Random(settings['seed'])
    path=ROOT/'evopress/search.json'
    history=read(path) if path.exists() else dict(generations=[],initial_parent=[0]*len(refs),
                                                controller='upstream elitist level-transfer mutations and staged selection')
    parent=history['generations'][-1]['winner'] if history['generations'] else history['initial_parent']
    if history['generations']:rng.setstate(seq.tuples(history['generations'][-1]['rng_after']))
    current=None
    for generation in range(len(history['generations']),settings['generations']):
        pending=ROOT/'evopress/pending.json'
        if pending.exists():
            record=read(pending)
            if record['generation']<generation:pending.unlink();record=None
            elif record['generation']!=generation:raise RuntimeError('pending generation mismatch')
        else:record=None
        if record is None:
            candidates=[]
            while len(candidates)<settings['offspring']:
                child=transfers(parent,rng,settings)
                if child not in candidates:candidates.append(child)
            indices=[]
            for tokens in settings['selection_tokens']:
                selected=[];used=0
                while used<tokens:
                    index=rng.randint(0,len(devstates)-1)
                    if index in selected:continue
                    length=len(devstates[index]['clean_ids'][0])
                    if used+length>tokens:raise RuntimeError('selection budget not state aligned')
                    selected.append(index);used+=length
                indices.append(selected)
            record=dict(generation=generation,candidates=candidates,indices=indices,
                        stage=0,scores=[],rng_after=rng.getstate())
            write(pending,record)
        rng.setstate(seq.tuples(record['rng_after']))
        while record['stage']<3:
            stage=record['stage'];candidates=record['candidates']
            if stage==2 and parent not in candidates:
                candidates.append(parent);write(pending,record)
            scores=record['scores']
            for index in range(len(scores),len(candidates)):
                current=apply_levels(mapping,refs,candidates[index],current)
                value=kl_fitness(model,devstates,target,record['indices'][stage])
                scores.append(value);write(pending,record)
                event('evopress','search',generation=generation+1,generations=settings['generations'],
                      selection_stage=stage+1,completed=index+1,total=len(candidates),masked_kl=value)
            order=np.argsort(scores)[:settings['survivors'][stage]]
            record['candidates']=[candidates[int(i)] for i in order]
            record['survivor_fitness']=[scores[int(i)] for i in order]
            record['scores']=[];record['stage']+=1;write(pending,record)
        parent=record['candidates'][0]
        history['generations'].append(dict(generation=generation,winner=parent,
                                          fitness=record['survivor_fitness'][0],rng_after=record['rng_after']))
        write(path,history);pending.unlink()
    apply_levels(mapping,refs,parent,current)
    entries=[]
    for ref,level in zip(refs,parent):
        block=int(ref['name'].split('.')[0].split('_')[1])
        rows=read(ROOT/'evopress/database'/f'block_{block:02d}.json')
        dbrow=next(r for r in rows if r['name']==ref['name'])
        chosen=next(r for r in dbrow['levels'] if r['level']==level)
        entries.append(dict(name=ref['name'],shape=ref['shape'],weights=ref['weights'],**chosen))
    zeros=sum(r['actual_zeros'] for r in entries)
    return dict(method='EvoPress_FastOBC_DLM50',config_sha256=sha(ROOT/'config.json'),entries=entries,
                nominal_pruned=experiment.TARGET,pruned=zeros,weights=experiment.TARGET*2,
                actual_sparsity=zeros/(experiment.TARGET*2),levels=parent)


def apply_result(model,mapping,manifest):
    for row in manifest['entries']:
        if sha(row['path'])!=row['file_sha256']:raise RuntimeError('EvoPress weight changed')
        w=torch.load(row['path'],map_location=mapping[row['name']].weight.device,weights_only=True)
        if int((w==0).sum())!=row['actual_zeros']:raise RuntimeError('EvoPress zero count changed')
        mapping[row['name']].weight.copy_(w)
