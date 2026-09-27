"""Pure CPU mathematical contracts; no model or CUDA imports."""
import math
import random
import numpy as np
from experiments.dlm_multiscale_ac50.core import seed_for, clean_sequences

ARMS=('MS-A','Short','Path','All','Multi','Square-A','Square-AC','Vector-A','Vector-AC','Exchange-AC')
CONTRASTS=(('Multi','Short'),('Multi','MS-A'),('Multi','Path'),('Multi','All'),('Square-AC','Square-A'),('Vector-AC','Vector-A'),('Exchange-AC','legacy_AC'))
EDGES=((0,1),(0,2),(1,3),(2,3))

def square_bank(source,split):
    seed=20260926 if split=='calibration' else 20260927
    rows=[]
    for idx,clean in sorted(clean_sequences(source).items()):
        qs=seed_for(seed,'square-query',idx)
        query=sorted(random.Random(qs).sample(range(256),8))
        eligible=[i for i in range(256) if i not in query]
        for j,p in enumerate([.05,.25,.5,.75,.9]):
            bs=seed_for(seed,'square-base',idx,j);gs=seed_for(seed,'square-groups',idx,j)
            rng=random.Random(bs);uu=[rng.random() for _ in eligible]
            visible=[i for i,u in zip(eligible,uu) if u<=p]
            remaining=sorted(set(eligible)-set(visible));random.Random(gs).shuffle(remaining)
            g=min(13,len(remaining)//2);r1=remaining[:g];r2=remaining[g:2*g]
            nodes=[]
            for k,extra in enumerate([[],r1,r2,r1+r2]):
                v=sorted(visible+extra);vs=set(v)
                ids=[clean[i] if i in vs else source['mask_id'] for i in range(256)]
                nodes.append(dict(state_id=f'{idx}:{j}:{k}',input_ids=ids,visible=v,masked_count=256-len(v)))
            rows.append(dict(sequence_index=idx,quartet_index=j,p=p,query=query,gold=[clean[i] for i in query],query_seed=qs,base_seed=bs,group_seed=gs,uniforms=uu,eligible=eligible,groups=[r1,r2],group_size=g,nodes=nodes))
    result=dict(family='Square',split=split,seed=seed,mask_id=source['mask_id'],quartets=rows,states=4*len(rows))
    validate_square(result)
    return result

def validate_square(bank):
    seen={}
    for r in bank['quartets']:
        q=set(r['query']);a,b=map(set,r['groups']);vv=[set(n['visible']) for n in r['nodes']]
        assert len(q)==8 and not a&b and len(a)==len(b)==r['group_size']
        assert seen.setdefault(r['sequence_index'],q)==q
        assert vv[0]=={i for i,u in zip(r['eligible'],r['uniforms']) if u<=r['p']}
        assert not (a|b)&(q|vv[0]);assert vv==[vv[0],vv[0]|a,vv[0]|b,vv[0]|a|b]
        for n,v in zip(r['nodes'],vv):
            assert not q&v and all(n['input_ids'][i]==bank['mask_id'] for i in q)
            assert sum(x==bank['mask_id'] for x in n['input_ids'])==n['masked_count']

def square_metrics(pred,dense):
    pred=np.asarray(pred,dtype=np.float64);dense=np.asarray(dense,dtype=np.float64)
    if pred.shape!=dense.shape or pred.ndim!=3 or pred.shape[1]!=4 or min(pred.shape)==0 or not np.isfinite(pred).all() or not np.isfinite(dense).all():raise ValueError('Square requires identical finite quartets x 4 x query')
    e=pred-dense
    a=np.mean(e**2,axis=(1,2));c=np.mean(np.stack([(e[:,j]-e[:,i])**2 for i,j in EDGES]),axis=(0,2))
    modes=np.einsum('kn,bnq->bkq',np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])/4,e)
    rows=[]
    for i in range(len(e)):
        m=np.mean(modes[i]**2,axis=1)
        rows.append(dict(A=float(a[i]),C=float(c[i]),AC=float(a[i]+c[i]),common2=float(m[0]),main1_2=float(m[1]),main2_2=float(m[2]),interaction2=float(m[3]),mixed2=float(16*m[3])))
    return dict(mean=mean_rows(rows),rows=rows)

def mean_rows(rows):
    if not rows:raise ValueError('empty reductions')
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}

def vector_pair(pred,dense,chunk=4096):
    """Raw FP32 inputs; FP64 full-vocabulary means, then streamed FP64 residuals."""
    shape=pred[0].shape
    if len(shape)!=2 or any(x.shape!=shape for x in [*pred,*dense]) or not all(np.isfinite(x).all() for x in [*pred,*dense]):raise ValueError('Invalid vector readout')
    q,v=shape
    means=[np.mean(x,axis=1,dtype=np.float64) for x in [*pred,*dense]]
    aa=cc=0.
    for j in range(0,v,chunk):
        e0=(np.asarray(pred[0][:,j:j+chunk],dtype=np.float64)-means[0][:,None])-(np.asarray(dense[0][:,j:j+chunk],dtype=np.float64)-means[2][:,None])
        e1=(np.asarray(pred[1][:,j:j+chunk],dtype=np.float64)-means[1][:,None])-(np.asarray(dense[1][:,j:j+chunk],dtype=np.float64)-means[3][:,None])
        aa+=float(np.sum(e0**2)+np.sum(e1**2));cc+=float(np.sum((e1-e0)**2))
    a=aa/(2*q*v);c=cc/(q*v)
    return dict(A=a,C=c,AC=a+c)

def exchange_bounds(anchor,refs):
    return [(min(k,math.ceil(.45*r['shape'][1])),max(k,math.floor(.55*r['shape'][1]))) for k,r in zip(anchor,refs)]

def proposals(counts,anchor,refs,costs):
    bounds=exchange_bounds(anchor,refs);result=[]
    increments=[41*(r['shape'][1]//4096) for r in refs]
    for b in range(32):
        if sorted(r['shape'][1] for r in refs[7*b:7*b+7])!=[4096]*6+[12288]:raise ValueError('Exchange shapes changed')
        if sum(increments[i]*refs[i]['shape'][0] for i in range(b*7,b*7+7))!=53248*41:raise ValueError('Exchange quantum changed')
    for donor in range(32):
        for receiver in range(32):
            if donor==receiver:continue
            out=list(counts)
            for b,sign in [(donor,1),(receiver,-1)]:
                for i in range(b*7,b*7+7):out[i]+=sign*increments[i]
            if all(lo<=k<=hi for k,(lo,hi) in zip(out,bounds)):
                result.append(dict(donor=donor,receiver=receiver,row_counts=out,predicted_gain=53248*41*(costs[receiver]-costs[donor])))
    return sorted(result,key=lambda x:(-x['predicted_gain'],x['donor'],x['receiver']))

def accept(loss,measurements,epsilon):
    if not math.isfinite(loss) or any(not math.isfinite(v) for v in measurements):raise ValueError('Nonfinite exchange loss')
    if not measurements:return None
    i=min(range(len(measurements)),key=lambda i:measurements[i])
    return i if measurements[i]<loss-epsilon else None
