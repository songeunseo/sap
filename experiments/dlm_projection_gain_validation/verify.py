"""Independent arithmetic/coverage checks of completed physical measurements."""
import json
from pathlib import Path
import numpy as np
from experiments.dlm_projection_gain_validation.run import validate,ROOT,prior
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write,file_sha256 as sha

def read(p):return json.loads(Path(p).read_text())
def main():
    cfg=validate();result=read(ROOT/'results.json');assert result['status']=='complete'
    assert read(ROOT/'progress.json')['stage']=='complete'
    curves=read(ROOT/'curves.json');assert curves['all_224_masks50_reproduced']
    assert curves['config_sha256']==sha(ROOT/'config.json')
    ref=read(prior.MAN)['entries'];D=np.stack([read(prior.ROOT/'damage'/f'{i:03d}.json')['delta_ce'] for i in range(224)])
    states=read(prior.CAL)['states'];q=np.array([s['sequence_index'] for s in states]);rng=np.random.default_rng(0);draws=rng.integers(0,8,(2000,8));Y={};sources={}
    for i,row in enumerate(curves['rows']):
        assert row['name']==ref[i]['name'];m=np.load(ROOT/f'marginal_{i:03d}.npy');assert len(m)==ref[i]['shape'][1] and np.isfinite(m).all() and (np.diff(m)>=-1e-10).all()
        for s in [.3,.5,.7]:assert np.isclose(m[:int(len(m)*s)].sum(),row['E'][str(s)],rtol=1e-12)
        assert np.isclose(row['g']*row['E']['0.5'],D[i].mean(),rtol=1e-12,atol=1e-14)
    for j in cfg['jobs']:
        p=ROOT/'measurements'/f"{j['id']}.json";a=read(p);assert a['job']==j and a['config_sha256']==sha(ROOT/'config.json') and a['physical_pruning'];assert abs(a['restoration_delta'])<1e-7
        Y[j['id']]=np.array(a['delta_ce']);assert Y[j['id']].shape==(80,) and np.isfinite(Y[j['id']]).all();sources[str(p)]=sha(p)
    for family,n in [('curve',12),('block',6),('pair',3)]:
        assert len(result[family])==n
        for row in result[family]:
            job=next(j for j in cfg['jobs'] if j['id']==row['id']);idx=job['indices'];actual=Y[row['id']]
            if family=='curve':
                i=idx[0];pred=D[i]*curves['rows'][i]['E'][str(job['sparsity'])]/curves['rows'][i]['E']['0.5'];r=row['residual'];target=row['predicted']
            elif family=='block':pred=D[idx].sum(0);r=row['interaction'];target=row['sum_projection']
            else:
                b=idx[0]//7;pred=Y[f'b{b:02d}_50']+Y[f'b{b+1:02d}_50'];r=row['interaction'];target=row['sum_block']
            assert np.isclose(actual.mean(),row['observed']['mean'],atol=1e-12)
            assert np.isclose(pred.mean(),target['mean'],atol=1e-12)
            residual=actual-pred;clusters=np.array([residual[q==s].mean() for s in range(8)]);boot=clusters[draws].mean(1);a=.05/(2*n)
            assert np.allclose(np.quantile(boot,[a,1-a]),r['simultaneous_conditional_ci'],atol=1e-12)
    values=np.array(read(ROOT/'dense_features.json')['values']);assert values.shape==(80,32,2) and np.isfinite(values).all()
    sources.update({str(ROOT/f):sha(ROOT/f) for f in ['config.json','curves.json','results.json','dense_features.json','dense_losses.json','report.md']})
    write(ROOT/'verification.json',dict(status='passed',config_sha256=sha(ROOT/'config.json'),measurements=21,states_per_condition=80,projection_gains=224,arithmetic_and_cluster_ci='independently recomputed',sources=sources))
    print('Independent verification passed')
if __name__=='__main__':main()
