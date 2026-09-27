"""CPU-only exact count/energy audit of already existing pruning masks."""
import csv
import time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from experiments.dlm_distribution_hypotheses.analyze import ROOT,REPO,OLD,read,sha,write
from experiments.dlm_loss_aggregation.core import unpack_mask,mask_sha256

torch.set_num_threads(4)
start=time.time()
stats={r['name']:r for r in read(OLD/'statistics.json')['rows']}
inp=torch.load(read(OLD/'input_collection.json')['path'],map_location='cpu',weights_only=False)['corrupted']
snap=Path('/home/tmluser1/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2')
index=read(snap/'model.safetensors.index.json')['weight_map']
sources=read(REPO/'experiments/dlm_allocation_backtrace/results.json')['sources']
verified={}
for fn in sorted(set(index.values())):
    p=snap/fn;h=sha(p);assert h==sources[str(p)];verified[str(p)]=h
    print('verified',fn,flush=True)
paths={'rowwise':REPO/'experiments/dlm_ppl50/uniform/mask_manifest.json',
       'layer_global':REPO/'experiments/dlm_ppl50_uniform_layer/mask_manifest.json'}
manifests={m:{r['name']:r for r in read(p)['entries']} for m,p in paths.items()}
rows=[]
for i,(name,r) in enumerate(stats.items()):
    key=f'model.transformer.blocks.{r["layer"]}.{r["type"]}.weight'
    with safe_open(str(snap/index[key]),framework='pt',device='cpu') as f:
        w=f.get_tensor(key).float().numpy()
    a=inp[name]['mean_energy'].float().numpy()
    score=np.abs(w)*np.sqrt(a)[None,:]
    mean=float(score.mean(dtype=np.float64));assert np.isclose(mean,r['score_dense']['mean_abs'],rtol=2e-6)
    tail=score>5*mean
    energy=score.astype(np.float64)**2;total=float(energy.sum())
    assert np.isclose(np.sqrt(total/score.size),r['score_dense']['rms'],rtol=2e-6)
    assert np.isclose(float(tail.mean()),r['score_dense']['outlier5_fraction'],atol=1e-12)
    for method,manifest in manifests.items():
        receipt=manifest[name]['selected_mask'];p=Path(receipt['path'])
        assert sha(p)==receipt['file_sha256']
        payload=torch.load(p,map_location='cpu',weights_only=False)
        assert mask_sha256(payload)==receipt['mask_sha256']
        mask=unpack_mask(payload).numpy();assert mask.shape==w.shape and int(mask.sum())==receipt['pruned']
        row_s=mask.mean(1)
        if method=='rowwise':assert np.all(row_s==.5)
        rows.append(dict(name=name,layer=r['layer'],type=r['type'],method=method,weights=mask.size,pruned=int(mask.sum()),
            tail_count=int(tail.sum()),tail_pruned=int((mask&tail).sum()),
            tail_pruned_fraction=float((mask&tail).sum()/tail.sum()),
            removed_diagonal_energy_fraction=float(energy[mask].sum()/total),
            tail_diagonal_energy_fraction=float(energy[tail].sum()/total),
            sparsity=float(mask.mean()),row_sparsity_min=float(row_s.min()),row_sparsity_max=float(row_s.max()),
            row_fraction_over75=float(np.mean(row_s>.75)),row_fraction_over90=float(np.mean(row_s>.9))))
    del w,score,energy,tail,mask
    print('completed',i+1,'/224',name,flush=True)
for method in manifests:
    assert sum(r['pruned'] for r in rows if r['method']==method)==3489660928
with (ROOT/'mask_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for method in manifests:
    rr=[r for r in rows if r['method']==method]
    summary[method]=dict(total_tail_count=sum(r['tail_count'] for r in rr),total_tail_pruned=sum(r['tail_pruned'] for r in rr),
        tail_pruned_fraction=sum(r['tail_pruned'] for r in rr)/sum(r['tail_count'] for r in rr),
        modules_with_any_tail_removed=sum(r['tail_pruned']>0 for r in rr),
        per_type={t:{k:[float(np.mean([r[k] for r in rr if r['type']==t and r['layer']//8==q])) for q in range(4)]
                       for k in ['removed_diagonal_energy_fraction','tail_pruned_fraction','sparsity','row_fraction_over75','row_fraction_over90']}
                  for t in sorted({r['type'] for r in rr})})
write('mask_audit.json',dict(status='complete',summary=summary,elapsed_seconds=time.time()-start,
    checkpoint_sha256=verified,manifest_sha256={str(p):sha(p) for p in paths.values()},script_sha256=sha(Path(__file__)),
    definition='Outlier threshold and diagonal energy use common original DENSE activation statistics. Both masks use their own historical sparse-prefix calibration. These are not exact reconstructions or causal attributions of the policy performance gap.'))
print(summary,flush=True)
print('DONE',time.time()-start,flush=True)
