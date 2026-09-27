"""Independent coverage/receipts and rank-correlation verification after collection."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.dlm_dual_role_allocation.io import file_sha256 as sha, atomic_write_json as write
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
c=read(ROOT/'config.json')
for p,h in c['sources'].items():assert sha(p)==h,p
manifest=read(next(p for p in c['sources'] if p.endswith('candidate_mask_manifest.json')))['entries']
cache=read(next(p for p in c['sources'] if p.endswith('lsa/statistics.json')))['rows']
cal=read(next(p for p in c['sources'] if p.endswith('calibration_state_manifest.json')))['states']
clean=[r for b in range(32) for r in read(ROOT/'clean'/f'{b:02d}.json')['rows']]
assert [r['name'] for r in clean]==[r['name'] for r in cache]==[r['name'] for r in manifest]
records=[read(ROOT/'damage'/f'{i:03d}.json') for i in range(224)]
for r,m in zip(records,manifest):
    assert r['config_sha256']==sha(ROOT/'config.json') and r['name']==m['name']
    assert r['mask_sha256']==m['masks'][0]['mask_sha256']
    assert abs(r['physical_hook_delta'])<1e-6 and abs(r['restoration_delta'])<1e-7
    assert len(r['delta_ce'])==80 and np.isfinite(r['delta_ce']).all()
a=np.array([r['metric'] for r in clean]);b=np.array([r['metric'] for r in cache]);y=np.array([r['delta_ce'] for r in records]);target=y.mean(1)
rho=lambda x,z:float(np.corrcoef(rankdata(x),rankdata(z))[0,1])
result=read(ROOT/'results.json')
for gran in ('projection','block'):
    x,z,t=a,b,target
    if gran=='block':x=abs(x.reshape(32,7).mean(1));z=abs(z.reshape(32,7).mean(1));t=t.reshape(32,7).mean(1)
    for key,v in [('rho_clean',rho(x,t)),('rho_corrupted',rho(z,t)),('rho_proxy_clean_corrupted',rho(x,z))]:assert abs(result[gran][key]-v)<1e-12
    assert abs(result[gran]['rho_difference']-(rho(z,t)-rho(x,t)))<1e-12
# Add descriptive allocation changes at original basic LSA settings, without building masks.
from experiments.dlm_allocation_baselines65.core import lsa_rates
ca,cb=lsa_rates(a),lsa_rates(b)
extra=dict(clean_vs_corrupted_ideal_block_sparsity_mean_absolute_difference_pp=float(abs(ca-cb).mean()*100),max_absolute_difference_pp=float(abs(ca-cb).max()*100),corrupted_ideal_block_sparsities=cb.tolist(),clean_ideal_block_sparsities=ca.tolist(),negative_damage_fraction=float((y<0).mean()),module_mean_negative_count=int((target<0).sum()))
paths=[ROOT/'config.json',ROOT/'results.json',ROOT/'dense_losses.json',*sorted((ROOT/'clean').glob('*.json')),*sorted((ROOT/'damage').glob('*.json'))]
receipt=dict(status='verified',projection_count=224,state_count=80,forward_hook_vs_physical_checks=224,restoration_checks=224,sources={str(p):sha(p) for p in paths},descriptive=extra)
write(ROOT/'verification.json',receipt);print(json.dumps(dict(status='verified',results=result,descriptive=extra),indent=2))
