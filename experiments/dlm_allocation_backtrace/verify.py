"""Independent receipt/rank verification plus cheap spectral-concentration summary."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import rankdata
from experiments.dlm_dual_role_allocation.io import file_sha256 as sha,atomic_write_json as write
ROOT=Path(__file__).resolve().parent; EXP=ROOT.parent
read=lambda p:json.loads(Path(p).read_text())
def rho(a,b):return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])

def main():
    torch.set_num_threads(8)
    r=read(ROOT/'results.json');c=read(ROOT/'collection.json')
    for p,h in r['sources'].items():assert sha(p)==h,p
    assert len(c['features'])==224 and len(c['owl_blocks'])==32
    for item in c['features']:
        assert abs(item['mean_score']/(item['mean_abs_weight']*item['mean_channel_rms']*item['alignment'])-1)<1e-12
    proxies={k:v['values'] for k,v in r['dlp']['counterfactuals'].items()}
    for field in ('dlp','lsa'):
        for name,p in r[field]['counterfactuals'].items():assert abs(p['depth_rho']-rho(np.arange(32),p['values']))<1e-12
    for k,v in r['actual_allocations'].items():
        if k!='uniform':assert abs(v['profile']['depth_rho']-rho(np.arange(32),v['profile']['values']))<1e-12
    # Existing verification already checked the 224x80 physical intervention data.
    cfg=read(EXP/'dlm_lsa_mask_mismatch/config.json')
    cal=read(next(p for p in cfg['sources'] if p.endswith('calibration_state_manifest.json')))['states']
    sequences=np.array([s['sequence_index'] for s in cal])
    target=np.array([read(EXP/'dlm_lsa_mask_mismatch/damage'/f'{i:03d}.json')['delta_ce'] for i in range(224)])
    target=target.mean(1).reshape(32,7).mean(1)
    vals={'owl':r['owl']['original_percent']['values'],'dlp':proxies['original'],
          'lsa':r['lsa']['counterfactuals']['raw']['values'],'alpha':r['alpha']['profile']['values']}
    for k,v in vals.items():assert abs(rho(v,target)-r['block_proxy_vs_signed_ce_damage'][k]['rho'])<1e-12
    # λmax is already stored by AlphaPruning; Frobenius energy needs only a CPU sum.
    snap=Path('/home/tmluser1/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2')
    index=read(snap/'model.safetensors.index.json')['weight_map']
    ar=read(EXP/'dlm_allocation_baselines65/alpha/statistics.json')['rows'];spectral=[]
    for i,a in enumerate(ar):
        b=i//7;t=a['name'].split('.')[1];key=f'model.transformer.blocks.{b}.{t}.weight'
        with safe_open(snap/index[key],framework='pt',device='cpu') as f:w=f.get_tensor(key).float()
        energy=float(w.double().norm().square());largest=a['spectral_norm'];stable=energy/largest
        assert stable>=1-1e-5
        spectral.append(dict(name=a['name'],alpha=a['alpha'],frobenius_energy=energy,
                             lambda_max=largest,stable_rank=stable,top_mode_energy_fraction=1/stable))
        if i%28==0:print('spectral energy',i,'/224',flush=True)
    alpha=np.array([a['alpha'] for a in spectral]);stable=np.array([a['stable_rank'] for a in spectral])
    extra=dict(projection_alpha_vs_stable_rank=rho(alpha,stable),
        block_alpha_vs_stable_rank=rho(alpha.reshape(32,7).mean(1),stable.reshape(32,7).mean(1)),
        stable_rank_block_quartiles=stable.reshape(4,8,7).mean((1,2)).tolist(),
        top_mode_fraction_block_quartiles=(1/stable).reshape(4,8,7).mean((1,2)).tolist(),
        by_type={spectral[j]['name'].split('.')[1]:dict(alpha_vs_stable_rank=rho(alpha[j::7],stable[j::7]),
            stable_rank_vs_depth=rho(stable[j::7],np.arange(32))) for j in range(7)},
        limitations='Top-mode fraction and stable rank do not identify full tail shape or prove redundancy/pruning tolerance.')
    write(ROOT/'spectral_concentration.json',dict(rows=spectral,summary=extra))
    write(ROOT/'verification.json',dict(status='verified',source_hash_count=len(r['sources']),
        factor_identity=True,independent_rank_checks=True,spectral_summary=extra,
        output_hashes={p.name:sha(p) for p in (ROOT/'results.json',ROOT/'collection.json',ROOT/'projection_features.csv',ROOT/'spectral_concentration.json')},
        verifier_sha256=sha(__file__)))
    print(json.dumps(extra,indent=2),flush=True)

if __name__=='__main__':main()
