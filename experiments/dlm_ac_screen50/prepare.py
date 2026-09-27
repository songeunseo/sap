"""Freeze scientific inputs without loading a model or polling GPUs."""
from pathlib import Path
import shutil
import numpy as np
from experiments.dlm_multiscale_ac50.artifacts import DEFAULT_ROOT as MULTI, LEGACY, REPO, read,write,freeze,sha,checked,digest,mask_identity
from experiments.dlm_multiscale_ac50.core import clean_sequences,metrics,rank_rates
from experiments.dlm_multiscale_ac50.evaluation import read_predictions,validate_prediction
from .core import ARMS,CONTRASTS,square_bank
ROOT=Path(__file__).resolve().parent/'output'
SPEC=REPO/'openspec/changes/parallel-ac-mini100-screen'

def legacy_audit():
    from experiments.dlm_multiscale_ac50.prepare import validate
    c=validate(MULTI);sources={};out={}
    def bind(p):sources[str(p)]=sha(p)
    for p in [MULTI/'config.json',MULTI/'allocation.json',MULTI/'requests.json',MULTI/'cached_development.json',LEGACY/'pairs.json',LEGACY/'allocation.json',LEGACY/'dense.pt']:bind(p)
    teacher=read(MULTI/'readouts/dense/calibration.json')['values'];bank=read(MULTI/'bank_calibration.json')
    alloc=read(MULTI/'allocation.json')['allocations']
    costs=[]
    for i in range(32):
        p=MULTI/'probes'/f'block{i:02d}.json';r=read(p);bind(p)
        if r['config_sha256']!=sha(MULTI/'config.json'):raise ValueError('Legacy probe config')
        for v in r['conditions'].values():
            checked(v['readout_path'],v['readout_sha256']);bind(Path(v['readout_path']))
            if metrics(read(v['readout_path'])['values'],teacher,bank)!=v['metrics']:raise ValueError('Legacy probe reduction')
        lo,hi=r['conditions']['0.48'],r['conditions']['0.52']
        cc={k:(hi['metrics']['mean'][k]-lo['metrics']['mean'][k])/(hi['pruned']-lo['pruned']) for k in alloc}
        if cc!=r['costs']:raise ValueError('Legacy marginal cost')
        costs.append(cc)
    from experiments.dlm_owl65.core import exact_row_counts
    refs=read(c['legacy_manifests']['uniform']['path'])['entries']
    for arm,a in alloc.items():
        if [x[arm] for x in costs]!=a['scores'] or rank_rates(a['scores']).tolist()!=a['rates']:raise ValueError('Legacy allocation rank')
        counts,budget=exact_row_counts(refs,a['rates'],3489660928)
        if counts!=a['row_counts'] or budget!=a['budget']:raise ValueError('Legacy exact-quota allocation mismatch')
    req=read(MULTI/'requests.json')['development']
    from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol,grade
    task,_,_=task_and_protocol(read(LEGACY/'config.json'))
    for arm,rows in read(MULTI/'cached_development.json').items():
        for r,q in zip(rows,req,strict=True):
            validate_prediction(r,q,c['protocol_hash'])
            if grade(task,q,r['generated_text'])!={k:r[k] for k in ['correct','extracted_answer']}:raise ValueError('Legacy raw answer regrade mismatch')
        out['legacy_'+arm]=dict(correct=sum(r['correct'] for r in rows),total=len(rows))
    if {k:v['correct'] for k,v in out.items()}!={'legacy_uniform':54,'legacy_A':55,'legacy_AC':61}:raise ValueError('Frozen reference scores changed')
    for arm in ['A','Short','Path','All','Multi']:
        p=MULTI/'gsm8k/development'/arm
        if (p/'identity.json').exists():
            fp=read(p/'identity.json')['fingerprint'];rows=read_predictions(p,req,fp,c['protocol_hash'],allow_partial=True)
            mi=read(MULTI/'candidates'/arm/'model_identity.json')
            if fp['config_sha256']!=sha(MULTI/'config.json') or fp['mask_identity']!=mi['mask_identity'] or fp['sparse_model_sha256']!=mi['sparse_model_sha256']:raise ValueError('Legacy prediction physical identity')
            for q in (p/'examples').glob('*.json'):bind(q)
        else:rows=[]
        out['MS-A' if arm=='A' else arm]=dict(correct=sum(r['correct'] for r in rows),total=len(rows))
    return c,dict(sources=sources,scores_at_import=out,legacy_probes=64,allocations=5)

def diagnostic_pairs(cal,diag):
    import torch
    from experiments.dlm_context_response50.core import make_pairs
    probs={s['timestep_index']:s['p_mask'] for s in cal['states']}
    if len(probs)!=10:raise ValueError('Expected ten calibration probabilities')
    states=[]
    seqs=clean_sequences(diag)
    if sorted(seqs)!=list(range(8,16)):raise ValueError('Diagnostic span IDs changed')
    for idx,ids in sorted(seqs.items()):
        for j,p in sorted(probs.items()):
            seed=20260928+10*(idx-8)+j
            mask=torch.rand(256,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)<p
            clean=torch.tensor([ids]);noisy=clean.clone();noisy[0,mask]=diag['mask_id']
            if int(mask.sum())<2:raise ValueError('Degenerate diagnostic mask; do not resample')
            states.append(dict(sequence_index=idx,timestep_index=j,p_mask=p,mask_seed=seed,clean_ids=clean.tolist(),noisy_ids=noisy.tolist()))
    return dict(pairs=make_pairs(states,diag['mask_id'],seed=20260929),mask_seed_base=20260928,reveal_seed=20260929,states=states,split='diagnostic')

def memory_estimate(banks,vocab=126464):
    bytes_by_split={split:sum(2*len(p['query'])*vocab*4+512 for p in b['pairs']) for split,b in banks.items()}
    maxq=max(len(p['query']) for b in banks.values() for p in b['pairs'])
    return dict(teacher_bytes=bytes_by_split,total_teacher_bytes=sum(bytes_by_split.values()),estimated_pair_ram_bytes=4*maxq*vocab*4+4*maxq*4096*8+maxq*vocab*8,vocab=vocab,chunk=4096)

def prepare():
    if (ROOT/'manifest.json').exists():return validate()
    c,imp=legacy_audit();sources=c['plan']['sources']
    cal=read(sources['calibration_source']['path']);diag=read(sources['diagnostic_source']['path'])
    # Existing overlap audit is authoritative; also forbid matching contiguous 32-token sequences.
    cc=clean_sequences(cal);dd=clean_sequences(diag)
    grams=lambda seqs:{tuple(x[i:i+32]) for x in seqs.values() for i in range(len(x)-31)}
    if grams(cc)&grams(dd):raise ValueError('Calibration/diagnostic token interval overlap')
    banks={}
    for split,source in [('calibration',cal),('diagnostic',diag)]:
        b=square_bank(source,split);p=ROOT/'banks'/f'Square_{split}.json';freeze(p,b);banks['Square_'+split]=dict(path=str(p),sha256=sha(p))
    vb={'calibration':read(LEGACY/'pairs.json'),'diagnostic':diagnostic_pairs(cal,diag)}
    banks['Vector_calibration']=dict(path=str(LEGACY/'pairs.json'),sha256=sha(LEGACY/'pairs.json'))
    p=ROOT/'banks/Vector_diagnostic.json';freeze(p,vb['diagnostic']);banks['Vector_diagnostic']=dict(path=str(p),sha256=sha(p))
    mem=memory_estimate(vb)
    from .preflight import checkpoint_contract,storage_check
    storage=storage_check(ROOT,mem)
    refs=read(c['legacy_manifests']['uniform']['path'])['entries']
    physical=checkpoint_contract(c,refs)
    freeze(ROOT/'legacy_import.json',imp)
    armdefs={}
    for a in ARMS:
        family='Square' if a.startswith('Square') else 'Vector' if a.startswith('Vector') else 'Exchange' if a.startswith('Exchange') else 'Multi'
        armdefs[a]=dict(bank_identity=({split:c['banks'][split] for split in ['calibration','diagnostic']} if family=='Multi' else {split:banks[('Vector' if family=='Exchange' else family)+'_'+split] for split in ['calibration','diagnostic']}),sampling_seeds=({'calibration':20260922,'diagnostic':20260923} if family=='Multi' else {'calibration':20260926,'diagnostic':20260927} if family=='Square' else {'calibration':'unchanged legacy bank','diagnostic_mask':20260928,'diagnostic_reveal':20260929}),numerical_tolerance=({'atol':1e-10,'rtol':1e-9} if family=='Vector' else {'acceptance':'max(1e-6,1e-5*abs(L0))'} if family=='Exchange' else {'arithmetic':'FP32 scalar,FP64 residual'}),family=family,objective_version='ac-screen-v1',lambda_response=0 if a in ['MS-A','Square-A','Vector-A'] else 1,definition=f'{SPEC}/design.md',edge_contract=('before-after' if family in ['Vector','Exchange'] else [[0,1],[0,2],[1,3],[2,3]] if family=='Square' else 'frozen-Multi-C1/C2/C4/Path/All'),defaults_provenance='design §0.8 and §3-6; no literature guarantee',readout='centered-emitted-logits-fp64' if family=='Vector' else 'legacy-fp32-gold-logodds-fp64-residual',reduction='equal-pair' if family in ['Vector','Exchange'] else 'equal-span-chain-query' if family=='Multi' else 'equal-span-quartet-query',allocator='bounded-hard-exchange' if family=='Exchange' else 'signed-rank-exact-quota',control=[b for aa,b in CONTRASTS if aa==a])
    references={name:dict(label=label,source=str(MULTI/'cached_development.json'),sha256=sha(MULTI/'cached_development.json'),correct=n,total=100) for name,label,n in [('legacy_Uniform','native row-wise sparse-prefix Uniform-Wanda',54),('legacy_A','legacy scalar pair A-only',55),('legacy_AC','legacy scalar pair A+C',61)]}
    for a,d in armdefs.items():
        f=d['family']
        d['bank']=d['bank_identity'];d['seed']=d['sampling_seeds']
        d['matched_control']=d.pop('control') or ({'MS-A':['Multi'],'Short':['Multi'],'Path':['Multi'],'All':['Multi'],'Square-A':['Square-AC'],'Vector-A':['Vector-AC']}.get(a,[]))
        d['hypothesis']=('Measured local exchanges improve the legacy AC anchor' if f=='Exchange' else 'Response preservation improves the matched A-only candidate' if a.endswith('-AC') else 'Different reveal scales preserve response variation' if f=='Multi' else 'Endpoint preservation control')
        if f=='Multi':
            c1=[[0,1],[2,3],[4,5],[6,7]];c2=[[0,2],[1,3],[4,6],[5,7]];c4=[[0,4],[1,5],[2,6],[3,7]]
            d['edge_contract']={'C1':c1,'C2':c2,'C4':c4,'Path':[[i,i+1] for i in range(7)],'All':[[i,j] for i in range(8) for j in range(i+1,8)]}
        d['edge_weights']=({'C1':.25,'C2':.25,'C4':.25,'Multi_scale':1/3,'Path':1/7,'All':1/28} if f=='Multi' else {'each_edge':.25} if f=='Square' else {'before_after':1.})
        d['reduction_denominators']=({'spans':8,'chains':2,'queries':8,'nodes':8} if f=='Multi' else {'spans':8,'quartets_per_span':5,'queries':8,'A_endpoints':4,'C_edges':4} if f=='Square' else {'pairs':80,'queries':'per-pair query count','A_endpoints':2,'C_edges':1,**({'vocabulary':126464} if f=='Vector' else {})})
        d['source_hashes']={str(SPEC/'design.md'):sha(SPEC/'design.md')}
    compatibility=dict(Multi='Exact frozen old bank, scores, rank mapping and quotas; MS-A is not legacy_A',Square='New branched bank: end-to-end family comparison, not legacy-pair-only ablation',Vector='Original legacy calibration pair/query bytes; new full-vocabulary centered readout',Exchange='Legacy AC anchor and scalar calibration objective; fixed-cost measured search',Uniform='Native row-wise sparse-prefix reference54; cached62 and layer-global Uniform are separate historical pipelines')
    paths=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+[p for p in SPEC.rglob('*.md') if p.name!='tasks.md']
    manifest=dict(**physical,legacy_teacher=dict(path=str(LEGACY/'dense.pt'),sha256=sha(LEGACY/'dense.pt')),references=references,compatibility_mapping=compatibility,dependency_receipt=dict(path=str(MULTI/'config.json'),sha256=sha(MULTI/'config.json'),sources=c['sources']),schema=1,arms=armdefs,contrasts=CONTRASTS,model=c['model'],evaluation=c['evaluation'],protocol_hash=c['protocol_hash'],legacy_config_sha256=sha(MULTI/'config.json'),requests_sha256=sha(MULTI/'requests.json'),banks=banks,source_hashes={str(p.resolve()):sha(p) for p in paths},import_sha256=sha(ROOT/'legacy_import.json'),target=3489660928,total_prunable=6979321856,probe_rates=[.48,.52],memory=mem,exchange=dict(d=41,rounds=3,proposals=8,max_new=24),smoke_cap=32,costs=dict(Square=11360,Vector=11360,Exchange_standalone_max=4640),deferred=['DKD','A-floor','full-GSM8K','confirmation'],source_ledger=str(SPEC/'design.md'),source_ledger_sha256=sha(SPEC/'design.md'))
    freeze(ROOT/'manifest.json',manifest)
    freeze(ROOT/'manifest_identity.json',dict(schema=1,manifest_sha256=sha(ROOT/'manifest.json')))
    write(ROOT/'preparation_receipt.json',dict(gpu_used=False,model_loaded=False,legacy_import=imp['scores_at_import'],memory=mem,storage=storage,missing_generations=sum(100-imp['scores_at_import'][a]['total'] if a in imp['scores_at_import'] else 100 for a in ARMS)))
    return manifest

def validate():
    checked(ROOT/'manifest.json',read(ROOT/'manifest_identity.json')['manifest_sha256'])
    m=read(ROOT/'manifest.json')
    validate_schema(m,read(MULTI/'config.json'))
    if m['probe_rates']!=[.48,.52] or m['exchange']!=dict(d=41,rounds=3,proposals=8,max_new=24) or m['total_prunable']!=6979321856 or m['smoke_cap']!=32:raise ValueError('Scientific constants changed')
    if m['model']!=read(MULTI/'config.json')['model'] or m['evaluation']!=read(MULTI/'config.json')['evaluation']:raise ValueError('Model/evaluation changed')
    if list(sorted(m['arms']))!=sorted(ARMS) or m['target']!=3489660928:raise ValueError('Scientific matrix changed')
    for p,h in m['source_hashes'].items():checked(p,h)
    checked(MULTI/'config.json',m['legacy_config_sha256']);checked(MULTI/'requests.json',m['requests_sha256']);checked(ROOT/'legacy_import.json',m['import_sha256'])
    legacy=read(MULTI/'config.json')
    for p,h in legacy['sources'].items():checked(p,h)
    checked(m['legacy_teacher']['path'],m['legacy_teacher']['sha256'])
    for p,h in m['checkpoint_sha256'].items():checked(p,h)
    required={'bank','seed','edge_weights','reduction_denominators','source_hashes','matched_control','hypothesis','defaults_provenance'}
    if any(not required<=set(a) for a in m['arms'].values()):raise ValueError('Missing arm objective/control contract')
    if set(m['references'])!={'legacy_Uniform','legacy_A','legacy_AC'}:raise ValueError('Missing legacy reference')
    if len(m['checkpoint_map'])!=224:raise ValueError('Missing checkpoint mapping')
    for b in m['banks'].values():checked(b['path'],b['sha256'])
    for p,h in read(ROOT/'legacy_import.json')['sources'].items():checked(p,h)
    return m


def validate_schema(m,legacy):
    if set(m['arms'])!=set(ARMS):raise ValueError('Scientific arm matrix changed')
    required={'bank','seed','edge_weights','reduction_denominators','source_hashes','matched_control','hypothesis','defaults_provenance'}
    if any(not required<=set(a) for a in m['arms'].values()):raise ValueError('Missing arm objective/control contract')
    if any(not a['matched_control'] for a in m['arms'].values()):raise ValueError('Missing matched control')
    if set(m['references'])!={'legacy_Uniform','legacy_A','legacy_AC'}:raise ValueError('Missing legacy reference')
    if m['target']!=3489660928 or m['total_prunable']!=6979321856 or m['probe_rates']!=[.48,.52]:raise ValueError('Scientific budget/probe changed')
    if m['exchange']!=dict(d=41,rounds=3,proposals=8,max_new=24):raise ValueError('Exchange scope changed')
    if m['model']!=legacy['model'] or m['evaluation']!=legacy['evaluation']:raise ValueError('Model/evaluation changed')
