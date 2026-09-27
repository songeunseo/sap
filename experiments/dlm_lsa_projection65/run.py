"""One official projection-wise LSA variant; no search or historical edits."""
import argparse
import ast
import fcntl
import importlib.util
from pathlib import Path
import numpy as np
import torch
from experiments.dlm_allocation_sequential65 import run as seq
from experiments.dlm_allocation_baselines65 import run as baseline
from experiments.dlm_lsa_projection65.core import projection_rates, exact_projection_rows

ROOT = Path(__file__).resolve().parent
UPSTREAM = baseline.DATA / 'sap-lsa-reference/layersp/blk.py'
STAT = baseline.ROOT / 'lsa/statistics.json'
METHOD = 'lsac'
read,write,sha = seq.read,seq.write,seq.sha


def public_mapping(metrics, sizes):
    """Execute ONLY the original global-mapping assignments for parity checking."""
    tree = ast.parse(UPSTREAM.read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'blk_score_global')
    statements = []
    for node in function.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
            {'layer_imp','layer_prune_numel','all_layer_ratio'} for t in node.targets):
            # Exclude all_layer_ratio=[] initialization earlier in this function.
            if not isinstance(node.value,ast.List):
                statements.append(node)
    from types import SimpleNamespace
    env = dict(torch=torch,all_layer_metric=torch.tensor(metrics,dtype=torch.float32).reshape(32,7),
               all_layer_numel=torch.tensor(sizes,dtype=torch.int64).reshape(32,7),
               args=SimpleNamespace(final_s=.65,Lamda=.07))
    exec(compile(ast.Module(body=statements,type_ignores=[]),str(UPSTREAM),'exec'),env)
    return env['all_layer_ratio'].reshape(-1).double().numpy()


def validate():
    cfg=read(ROOT/'config.json')
    for path,digest in cfg['sources'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen source changed: '+path)
    return cfg


def freeze():
    if (ROOT/'config.json').exists():
        return validate()
    parent=seq.validate(); baseline.validate()
    ref=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
    saved=read(STAT); bcfg=baseline.ROOT/'lsa/config.json'
    if saved['config_sha256'] != sha(bcfg):
        raise RuntimeError('Cached LSA statistics provenance mismatch')
    if [r['name'] for r in saved['rows']] != [r['name'] for r in ref] or len(ref)!=224:
        raise RuntimeError('Projection ordering mismatch')
    metrics=[r['metric'] for r in saved['rows']]; sizes=[r['weights'] for r in ref]
    rates=projection_rates(metrics,sizes)
    expected=public_mapping(metrics,sizes)
    if not np.array_equal(rates,expected):
        raise RuntimeError('Official lsac mapping parity failed')
    counts,budget=exact_projection_rows(ref,rates,parent['pruned'])
    allocation=dict(ideal_projection_sparsities=rates.tolist(),row_counts=counts,budget=budget,
                    actual_global_sparsity=parent['pruned']/parent['weights'])
    sources=dict(parent['sources'])
    sources.update({str(p):sha(p) for p in [STAT,bcfg,baseline.ROOT/'config.json',UPSTREAM,
        seq.ROOT/'config.json',seq.ROOT/'uniform/predictions.jsonl',seq.ROOT/'lsa/predictions.jsonl',
        *ROOT.glob('*.py'),ROOT/'README.md',ROOT/'run.sh']})
    cfg=dict(model=parent['model'],target=.65,pruned=parent['pruned'],weights=parent['weights'],
        protocol_hash=parent['protocol_hash'],sources=sources,allocations={METHOD:allocation},
        method='official lsac projection-wise allocation + native sequential Standard Wanda',
        upstream_commit=baseline.UPSTREAM['lsa'][1],lam=.07,probe_sparsity=.5,group_size=128,
        calibration_states=80,mini_examples=100,
        primary='User explicitly requested historical GSM8K mini100; no automatic PPL/full/search',
        comparison='Sequential Uniform and basic LSA; granularity AND official lambda differ (.07 vs .1)',
        parity='224 official projection rates bit-exact; full matrices and parameter counts, no abs or averaging',
        supplementary='Reuse sequential runner heldout40 masked CE diagnostic; not candidate selection')
    seq.old.frozen(ROOT/'config.json',cfg)
    write(ROOT/'allocation.json',allocation)
    write(ROOT/'allocation_summary.json',dict(min=float(rates.min()),max=float(rates.max()),
        mean=float(rates.mean()),parameter_weighted=float(np.average(rates,weights=sizes)),budget=budget,
        per_type={t:float(np.mean([rates[i] for i,r in enumerate(ref) if r['name'].split('.')[1]==t]))
                  for t in {r['name'].split('.')[1] for r in ref}}))
    print('Frozen lsac:',read(ROOT/'allocation_summary.json'),flush=True)
    return cfg


def harness():
    spec=importlib.util.spec_from_file_location('lsac_sequential_harness',seq.__file__)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.ROOT=ROOT;h.STORE=Path('/DATA/tmluser1/dlm-lsa-projection65');h.validate=validate
    h.summarize=summarize
    return h


def summarize():
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    cfg=validate(); predictions=seq.old.read_jsonl(ROOT/METHOD/'predictions.jsonl')
    result={}
    for method in ('uniform','lsa'):
        reference=seq.old.read_jsonl(seq.ROOT/method/'predictions.jsonl')
        _validate_rows(predictions,reference,cfg['protocol_hash'])
        result[method]=paired_binary_comparison([r['correct'] for r in reference],[r['correct'] for r in predictions])
    write(ROOT/'comparisons.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run'])
    args=parser.parse_args()
    if args.phase=='freeze':freeze()
    else:
        (ROOT/METHOD).mkdir(parents=True,exist_ok=True)
        with (ROOT/METHOD/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            h=harness()
            try:h.run(METHOD)
            except BaseException as exc:
                h.event(METHOD,'failed',error=f'{type(exc).__name__}: {exc}')
                raise
