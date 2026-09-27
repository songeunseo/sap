"""Independent aggregate checks for the exact-mask pass."""
import csv
from pathlib import Path
import numpy as np
from experiments.dlm_distribution_hypotheses.analyze import ROOT,OLD,read,sha,write

d=read(ROOT/'mask_audit.json');rows=list(csv.DictReader((ROOT/'mask_audit.csv').open()))
assert len(rows)==448
assert d['script_sha256']==sha(ROOT/'audit_masks.py')
for p,h in d['manifest_sha256'].items():assert sha(Path(p))==h
stats={r['name']:r for r in read(OLD/'statistics.json')['rows']}
for method,s in d['summary'].items():
    rr=[r for r in rows if r['method']==method]
    assert len(rr)==len({r['name'] for r in rr})==224
    assert sum(int(r['pruned']) for r in rr)==3489660928
    assert sum(int(r['weights']) for r in rr)==6979321856
    assert sum(int(r['tail_count']) for r in rr)==s['total_tail_count']==38963987
    assert sum(int(r['tail_pruned']) for r in rr)==s['total_tail_pruned']==0
    for r in rr:
        st=stats[r['name']]['score_dense']
        assert np.isclose(int(r['tail_count'])/int(r['weights']),st['outlier5_fraction'],atol=1e-12,rtol=0)
        assert np.isclose(float(r['tail_diagonal_energy_fraction']),st['outlier5_energy_fraction'],atol=2e-6)
        assert 0<=float(r['removed_diagonal_energy_fraction'])<=1
    for typ,metrics in s['per_type'].items():
        for metric,values in metrics.items():
            check=[np.mean([float(r[metric]) for r in rr if r['type']==typ and int(r['layer'])//8==q]) for q in range(4)]
            assert np.allclose(check,values,rtol=0,atol=1e-12)
write('mask_verification.json',dict(status='passed',rows=448,exact_budget_each=3489660928,
    tail_count_each=38963987,tail_removed_each=0,summary_recomputed=True,
    original_full_tensor_tail_count_energy_reproduced=True,
    output_sha256={p.name:sha(p) for p in ROOT.iterdir() if p.suffix in ['.json','.csv','.png','.md'] and p.name!='mask_verification.json'},
    script_sha256=sha(Path(__file__))))
print('PASS:448 masks; exact budgets; zero outlier removal; all quartile summaries independently reproduced.')
