import ast
import random
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from experiments.dlm_ppl50 import run
from experiments.dlm_ppl50.evo import fast_obc, transfers
from experiments.dlm_lsa_projection65.core import projection_rates

class Checks(unittest.TestCase):
    def test_lsa_official50(self):
        upstream=run.baseline.DATA/'sap-lsa-reference/layersp/blk.py'
        metrics=[r['metric'] for r in run.read(run.baseline.ROOT/'lsa/statistics.json')['rows']]
        refs=run.read(run.seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
        tree=ast.parse(upstream.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='blk_score_global')
        nodes=[n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'layer_imp','layer_prune_numel','all_layer_ratio'} for t in n.targets) and not isinstance(n.value,ast.List)]
        env=dict(torch=torch,args=SimpleNamespace(final_s=.5,Lamda=.07),all_layer_metric=torch.tensor(metrics).reshape(32,7),all_layer_numel=torch.tensor([r['weights'] for r in refs]).reshape(32,7))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(upstream),'exec'),env)
        self.assertTrue(np.array_equal(projection_rates(metrics,[r['weights'] for r in refs],target=.5),env['all_layer_ratio'].reshape(-1).double().numpy()))
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='alpha' for t in n.targets) and isinstance(n.value,ast.List))
        self.assertEqual(ast.literal_eval(assignment.value)[5],.04)
    def test_evo_compensation(self):
        torch.manual_seed(17)
        layer=torch.nn.Linear(16,4,bias=False)
        original=layer.weight.detach().clone()
        handle=fast_obc()(layer,rel_damp=.01,block_size=8)
        handle.update(torch.randn(1,32,16))
        values=handle.prune([.25,.5,.75])
        self.assertEqual([int((w==0).sum()) for w in values],[16,32,48])
        for w in values:
            self.assertTrue(torch.isfinite(w).all())
            self.assertFalse(torch.equal(w[w!=0],original[w!=0]))
        self.assertTrue(torch.equal(layer.weight,original))
    def test_evo_budget_and_bounds(self):
        rng=random.Random(0);parent=[0]*224
        for _ in range(5000):
            parent=transfers(parent,rng,{'num_levels':8})
            self.assertEqual(sum(parent),0)
            self.assertLessEqual(max(map(abs,parent)),8)
    def test_allocations50(self):
        cfg=run.freeze()
        refs=run.read(run.seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
        for method,alloc in cfg['allocations'].items():
            self.assertEqual(sum(k*r['shape'][0] for k,r in zip(alloc['row_counts'],refs)),run.TARGET,method)
            self.assertEqual(len(alloc['row_counts']),224)

if __name__=='__main__':unittest.main()
