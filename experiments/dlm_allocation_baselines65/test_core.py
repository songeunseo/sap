"""Numerical parity against pinned public functions, plus fail-closed budget tests."""
import ast
import importlib.util
import math
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.dlm_allocation_baselines65 import core
from experiments.dlm_owl65.core import exact_row_counts

DATA=Path('/DATA/tmluser1/dlm_allocation_baselines65')
def definition(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    ns={'torch':torch,'nn':nn,'math':math,'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]

class Parity(unittest.TestCase):
    def test_lsa_public_metric(self):
        cls=definition(DATA/'sap-lsa-reference/layersp/blk.py','BlockWanda')
        torch.manual_seed(7);layer=nn.Linear(32,12,bias=False);x=torch.randn(80,32);H=x.T@x
        obj=cls(layer);obj.H=H.clone();old=layer.weight.clone()
        with torch.no_grad():
            a=obj.blk_s(block_size=8,s=.5);b=core.lsa_metric(layer.weight,H,8,.5)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(layer.weight,old));self.assertTrue(torch.equal(H,obj.H))

    def test_alpha_public_fit(self):
        f=definition(DATA/'sap-alphapruning-reference/lib/esd_utils.py','net_esd_estimator')
        torch.manual_seed(3);layer=nn.Linear(96,64,bias=False)
        expected=f(nn.Sequential(layer),fix_fingers='xmin_peak')
        actual=core.alpha_from_eigs(torch.tensor(expected['eigs'][0]))
        self.assertEqual(actual['alpha'],expected['alpha'][0]);self.assertEqual(actual['D'],expected['D'][0])

    def test_dsa_public_graph(self):
        p=DATA/'sap-dsa-reference/lib/autolayer.py';spec=importlib.util.spec_from_file_location('public_dsa',p)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        graph=mod.LayerEngine('W:(ABSLOG)-(VAR)-(ATAN,ASIN)-(7)')
        for x in (torch.exp(torch.linspace(-.2,.2,128)),torch.exp(torch.linspace(-5,5,128)),torch.ones(128)):
            self.assertEqual(float(graph.compute_importance(x.clone())),core.dsa_score(x)['value'])

    def test_invalid_not_repaired(self):
        with self.assertRaises(ValueError):core.dsa_rates([0.]*32)
        with self.assertRaises(ValueError):core.validate_rates(np.full(32,float('nan')))
        with self.assertRaises(ValueError):core.alpha_from_eigs(torch.zeros(16))

    def test_rates_and_exact_budget(self):
        import json
        p=Path('experiments/projection_capacity_allocation_65/candidate_mask_manifest.json')
        entries=json.loads(p.read_text())['entries'];weights=[r['weights'] for r in entries]
        rates=[core.dlp_rates(np.linspace(.1,1,32)),core.dsa_rates(np.linspace(.1,1,32)),
               core.alpha_rates(np.linspace(1.5,4,224),weights),core.lsa_rates(np.linspace(.1,1,224))]
        for s in rates:
            core.validate_rates(s);self.assertAlmostEqual(float(s.mean()),.65,places=6)
            counts,b=exact_row_counts(entries,s,4536008704)
            self.assertEqual(sum(k*r['shape'][0] for k,r in zip(counts,entries)),4536008704)
            self.assertEqual(b['budget_error'],0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA parity optional on CPU')
    def test_cuda_spectrum(self):
        torch.manual_seed(12);w=torch.randn(192,128)
        cpu=torch.linalg.svdvals(w);gpu=torch.linalg.svdvals(w.cuda(),driver='gesvd').cpu()
        torch.testing.assert_close(cpu,gpu,rtol=2e-5,atol=2e-5)

if __name__=='__main__':unittest.main()
