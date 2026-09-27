import random
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from experiments.dlm_allocation_sequential65.run import collect_block, wanda_mask, canonical, tuples
from lib.layerwrapper import WrappedGPT

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Linear(4,4,bias=False) for _ in range(7)])
    def forward(self,x):
        for projection in self.projections: x=x+projection(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(12,4)
        self.blocks=nn.ModuleList([Block(),Block()])
        self.model=SimpleNamespace(transformer=SimpleNamespace(blocks=self.blocks))
    def forward(self,ids):
        x=self.embedding(ids)
        for block in self.blocks: x=block(x)
        return x

class SequentialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model=Model().eval()
        self.mapping={f'block_{b:02d}.p{j}':m for b,block in enumerate(self.model.blocks) for j,m in enumerate(block.projections)}
        self.refs=[{'name':n} for n in self.mapping]
        self.states=[{'noisy_ids':[[1,2,3]]},{'noisy_ids':[[4,5,6]]}]
    def test_native_prefix_and_sparse_remeasurement(self):
        sums={n:torch.zeros(4) for n in list(self.mapping)[7:]}; handles=[]
        for name in sums:
            def hook(m,inp,out,n=name): sums[n].add_(inp[0].reshape(-1,4).detach().square().sum(0))
            handles.append(self.mapping[name].register_forward_hook(hook))
        with torch.no_grad():
            for state in self.states: self.model(torch.tensor(state['noisy_ids']))
        for h in handles: h.remove()
        measured=collect_block(self.model,self.mapping,self.refs,self.states,1)
        for name in sums: torch.testing.assert_close(measured[name],sums[name]/2,rtol=0,atol=0)
        with torch.no_grad():
            for module in self.model.blocks[0].projections: module.weight.zero_()
        sparse=collect_block(self.model,self.mapping,self.refs,self.states,1)
        self.assertGreater(float((sparse[list(sparse)[0]]-measured[list(sparse)[0]]).norm()),0)
        self.assertTrue(all(not m._forward_hooks for m in self.mapping.values()))
        self.assertTrue(all(not b._forward_hooks for b in self.model.blocks))
    def test_wanda_statistics_equal_original_wrappedgpt(self):
        measured=collect_block(self.model,self.mapping,self.refs,self.states,0)
        name=list(self.mapping)[0]; wrapper=WrappedGPT(self.mapping[name])
        with torch.no_grad():
            for state in self.states: wrapper.add_batch(self.model.embedding(torch.tensor(state['noisy_ids'])),None)
        torch.testing.assert_close(measured[name],wrapper.scaler_row)
        expected=torch.argsort(self.mapping[name].weight.abs()*wrapper.scaler_row.sqrt()[None,:],dim=1,stable=True)[:,:2]
        mask=wanda_mask(self.mapping[name].weight,measured[name],2)
        self.assertTrue((mask.sum(1)==2).all()); self.assertTrue(mask.gather(1,expected).all())
    def test_cleanup_on_failure(self):
        with self.assertRaises(KeyError): collect_block(self.model,self.mapping,self.refs,[{}],0)
        self.assertTrue(all(not m._forward_hooks for m in self.mapping.values()))
        self.assertTrue(all(not b._forward_hooks for b in self.model.blocks))
    def test_graph_dedup_and_rng_roundtrip(self):
        self.assertEqual(canonical('W:(ABS)-(VAR)-(LOG)-(8)'),canonical('W:(ABS)-(VAR)-(LOG)-(10)'))
        import json
        rng=random.Random(2); state=json.loads(json.dumps(rng.getstate())); expected=rng.random()
        rng.setstate(tuples(state)); self.assertEqual(rng.random(),expected)

if __name__=='__main__': unittest.main()
