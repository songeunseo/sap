import unittest
import torch
from experiments.dlm_ppl50_uniform_layer.run import exact_global_masks,freeze,TARGET,read,seq
class TestLayerGlobal(unittest.TestCase):
 def test_exact_and_ties(self):
  a=torch.tensor([[1.,1.],[0.,3.]])
  b=torch.tensor([[2.,1.,5.,4.]])
  masks,t,k=exact_global_masks([a,b])
  self.assertEqual(k,4);self.assertEqual(sum(int(x.sum()) for x in masks),4)
  self.assertTrue(torch.equal(masks[0],torch.tensor([[True,True],[True,False]])))
  self.assertTrue(masks[1][0,1])
 def test_budget(self):
  cfg=freeze();refs=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
  self.assertEqual(sum(r['weights'] for r in refs)//2,TARGET)
  self.assertTrue(all(sum(r['weights'] for r in refs[b*7:(b+1)*7])%2==0 for b in range(32)))
if __name__=='__main__':unittest.main()
