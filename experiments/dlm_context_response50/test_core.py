import unittest
import numpy as np
import torch
from experiments.dlm_context_response50.core import make_pairs,log_odds,distortion,rank_rates
from experiments.dlm_owl65.core import exact_row_counts
class Tests(unittest.TestCase):
    def test_pair_and_seed(self):
        s=dict(noisy_ids=[[9,9,1,9,9,2,9,9]],clean_ids=[[3,4,1,5,6,2,7,8]],sequence_index=0,p_mask=.75)
        a=make_pairs([s],9);self.assertEqual(a,make_pairs([s],9))
        p=a[0];self.assertTrue(set(p['query']).isdisjoint(p['reveal']))
        for q in p['query']:self.assertEqual(p['before'][0][q],9);self.assertEqual(p['after'][0][q],9)
    def test_odds_invariance_and_extreme(self):
        z=torch.tensor([[100.,-100.,-99.],[1.,2.,3.]])
        y=[0,1];a=log_odds(z,y)
        torch.testing.assert_close(a,log_odds(z+1000,y),atol=1e-4,rtol=1e-5)
        self.assertTrue(torch.isfinite(a).all());self.assertGreater(a[0],190)
    def test_response_not_pointwise(self):
        d=torch.zeros(2,3)
        a=distortion(torch.ones(2,3),d);b=distortion(torch.tensor([[-1.]*3,[1.]*3]),d)
        self.assertEqual(a['A'],b['A']);self.assertEqual(a['C'],0);self.assertEqual(b['C'],4)
    def test_signed_ranks_and_budget(self):
        rates=rank_rates(np.arange(32)-16)
        self.assertGreater(rates[0],rates[-1]);self.assertAlmostEqual(rates.mean(),.5)
        np.testing.assert_allclose(rank_rates(np.ones(32)),.5)
        refs=[dict(name=f'block_{b:02d}.p{j}',shape=[8,64],weights=512) for b in range(32) for j in range(7)]
        counts,budget=exact_row_counts(refs,rates,32*7*512//2)
        self.assertEqual(sum(8*k for k in counts),32*7*512//2)
if __name__=='__main__':unittest.main()
