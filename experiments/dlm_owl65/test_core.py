import itertools
import unittest
import numpy as np
from experiments.dlm_owl65.core import owl_sparsities,exact_row_counts


class Tests(unittest.TestCase):
    def test_official_formula(self):
        d=np.array([1.,3.,8.,2.]);lam=.08;s=.65
        x=((d-d.min())*(1/(d.max()-d.min())*lam*2))
        expected=1-(x-np.mean(x)+(1-s))
        np.testing.assert_array_equal(owl_sparsities(d,s,lam),expected)
        self.assertAlmostEqual(expected.mean(),s)
        self.assertLess(expected[2],expected[0])

    def test_scale_invariant(self):
        np.testing.assert_allclose(owl_sparsities([1,3,6]),owl_sparsities([100,300,600]))

    def test_flat_stops(self):
        with self.assertRaises(ValueError):owl_sparsities([1,1,1])

    def test_exact_grouped_budget_and_optimality(self):
        entries=[dict(name=f'block_00.p{i}',shape=[2,10 if i<6 else 30]) for i in range(7)]
        s=np.array([.651]);target=2*(6*6+19)
        counts,receipt=exact_row_counts(entries,s,target)
        self.assertEqual(sum(k*e['shape'][0] for k,e in zip(counts,entries)),target)
        self.assertEqual(len(set(counts[:6])),1)
        feasible=[]
        for a,b in itertools.product((5,6,7),(18,19,20)):
            if 2*(6*a+b)==target:feasible.append(12*10*(a/10-s[0])**2+2*30*(b/30-s[0])**2)
        self.assertAlmostEqual(receipt['weighted_squared_rate_error'],min(feasible))

    def test_real_shapes(self):
        shapes=[[4096,4096],[4096,12288],[4096,4096],[4096,4096],[4096,4096],[12288,4096],[12288,4096]]
        e=[dict(name=f'block_{b:02d}.p{i}',shape=v) for b in range(32) for i,v in enumerate(shapes)]
        s=owl_sparsities(np.arange(32)**2+1)
        counts,r=exact_row_counts(e,s,4536008704)
        self.assertEqual(r['budget_error'],0)
        self.assertTrue(all(abs(k-int(v['shape'][1]*s[i//7]))<=1 for i,(k,v) in enumerate(zip(counts,e))))


if __name__=='__main__':unittest.main()
