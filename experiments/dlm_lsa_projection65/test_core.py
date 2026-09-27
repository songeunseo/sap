import itertools
import unittest
import numpy as np
from experiments.dlm_lsa_projection65.core import projection_rates, exact_projection_rows
from experiments.dlm_lsa_projection65.run import public_mapping


class ProjectionTests(unittest.TestCase):
    def test_official_numeric_parity(self):
        metrics=np.random.default_rng(0).uniform(1,100,224)
        sizes=np.tile([4096*4096,4096*12288,4096*4096,4096*4096,4096*4096,12288*4096,12288*4096],32)
        actual=projection_rates(metrics,sizes)
        np.testing.assert_array_equal(actual,public_mapping(metrics,sizes))
        self.assertAlmostEqual(np.average(actual,weights=sizes),.65,places=6)
        permutation=np.random.default_rng(1).permutation(224)
        np.testing.assert_allclose(projection_rates(metrics[permutation],sizes[permutation]),actual[permutation],atol=2e-6)

    def test_exact_dp_vs_brute_force(self):
        rows=[dict(shape=[2,10]),dict(shape=[4,8]),dict(shape=[2,12])]
        rates=np.array([.56,.63,.71]);target=48
        counts,budget=exact_projection_rows(rows,rates,target)
        candidates=[]
        bases=[int(r['shape'][1]*s) for r,s in zip(rows,rates)]
        for ds in itertools.product((-1,0,1),repeat=3):
            ks=[b+d for b,d in zip(bases,ds)]
            if sum(k*r['shape'][0] for k,r in zip(ks,rows))==target:
                candidates.append(sum(r['shape'][0]*r['shape'][1]*(k/r['shape'][1]-s)**2 for k,r,s in zip(ks,rows,rates)))
        self.assertAlmostEqual(budget['weighted_squared_rate_error'],min(candidates))
        self.assertEqual(sum(k*r['shape'][0] for k,r in zip(counts,rows)),target)

    def test_invalid_not_clipped(self):
        with self.assertRaises(ValueError):projection_rates([1,1],[10,10])
        with self.assertRaises(ValueError):projection_rates([100,1],[1,100000])
        with self.assertRaises(ValueError):exact_projection_rows([dict(shape=[2,10])],[.5],11)


if __name__=='__main__':unittest.main()
